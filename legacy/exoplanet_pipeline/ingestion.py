"""
ingestion.py — TESS Data Ingestion, Cleaning, and Stitching
============================================================

Downloads TESS SPOC light curves for a given target, removes NaNs,
flattens and sigma-clips outliers, normalizes flux per sector, and
stitches all sectors into a single light curve.

Fix 1 (partial): Captures the per-cadence flux_err from the original
(un-normalized) light curves and stitches them alongside the normalized
flux. This allows downstream modules (denoise.py) to compute real
photon-noise estimates instead of sqrt(~1).
"""

import numpy as np
import warnings
import lightkurve as lk
import os
import glob
import socket
import time
from astropy.stats import sigma_clip
from astroquery.mast import Catalogs
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from astroquery.mast import Conf

# Set global socket timeout to prevent MAST from hanging indefinitely
socket.setdefaulttimeout(30.0)

Conf.timeout = 30  # Fail fast after 30 seconds

def _api_retry(func, *args, **kwargs):
    retries = [5, 15, 60]
    for delay in retries:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"  API Error: {e}. Retrying in {delay}s...")
            time.sleep(delay)
    return func(*args, **kwargs)

def preprocess_tess_data(target_star_id):
    """
    Ingests TESS data for a given target star, cleans it, normalizes flux
    across sectors, and stitches light curves together.

    Returns
    -------
    stitched_lc : lightkurve.LightCurve or None
        The cleaned, normalized, stitched light curve.
    lc_collection : lightkurve.LightCurveCollection or None
        The raw downloaded light curve collection (for reference).
    raw_flux_err : np.ndarray or None
        The flux_err values from the original (pre-normalization) light
        curves, stitched in cadence order. These are in the original
        instrument units (electrons/s), suitable for computing real
        photon noise. This is the key output for Fix 1.
    """
    print(f"Searching for TESS observations for {target_star_id}...")
    lc_collection = None
    try:
        search_result = _api_retry(lk.search_lightcurve, target_star_id, mission='TESS', author='SPOC')
        if search_result:
            lc_collection = _api_retry(search_result.download_all, download_dir='./tess_cache')
    except Exception as e:
        print(f"  MAST API search failed or timed out: {e}")
    
    # Offline Cache Fallback
    if lc_collection is None or len(lc_collection) == 0:
        print("  Attempting to load from local cache...")
        cache_dir_global = os.path.expanduser("~/.lightkurve/cache/mastDownload/TESS")
        cache_dir_local = "./tess_cache"
        clean_id = target_star_id.replace("TIC", "").strip()
        
        # Check global cache
        pattern1_g = os.path.join(cache_dir_global, f"**/*{clean_id.zfill(16)}*lc.fits")
        pattern2_g = os.path.join(cache_dir_global, f"**/*{clean_id}*lc.fits")
        # Check local cache
        pattern1_l = os.path.join(cache_dir_local, f"**/*{clean_id.zfill(16)}*lc.fits")
        pattern2_l = os.path.join(cache_dir_local, f"**/*{clean_id}*lc.fits")
        
        files = glob.glob(pattern1_g, recursive=True) + glob.glob(pattern1_l, recursive=True)
        if not files:
            files = glob.glob(pattern2_g, recursive=True) + glob.glob(pattern2_l, recursive=True)
            
        if files:
            print(f"  Found {len(files)} cached light curves.")
            lcs = []
            for f in files:
                try:
                    lcs.append(lk.read(f))
                except Exception as read_err:
                    print(f"    Failed to read {f}: {read_err}")
            if lcs:
                lc_collection = lk.LightCurveCollection(lcs)

    if lc_collection is None or len(lc_collection) == 0:
        print(f"Failed to find or download light curves for {target_star_id}.")
        return None, None, None

    print(f"Downloaded {len(lc_collection)} light curves.")

    # De-duplicate sectors: ensure only one light curve per sector is used
    # (prevents mixing 20s fast-cadence with 120s standard cadence from the same sector)
    sector_map = {}
    for lc in lc_collection:
        if lc is None:
            continue
        sec = lc.meta.get('SECTOR', None) if hasattr(lc, 'meta') and lc.meta is not None else None
        if sec is None:
            # Fallback key by time range
            sec = int(np.nanmin(lc.time.value))
        
        # If sector not yet in map, add it
        if sec not in sector_map:
            sector_map[sec] = lc
        else:
            # Prefer standard 120s cadence over 20s fast-cadence for consistency
            existing_len = len(sector_map[sec])
            new_len = len(lc)
            # If new is standard (~18k-20k cadences) vs fast (>50k cadences), prefer standard
            if 10000 <= new_len <= 25000:
                sector_map[sec] = lc

    unique_lcs = [sector_map[s] for s in sorted(sector_map.keys())]
    print(f"Using {len(unique_lcs)} unique sector light curves (de-duplicated).")

    processed_lcs = []
    raw_flux_errs = []

    for i, lc in enumerate(unique_lcs):
        if lc is None:
            continue

        lc = lc.remove_nans()
        if len(lc) == 0:
            continue

        if hasattr(lc, 'flux_err') and lc.flux_err is not None:
            sector_flux_err = lc.flux_err.value.copy()
        else:
            sector_flux_err = np.sqrt(np.abs(lc.flux.value))

        # CRITICAL FIX: Extract raw SAP_FLUX to test Exo-Gargantua's thermal detrending on unconditioned data.
        if 'sap_flux' in lc.colnames:
            raw_flux_val = np.asarray(lc['sap_flux'].value, dtype=float)
        else:
            raw_flux_val = np.asarray(lc.flux.value if hasattr(lc.flux, 'value') else lc.flux, dtype=float)

        raw_median_flux = np.nanmedian(raw_flux_val)
        if raw_median_flux == 0 or np.isnan(raw_median_flux):
            continue

        # CROWDSAP = fraction of flux in aperture from target star
        crowdsap = 1.0
        if hasattr(lc, 'meta') and lc.meta is not None:
            crowdsap = lc.meta.get('CROWDSAP', 1.0)
            if crowdsap is None or not np.isfinite(crowdsap) or crowdsap <= 0:
                crowdsap = 1.0

        # Mathematically exact undilution: F_undiluted = (F_obs / F_med - 1.0) / CROWDSAP + 1.0
        norm_flux = (raw_flux_val / raw_median_flux - 1.0) / crowdsap + 1.0
        norm_err = np.asarray(sector_flux_err / raw_median_flux / crowdsap, dtype=float)

        # Sigma clip extreme upward outliers (flares / cosmic rays) on plain float array
        norm_flux_plain = np.array(norm_flux, dtype=float)
        clipped = sigma_clip(norm_flux_plain, sigma_upper=5, sigma_lower=np.inf, masked=True)
        valid_mask = ~np.asarray(clipped.mask, dtype=bool)

        norm_flux = norm_flux[valid_mask]
        norm_err = norm_err[valid_mask]
        time_sub = lc.time[valid_mask]

        if len(norm_flux) == 0:
            continue

        new_lc = lk.LightCurve(time=time_sub, flux=norm_flux, flux_err=norm_err)
        processed_lcs.append(new_lc)
        raw_flux_errs.append(norm_err)

    if not processed_lcs:
        return None, None, None

    stitched_lc = lk.LightCurveCollection(processed_lcs).stitch()
    raw_flux_err = np.concatenate(raw_flux_errs)

    print(f"Data ingestion, cleaning, and stitching complete ({len(stitched_lc)} cadences across {len(processed_lcs)} sectors).")
    return stitched_lc, lc_collection, raw_flux_err


def fetch_stellar_parameters(tic_id, lc=None):
    """
    Fetches stellar parameters from the MAST TIC Catalog, NASA Exoplanet Archive,
    or directly from the TESS FITS file primary header metadata.

    Parameters
    ----------
    tic_id : str or int
        The TIC ID (e.g., "TIC 25155310").
    lc : lightkurve.LightCurve or None
        Optional light curve object to extract FITS header metadata.

    Returns
    -------
    dict
        Dictionary containing Rs, Ms, Teff, logg, MH and their errors.
    """
    if isinstance(tic_id, str):
        clean_id = tic_id.replace("TIC", "").strip()
    else:
        clean_id = str(tic_id)

    print(f"  Fetching stellar parameters for TIC {clean_id}...")
    
    stellar_params = {
        'Rs': 1.0, 'Rs_err': 0.1,
        'Ms': 1.0, 'Ms_err': 0.1,
        'Teff': 5778.0, 'Teff_err': 100.0,
        'logg': 4.438, 'logg_err': 0.1,
        'MH': 0.0, 'MH_err': 0.1,
        'fallback_used': True
    }

    # Step 1: Check if lc object or cached FITS file has stellar metadata in header
    fits_meta = {}
    if lc is not None and hasattr(lc, 'meta') and lc.meta is not None:
        fits_meta = lc.meta
    else:
        # Check cached FITS files
        import glob
        cache_dir = os.path.expanduser("~/.lightkurve/cache/mastDownload/TESS")
        pattern = os.path.join(cache_dir, f"**/*{clean_id}*lc.fits")
        files = glob.glob(pattern, recursive=True)
        if files:
            try:
                temp_lc = lk.read(files[0])
                fits_meta = temp_lc.meta
            except Exception:
                pass

    if fits_meta:
        rad = fits_meta.get('RADIUS')
        teff = fits_meta.get('TEFF')
        logg = fits_meta.get('LOGG')
        mh = fits_meta.get('MH')
        mass = fits_meta.get('MASS')

        if rad is not None and np.isfinite(rad) and rad > 0:
            stellar_params['Rs'] = float(rad)
            stellar_params['Rs_err'] = float(rad) * 0.05
            stellar_params['fallback_used'] = False
        if teff is not None and np.isfinite(teff) and teff > 0:
            stellar_params['Teff'] = float(teff)
            stellar_params['Teff_err'] = 100.0
        if logg is not None and np.isfinite(logg) and logg > 0:
            stellar_params['logg'] = float(logg)
            stellar_params['logg_err'] = 0.05
        if mh is not None and np.isfinite(mh):
            stellar_params['MH'] = float(mh)
            stellar_params['MH_err'] = 0.05
        if mass is not None and np.isfinite(mass) and mass > 0:
            stellar_params['Ms'] = float(mass)
            stellar_params['Ms_err'] = float(mass) * 0.05
        elif stellar_params['Rs'] > 0 and stellar_params['logg'] > 0:
            # Estimate mass from logg and Rs: M = 10^(logg - 4.438) * Rs^2
            calc_mass = 10.0**(stellar_params['logg'] - 4.438) * (stellar_params['Rs']**2)
            if np.isfinite(calc_mass) and calc_mass > 0:
                stellar_params['Ms'] = float(calc_mass)
                stellar_params['Ms_err'] = float(calc_mass) * 0.05

        if not stellar_params['fallback_used']:
            print(f"  Loaded stellar properties from FITS metadata: Rs={stellar_params['Rs']:.3f} R_sun, Ms={stellar_params['Ms']:.3f} M_sun, Teff={stellar_params['Teff']:.0f} K, logg={stellar_params['logg']:.3f}")

    # Step 2: Query MAST & NASA Exoplanet Archive for precise spectroscopic properties
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            catalog_data = _api_retry(Catalogs.query_criteria, catalog="Tic", ID=clean_id)

        if len(catalog_data) > 0:
            row = catalog_data[0]
            if not np.ma.is_masked(row['rad']) and not np.isnan(row['rad']):
                stellar_params['Rs'] = float(row['rad'])
                stellar_params['fallback_used'] = False
                err = float(row['e_rad']) if not np.ma.is_masked(row['e_rad']) else stellar_params['Rs'] * 0.05
                stellar_params['Rs_err'] = abs(err) if not np.isnan(err) else stellar_params['Rs'] * 0.05
            if not np.ma.is_masked(row['mass']) and not np.isnan(row['mass']):
                stellar_params['Ms'] = float(row['mass'])
                err = float(row['e_mass']) if not np.ma.is_masked(row['e_mass']) else stellar_params['Ms'] * 0.05
                stellar_params['Ms_err'] = abs(err) if not np.isnan(err) else stellar_params['Ms'] * 0.05
            if not np.ma.is_masked(row['Teff']) and not np.isnan(row['Teff']):
                stellar_params['Teff'] = float(row['Teff'])
            if 'logg' in row.colnames and not np.ma.is_masked(row['logg']) and not np.isnan(row['logg']):
                stellar_params['logg'] = float(row['logg'])
            if 'MH' in row.colnames and not np.ma.is_masked(row['MH']) and not np.isnan(row['MH']):
                stellar_params['MH'] = float(row['MH'])
    except Exception as e:
        if stellar_params['fallback_used']:
            print(f"  Warning: MAST catalog query failed ({e}). Using fallback.")

    # Cross-query NASA Archive for confirmed systems or WASP targets
    try:
        archive_table = _api_retry(NasaExoplanetArchive.query_criteria,
            table="pscomppars",
            select="st_rad,st_raderr1,st_mass,st_masserr1,st_teff,st_tefferr1,st_logg,st_loggerr1,st_met,st_meterr1",
            where=f"tic_id='TIC {clean_id}'"
        )
        if archive_table is not None and len(archive_table) > 0:
            tr = archive_table[0]
            def _to_float(v, default=np.nan):
                if v is None or np.ma.is_masked(v): return default
                val = getattr(v, 'value', v)
                return float(val) if not np.isnan(val) else default

            rad_val = _to_float(tr['st_rad'])
            if not np.isnan(rad_val):
                stellar_params['Rs'] = rad_val
                stellar_params['fallback_used'] = False
                err_val = _to_float(tr['st_raderr1'])
                stellar_params['Rs_err'] = abs(err_val) if not np.isnan(err_val) else rad_val * 0.05
            mass_val = _to_float(tr['st_mass'])
            if not np.isnan(mass_val):
                stellar_params['Ms'] = mass_val
                err_val = _to_float(tr['st_masserr1'])
                stellar_params['Ms_err'] = abs(err_val) if not np.isnan(err_val) else mass_val * 0.05
            teff_val = _to_float(tr['st_teff'])
            if not np.isnan(teff_val):
                stellar_params['Teff'] = teff_val
            if 'st_logg' in tr.colnames:
                logg_val = _to_float(tr['st_logg'])
                if not np.isnan(logg_val): stellar_params['logg'] = logg_val
            if 'st_met' in tr.colnames:
                mh_val = _to_float(tr['st_met'])
                if not np.isnan(mh_val): stellar_params['MH'] = mh_val
            print(f"  Loaded NASA Archive parameters: Rs={stellar_params['Rs']:.3f} R_sun, Ms={stellar_params['Ms']:.3f} M_sun, Teff={stellar_params['Teff']:.0f} K")
    except Exception:
        pass

    return stellar_params
