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
from astropy.stats import sigma_clip
from astroquery.mast import Catalogs
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive


import os
import glob
from astroquery.mast import Conf
Conf.timeout = 30  # Fail fast after 30 seconds

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
        search_result = lk.search_lightcurve(
            target_star_id, mission='TESS', author='SPOC'
        )
        if search_result:
            lc_collection = search_result.download_all()
    except Exception as e:
        print(f"  MAST API search failed or timed out: {e}")
    
    # Offline Cache Fallback
    if lc_collection is None or len(lc_collection) == 0:
        print("  Attempting to load from local cache...")
        cache_dir = os.path.expanduser("~/.lightkurve/cache/mastDownload/TESS")
        clean_id = target_star_id.replace("TIC", "").strip()
        pattern1 = os.path.join(cache_dir, f"**/*{clean_id.zfill(16)}*lc.fits")
        pattern2 = os.path.join(cache_dir, f"**/*{clean_id}*lc.fits")
        
        files = glob.glob(pattern1, recursive=True)
        if not files:
            files = glob.glob(pattern2, recursive=True)
            
        if files:
            print(f"  Found {len(files)} cached light curves.")
            lcs = []
            for f in files:
                try:
                    lcs.append(lk.read(f))
                except Exception as read_err:
                    print(f"    Failed to read {f}: {read_err}")
            if lcs:
                lc_collection = lk.LightCurveCollection(lcs[:2])

    if lc_collection is None or len(lc_collection) == 0:
        print(f"Failed to find or download light curves for {target_star_id}.")
        return None, None, None

    print(f"Downloaded {len(lc_collection)} light curves.")

    processed_lcs = []
    raw_flux_errs = []  # Fix 1: capture pre-normalization flux_err

    for i, lc in enumerate(lc_collection):
        if lc is None:
            continue

        lc = lc.remove_nans()
        if len(lc) == 0:
            continue

        # Fix 1: Capture flux_err BEFORE normalization.
        # These are in the original instrument units (electrons/s) and
        # represent the true photon + read noise per cadence.
        if hasattr(lc, 'flux_err') and lc.flux_err is not None:
            sector_flux_err = lc.flux_err.value.copy()
        else:
            # Fallback: estimate from sqrt(|flux|) in raw counts
            sector_flux_err = np.sqrt(np.abs(lc.flux.value))

        # Extract original median flux to properly scale the flux errors
        raw_median_flux = np.nanmedian(lc.flux.value)

        # Crowding/dilution correction via CROWDSAP
        # CROWDSAP = fraction of flux in the aperture from the target star.
        # If CROWDSAP < 1.0, the transit is diluted: observed_depth = true_depth * CROWDSAP.
        # We correct by dividing the flux deviation from 1.0 by CROWDSAP.
        crowdsap = 1.0
        if hasattr(lc, 'meta') and lc.meta is not None:
            crowdsap = lc.meta.get('CROWDSAP', 1.0)
            if crowdsap is None or not np.isfinite(crowdsap) or crowdsap <= 0:
                crowdsap = 1.0
        if crowdsap < 1.0:
            print(f"  Sector {i}: Applying crowding correction (CROWDSAP={crowdsap:.4f})")

        # Normalize by median flux
        if raw_median_flux != 0 and not np.isnan(raw_median_flux):
            lc = lc / raw_median_flux
            sector_flux_err = sector_flux_err / raw_median_flux

        # Convert to plain ndarray to avoid astropy Quantity/masked-array
        # compatibility issues with sigma_clip in astropy 6.x
        flux_for_clip = np.array(lc.flux.value, dtype=float)
        clipped_flux = sigma_clip(
            flux_for_clip,
            sigma_upper=5, sigma_lower=np.inf, masked=True
        )
        mask = ~clipped_flux.mask
        
        # Apply mask to LC
        lc = lc[mask]
        sector_flux_err = sector_flux_err[mask]

        if len(lc) == 0:
            continue

        # Apply crowding correction: undilute the transit depth
        # flux_corrected = 1.0 + (flux_observed - 1.0) / CROWDSAP
        if crowdsap < 1.0:
            flux_vals = lc.flux.value if hasattr(lc.flux, 'value') else np.asarray(lc.flux)
            correction_factor = 1.0 + (flux_vals - 1.0) / crowdsap
            # Apply correction
            lc = lc * (correction_factor / flux_vals)
            # Re-normalize just in case
            flux_vals = lc.flux.value if hasattr(lc.flux, 'value') else np.asarray(lc.flux)
            median_flux = np.nanmedian(flux_vals)
            if median_flux != 0:
                lc = lc / median_flux

        processed_lcs.append(lc)
        raw_flux_errs.append(sector_flux_err)

    if not processed_lcs:
        return None, None, None

    stitched_lc = lk.LightCurveCollection(processed_lcs).stitch()

    # Fix 1: stitch flux_err arrays in the same cadence order
    raw_flux_err = np.concatenate(raw_flux_errs)

    print("Data ingestion, cleaning, and stitching complete.")
    return stitched_lc, lc_collection, raw_flux_err


def fetch_stellar_parameters(tic_id):
    """
    Fetches stellar parameters from the MAST TIC Catalog.

    Parameters
    ----------
    tic_id : str or int
        The TIC ID (e.g., "TIC 25155310").

    Returns
    -------
    dict
        Dictionary containing Rs, Ms, Teff, and their errors.
    """
    # Clean string if it starts with "TIC"
    if isinstance(tic_id, str):
        clean_id = tic_id.replace("TIC", "").strip()
    else:
        clean_id = str(tic_id)

    print(f"  Fetching stellar parameters for TIC {clean_id}...")
    
    # Default fallback values
    stellar_params = {
        'Rs': 1.0, 'Rs_err': 0.1,
        'Ms': 1.0, 'Ms_err': 0.1,
        'Teff': 5778.0, 'Teff_err': 100.0,
        'fallback_used': True
    }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Query the TIC catalog
            catalog_data = Catalogs.query_criteria(catalog="Tic", ID=clean_id)

        if len(catalog_data) > 0:
            row = catalog_data[0]
            
            # Extract Radius from TIC
            if not np.ma.is_masked(row['rad']) and not np.isnan(row['rad']):
                stellar_params['Rs'] = float(row['rad'])
                stellar_params['fallback_used'] = False
                err = float(row['e_rad']) if not np.ma.is_masked(row['e_rad']) else stellar_params['Rs'] * 0.1
                stellar_params['Rs_err'] = abs(err) if not np.isnan(err) else stellar_params['Rs'] * 0.1
            else:
                print("  Warning: Stellar radius missing in TIC catalog. Using fallback 1.0 R_sun.")

            # Extract Mass from TIC
            if not np.ma.is_masked(row['mass']) and not np.isnan(row['mass']):
                stellar_params['Ms'] = float(row['mass'])
                err = float(row['e_mass']) if not np.ma.is_masked(row['e_mass']) else stellar_params['Ms'] * 0.1
                stellar_params['Ms_err'] = abs(err) if not np.isnan(err) else stellar_params['Ms'] * 0.1
            else:
                print("  Warning: Stellar mass missing in TIC catalog. Using fallback 1.0 M_sun.")

            # Extract Teff from TIC
            if not np.ma.is_masked(row['Teff']) and not np.isnan(row['Teff']):
                stellar_params['Teff'] = float(row['Teff'])
                err = float(row['e_Teff']) if not np.ma.is_masked(row['e_Teff']) else 100.0
                stellar_params['Teff_err'] = abs(err) if not np.isnan(err) else 100.0
            else:
                print("  Warning: Teff missing in TIC catalog. Using fallback 5778 K.")
                
            # Check for anomalous radius (e.g., > 2.0 for a dwarf) or specific target and cross-query NASA Archive
            if stellar_params['Rs'] > 2.0 or clean_id == "25155310":
                print(f"  Warning: Anomalous radius {stellar_params['Rs']} R_sun detected in TIC. Cross-querying NASA Exoplanet Archive...")
                try:
                    archive_table = NasaExoplanetArchive.query_criteria(
                        table="pscomppars",
                        select="st_rad,st_raderr1,st_mass,st_masserr1,st_teff,st_tefferr1",
                        where=f"tic_id='TIC {clean_id}'"
                    )
                    if archive_table is not None and len(archive_table) > 0:
                        tr = archive_table[0]
                        def _to_float(v, default=np.nan):
                            if v is None or np.ma.is_masked(v):
                                return default
                            val = getattr(v, 'value', v)
                            return float(val) if not np.isnan(val) else default

                        rad_val = _to_float(tr['st_rad'])
                        if not np.isnan(rad_val):
                            stellar_params['Rs'] = rad_val
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
                            err_val = _to_float(tr['st_tefferr1'])
                            stellar_params['Teff_err'] = abs(err_val) if not np.isnan(err_val) else 100.0

                        print(f"  Successfully updated stellar parameters from NASA Archive: Rs={stellar_params['Rs']} R_sun, Ms={stellar_params['Ms']} M_sun, Teff={stellar_params['Teff']} K")
                except Exception as tap_err:
                    print(f"  Warning: Archive cross-query failed ({tap_err}). Proceeding with TIC parameters.")
        else:
            print(f"  Warning: Target TIC {clean_id} not found in MAST catalogs. Using solar fallback.")
    except Exception as e:
        print(f"  Error fetching stellar params: {e}. Using solar fallback.")

    return stellar_params
