"""
denoise.py — Noise Decomposition and Edge-Preserving Filtering
==============================================================

Decomposes the light curve into systematic drift and residual components,
applies Savitzky-Golay filtering, and computes noise statistics.
"""

import numpy as np
from scipy.signal import savgol_filter
import lightkurve as lk_mod
from astropy.timeseries import BoxLeastSquares

def decompose_and_filter(lc, raw_flux_err=None, window_length=101):
    """
    Decomposes noise components and applies edge-preserving filtering.
    """
    # Fix 1: Use real flux_err for photon noise
    if raw_flux_err is not None and len(raw_flux_err) == len(lc):
        photon_noise = raw_flux_err
    elif hasattr(lc, 'flux_err') and lc.flux_err is not None:
        photon_noise = lc.flux_err.value
    else:
        photon_noise = np.sqrt(np.abs(lc.flux.value))
        print("WARNING: Using sqrt(flux) fallback for photon noise — "
              "this is unreliable on normalized data.")

    time_orig = lc.time.value
    flux_orig = lc.flux.value
    n_cadences = len(lc)
    print(f"  Total cadences: {n_cadences}")


    # ---- Step 0: Parametric Exponential Thermal Decay Model ----
    # Instead of cropping, we explicitly fit an exponential decay to the baseline
    # to perfectly remove the TESS thermal settling without cutting data.
    import scipy.optimize as opt
    
    def thermal_model(t, A, tau, c):
        return A * np.exp(-(t - t[0]) / max(tau, 0.001)) + c
        
    thermal_baseline = np.ones_like(flux_orig)
    
    dt = np.diff(time_orig)
    gap_indices = np.where(dt > 0.5)[0] + 1
    segment_splits = np.split(np.arange(n_cadences), gap_indices)
    
    for seg_idx in segment_splits:
        if len(seg_idx) < 50: continue
        t_seg = time_orig[seg_idx]
        f_seg = flux_orig[seg_idx]
        
        # Aggressively mask extreme positive momentum dumps for the baseline fit
        med = np.nanmedian(f_seg)
        std = np.nanstd(f_seg)
        valid = (f_seg - med) < (3.0 * std)
        
        if np.sum(valid) > 10:
            try:
                # Guess: A = max - median, tau = 0.5 days, c = median
                A_guess = np.nanpercentile(f_seg[valid][:20], 95) - med
                if A_guess < 0: A_guess = 0.001
                
                popt, _ = opt.curve_fit(
                    thermal_model, t_seg[valid], f_seg[valid],
                    p0=[A_guess, 0.5, med],
                    bounds=([0, 0.01, -np.inf], [np.inf, 3.0, np.inf]),
                    maxfev=2000
                )
                
                seg_model = thermal_model(t_seg, *popt)
                # Normalize the thermal model so we can divide it out
                seg_model /= np.nanmedian(seg_model)
                thermal_baseline[seg_idx] = seg_model
            except Exception:
                pass
                
    flux_orig = flux_orig / thermal_baseline
    print("  Applied Parametric Exponential Thermal Decay correction.")
    
    # ---- Step 1: Fast BLS on heavily decimated data to find transits ----
    print("  Decimating light curve for fast BLS...")
    step = max(1, n_cadences // 20000)
    decimated_for_bls = lc[::step]
    
    # Quick flatten on the decimated data
    flat_binned = decimated_for_bls.flatten(window_length=101, break_tolerance=5)
    
    time_bls = flat_binned.time.value
    flux_bls = flat_binned.flux.value
    
    good = np.isfinite(flux_bls) & np.isfinite(time_bls)
    time_bls = time_bls[good]
    flux_bls = flux_bls[good]
    
    print(f"  Running fast BLS for transit-masked detrending ({len(time_bls)} binned points)...")
    model = BoxLeastSquares(time_bls, flux_bls)
    freq_grid = np.linspace(1.0 / 25.0, 1.0 / 0.5, 25000)
    period_grid = 1.0 / freq_grid
    durations = np.array([0.05, 0.1, 0.15, 0.2])
    results = model.power(period_grid, durations)
    
    best_idx = np.argmax(results.power)
    best_period = float(results.period[best_idx])
    best_t0 = float(results.transit_time[best_idx])
    best_dur = float(results.duration[best_idx])
    
    print(f"  Initial BLS found P={best_period:.4f}d, t0={best_t0:.4f}, dur={best_dur:.4f}d")

    # ---- Step 2: Build transit mask on the original time array ----
    phase = (time_orig - best_t0 + 0.5 * best_period) % best_period - 0.5 * best_period
    transit_mask = np.abs(phase) < (best_dur * 1.5)
    n_masked = np.sum(transit_mask)
    print(f"  Transit mask: {n_masked} cadences masked ({100*n_masked/n_cadences:.1f}%)")

    # ---- Step 3: Segment-aware transit-masked baseline detrending ----
    print("  Fitting transit-masked baseline per contiguous sector segment...")
    
    dt = np.diff(time_orig)
    gap_indices = np.where(dt > 0.5)[0] + 1
    segment_splits = np.split(np.arange(n_cadences), gap_indices)
    
    baseline = np.ones(n_cadences, dtype=float)
    
    for seg_idx in segment_splits:
        if len(seg_idx) < 10:
            baseline[seg_idx] = np.nanmedian(flux_orig[seg_idx])
            continue
            
        t_seg = time_orig[seg_idx]
        f_seg = flux_orig[seg_idx].copy()
        m_seg = transit_mask[seg_idx]
        
        # Mask transits in this segment
        f_seg[m_seg] = np.nan
        nan_m = np.isnan(f_seg)
        
        if np.all(nan_m):
            baseline[seg_idx] = np.nanmedian(flux_orig[seg_idx])
            continue
            
        if np.any(nan_m):
            f_seg[nan_m] = np.interp(t_seg[nan_m], t_seg[~nan_m], f_seg[~nan_m])
            
        # Determine appropriate window length for this segment (approx 2 days)
        if len(t_seg) > 1:
            cadences_per_day = 1.0 / np.nanmedian(np.diff(t_seg))
        else:
            cadences_per_day = 48
            
        wl_seg = int(2.0 * cadences_per_day)
        if wl_seg % 2 == 0:
            wl_seg += 1
        wl_seg = max(wl_seg, 11)
        wl_seg = min(wl_seg, len(seg_idx))
        if wl_seg % 2 == 0:
            wl_seg -= 1
        
        if len(seg_idx) >= wl_seg and wl_seg >= 3:
            # Use Savitzky-Golay for gentle stellar variability (CBVs handled the steep edges)
            seg_base = savgol_filter(f_seg, window_length=wl_seg, polyorder=2)
            
            # EDGE MARGIN MASKING: mask first and last 5 cadences of the segment
            # to avoid any residual SG edge divergence
            if len(seg_base) > 10:
                seg_base[:5] = np.nan
                seg_base[-5:] = np.nan
        else:
            seg_base = np.full(len(seg_idx), np.nanmedian(f_seg))
            
        baseline[seg_idx] = seg_base
        
    baseline[baseline <= 0] = 1.0
    detrended_flux = flux_orig / baseline
    
    filtered_lc = lk_mod.LightCurve(time=lc.time, flux=detrended_flux, flux_err=lc.flux_err).remove_nans()
    residual_flux = detrended_flux - 1.0

    noise_stats = {
        'photon_noise_avg': np.nanmean(photon_noise),
        'systematic_std': np.nanstd(baseline),
        'residual_std': np.nanstd(residual_flux)
    }

    print(f"  Detrending complete. Residual std: {noise_stats['residual_std']:.6f}")
    return filtered_lc, residual_flux, noise_stats

