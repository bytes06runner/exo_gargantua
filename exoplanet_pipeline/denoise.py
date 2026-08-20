"""
denoise.py — Noise Decomposition and Edge-Preserving Filtering
==============================================================

Decomposes the light curve into systematic drift and residual components,
applies Savitzky-Golay filtering, and computes noise statistics.

Fix 1 (completed here): Uses the raw_flux_err from ingestion.py to compute
real photon noise, instead of sqrt(normalized_flux) ≈ sqrt(1) ≈ 1.
"""

import numpy as np
from scipy.signal import savgol_filter


def decompose_and_filter(lc, raw_flux_err=None, window_length=101):
    """
    Decomposes noise components and applies edge-preserving filtering.

    The photon noise estimate uses the actual measurement uncertainties
    (flux_err) from the instrument, captured before normalization. This
    gives a physically meaningful noise floor rather than sqrt(~1).

    Parameters
    ----------
    lc : lightkurve.LightCurve
        The cleaned, normalized, stitched light curve.
    raw_flux_err : np.ndarray or None
        Per-cadence flux errors from the pre-normalization light curves,
        normalized by the same median divisor. If None, falls back to
        the light curve's own flux_err attribute.
    window_length : int
        Window length for the Savitzky-Golay filter (must be odd).

    Returns
    -------
    filtered_lc : lightkurve.LightCurve
        The filtered light curve.
    residual_flux : np.ndarray
        Flux minus the systematic drift component.
    noise_stats : dict
        Dictionary with 'photon_noise_avg', 'systematic_std',
        'residual_std'. photon_noise_avg is now a real measurement
        uncertainty, typically ~1e-4 to 1e-3 for TESS data, not ~1.0.
    """
    # Fix 1: Use real flux_err for photon noise, not sqrt(normalized_flux).
    #
    # Before this fix, photon_noise was computed as:
    #     np.sqrt(np.abs(lc.flux.value))
    # But since lc.flux is normalized to ~1.0, this just gives ~1.0,
    # which is meaningless. The correct approach is to use the actual
    # per-cadence measurement uncertainty from the instrument.
    if raw_flux_err is not None and len(raw_flux_err) == len(lc):
        photon_noise = raw_flux_err
    elif hasattr(lc, 'flux_err') and lc.flux_err is not None:
        photon_noise = lc.flux_err.value
    else:
        # Last resort fallback (should not normally be reached)
        photon_noise = np.sqrt(np.abs(lc.flux.value))
        print("WARNING: Using sqrt(flux) fallback for photon noise — "
              "this is unreliable on normalized data.")

    from astropy.timeseries import BoxLeastSquares

    time_orig = lc.time.value
    flux_orig = lc.flux.value
    n_cadences = len(lc)
    print(f"  Total cadences: {n_cadences}")

    # ---- Step 1: Fast BLS on heavily decimated data to find transits ----
    # Decimate heavily to ~20k points. Lightkurve's .bin() is too slow on 2M+ points.
    print("  Decimating light curve for fast BLS...")
    step = max(1, n_cadences // 20000)
    decimated_for_bls = lc[::step]
    
    # Quick flatten on the decimated data (fast: only ~20k points)
    flat_binned = decimated_for_bls.flatten(window_length=101, break_tolerance=5)
    
    time_bls = flat_binned.time.value
    flux_bls = flat_binned.flux.value
    
    # Remove NaNs
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
    # Split light curve into contiguous segments across data gaps (gap > 0.5 days)
    # This prevents savgol_filter from distorting the continuum across multi-week/year sector breaks.
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
            
        # Determine appropriate window length for this segment
        wl_seg = min(501, len(seg_idx) // 2)
        if wl_seg % 2 == 0:
            wl_seg += 1
        wl_seg = max(wl_seg, 11)
        
        if len(seg_idx) > wl_seg:
            from scipy.ndimage import median_filter
            # Apply a median filter first to reject edge discontinuities
            f_med = median_filter(f_seg, size=min(11, len(seg_idx)))
            seg_base = savgol_filter(f_med, window_length=wl_seg, polyorder=2)
            
            # EDGE MARGIN MASKING: mask first and last 5 cadences of the segment
            if len(seg_base) > 10:
                seg_base[:5] = np.nan
                seg_base[-5:] = np.nan
        else:
            seg_base = np.full(len(seg_idx), np.nanmedian(f_seg))
            
        baseline[seg_idx] = seg_base
    
    # Avoid divide-by-zero
    baseline[baseline <= 0] = 1.0
    detrended_flux = flux_orig / baseline
    
    # Build a new LightCurve with the detrended flux
    import lightkurve as lk_mod
    filtered_lc = lk_mod.LightCurve(time=lc.time, flux=detrended_flux, flux_err=lc.flux_err).remove_nans()
    residual_flux = detrended_flux - 1.0

    noise_stats = {
        'photon_noise_avg': np.nanmean(photon_noise),
        'systematic_std': np.nanstd(baseline),
        'residual_std': np.nanstd(residual_flux)
    }

    print(f"  Detrending complete. Residual std: {noise_stats['residual_std']:.6f}")
    return filtered_lc, residual_flux, noise_stats

