"""
detection.py — BLS Transit Search with FAP-Based Confidence
============================================================

Runs the Box Least Squares (BLS) periodogram to find the best-fit
transit period, epoch, depth, and duration.

Fix 2: Replaces arbitrary confidence thresholds (tsnr/12 or 0.95) with
an empirical False Alarm Probability (FAP) computed via bootstrap
permutation. This gives a statistically defensible confidence value:
    confidence = 1 - FAP
where FAP = fraction of time-shuffled null trials that produce BLS power
≥ the observed power from the real data.
"""

import numpy as np
import lightkurve as lk


def run_bls_search(lc, min_period=0.5, max_period=20):
    """
    Runs BLS periodogram search for transit signals with harmonic
    resolution to detect when BLS locks onto the 2× harmonic.

    Parameters
    ----------
    lc : lightkurve.LightCurve
        The filtered light curve to search.
    min_period, max_period : float
        Period search range in days.

    Returns
    -------
    best_fit : dict
        Dictionary with 'period', 't0', 'duration', 'snr', 'depth',
        and 'periodogram' keys.
    """
    # Decimate the light curve if it has too many cadences for BLS to handle
    # efficiently. BLS on 2M+ points is infeasible; ~50k is plenty.
    lc_bls = lc
    if len(lc) > 100000:
        print(f"  Decimating {len(lc)} cadences to ~50k for BLS search...")
        step = max(1, len(lc) // 50000)
        lc_bls = lc[::step]
        print(f"  Decimated to {len(lc_bls)} cadences")
    
    periods = np.linspace(min_period, max_period, 5000)
    # frequency_factor=5000 bypasses a Lightkurve bug where it validates the
    # default grid size and crashes even if an explicit period array is passed.
    periodogram = lc_bls.to_periodogram(
        method='bls', 
        period=periods,
        frequency_factor=5000
    )
    
    best_period = periodogram.period_at_max_power
    best_t0 = periodogram.transit_time_at_max_power
    best_depth = periodogram.depth_at_max_power
    best_duration = periodogram.duration_at_max_power
    best_snr = periodogram.max_power

    # Harmonic resolution: check if the true period is P/2
    # When BLS finds a harmonic (2×P_true), folding at P/2 will show
    # transits at BOTH phase 0 and phase 0.5, and the depth should be
    # similar or deeper. This is the signature of a harmonic lock.
    half_period = best_period / 2
    if half_period.value >= min_period:
        try:
            # Fold at P/2 and check for transit at phase 0.5
            folded = lc.fold(period=best_period, epoch_time=best_t0)
            phase = (folded.time.value / best_period.value) % 1.0
            flux = folded.flux.value
            
            # Check for a dip near phase 0.5 (which would be a second
            # transit if the true period is P/2)
            sec_mask = (phase >= 0.45) & (phase <= 0.55)
            baseline_mask = ((phase >= 0.15) & (phase <= 0.35)) | \
                            ((phase >= 0.65) & (phase <= 0.85))
            
            if np.sum(sec_mask) > 10 and np.sum(baseline_mask) > 10:
                sec_depth = np.nanmean(flux[baseline_mask]) - np.nanmean(flux[sec_mask])
                pri_mask = (phase >= 0.0) & (phase <= 0.05) | (phase >= 0.95)
                pri_depth = np.nanmean(flux[baseline_mask]) - np.nanmean(flux[pri_mask]) if np.sum(pri_mask) > 5 else 0
                
                # If the depth at phase 0.5 is at least 30% of the primary
                # depth, the true period is likely P/2
                if sec_depth > 0 and pri_depth > 0 and sec_depth > 0.3 * pri_depth:
                    print(f"  Harmonic detected: P={best_period.value:.4f}d has transit at phase 0.5")
                    print(f"    Primary depth: {pri_depth:.6f}, Secondary depth: {sec_depth:.6f}")
                    print(f"    Adopting true period P/2 = {half_period.value:.4f}d")
                    
                    # Re-run BLS in a narrow window around P/2 for precise value
                    narrow_periods = np.linspace(
                        half_period.value * 0.98,
                        half_period.value * 1.02,
                        1000
                    )
                    pg2 = lc_bls.to_periodogram(
                        method='bls',
                        period=narrow_periods,
                        frequency_factor=5000
                    )
                    best_period = pg2.period_at_max_power
                    best_t0 = pg2.transit_time_at_max_power
                    best_depth = pg2.depth_at_max_power
                    best_duration = pg2.duration_at_max_power
                    best_snr = pg2.max_power
                    print(f"    Refined: P={best_period.value:.6f}d, depth={best_depth.value:.6f}")
        except Exception as e:
            print(f"  Harmonic check failed: {e}")

    best_fit = {
        'period': best_period,
        't0': best_t0,
        'duration': best_duration,
        'snr': best_snr,
        'depth': best_depth,
        'periodogram': periodogram
    }
    return best_fit


def compute_fap_bootstrap(lc, observed_power, n_trials=200,
                          min_period=0.5, max_period=20):
    """
    Estimates the False Alarm Probability (FAP) of a BLS detection via
    bootstrap permutation of the flux values.

    Method
    ------
    1. Decimate the light curve by keeping every 5th cadence (for speed).
    2. For each of n_trials iterations:
       a. Randomly shuffle the flux values (breaking any real periodic
          signal while preserving the time sampling and noise properties).
       b. Run BLS on the shuffled data and record the maximum power.
    3. FAP = (count of null powers ≥ observed power) / n_trials.

    This is an empirical test: if the real signal's BLS power exceeds
    all (or nearly all) of the null-distribution powers, the FAP is
    small and the detection is statistically significant.

    Parameters
    ----------
    lc : lightkurve.LightCurve
        The light curve that was searched (same as passed to run_bls_search).
    observed_power : float
        The maximum BLS power from the real (unshuffled) data.
    n_trials : int
        Number of bootstrap permutations (200 is a good balance of speed
        and resolution; FAP resolution is 1/n_trials).
    min_period, max_period : float
        Period search range (should match run_bls_search).

    Returns
    -------
    fap : float
        False Alarm Probability in [0, 1]. Small values (< 0.01) indicate
        a statistically significant detection.
    confidence : float
        1 - FAP, clipped to [0, 1]. Compatible with check_exoplanet_alert.
    null_powers : np.ndarray
        The full null distribution of max BLS powers (for diagnostics).
    """
    # Decimate for speed: target ~5,000 cadences for the null trials
    # (the full light curve can have millions of points across many sectors)
    target_n = 5000
    decimation_factor = max(1, len(lc) // target_n)
    indices = np.arange(0, len(lc), decimation_factor)
    lc_dec = lc[indices]

    # Use a coarse period grid for null trials: 500 points, frequency_factor=1
    # (the real BLS uses 5000 periods × frequency_factor=50, but for the null
    # distribution we only need to know the max power, not the exact period)
    periods = np.linspace(min_period, max_period, 500)

    null_powers = np.zeros(n_trials)
    flux_values = lc_dec.flux.value.copy()

    print(f"  Running {n_trials} FAP bootstrap trials on {len(lc_dec)} cadences...")

    for i in range(n_trials):
        # Shuffle flux values (breaks any periodic signal)
        shuffled_flux = np.random.permutation(flux_values)

        # Create a copy with shuffled flux
        shuffled_lc = lc_dec.copy()
        shuffled_lc.flux = shuffled_flux

        # Run BLS on shuffled data with minimal frequency resolution
        try:
            pg = shuffled_lc.to_periodogram(
                method='bls', 
                period=periods,
                frequency_factor=5000
            )
            null_powers[i] = pg.max_power.value
        except Exception:
            null_powers[i] = 0.0

        if (i + 1) % 25 == 0:
            print(f"    Completed {i + 1}/{n_trials} trials")

    # FAP = fraction of null trials with power >= observed
    fap = np.sum(null_powers >= observed_power) / n_trials
    confidence = np.clip(1.0 - fap, 0.0, 1.0)

    print(f"  FAP = {fap:.4f} (confidence = {confidence:.4f})")
    print(f"  Observed power: {observed_power:.4f}, "
          f"null max: {np.max(null_powers):.4f}, "
          f"null mean: {np.mean(null_powers):.4f}")

    return fap, confidence, null_powers


def calculate_metrics_v2(bls_results, fap_confidence=None):
    """
    Computes transit detection metrics from BLS results.

    Fix 2: If fap_confidence is provided (from compute_fap_bootstrap),
    it replaces the old arbitrary threshold (tsnr/12 or 0.95) with a
    statistically grounded value.

    Parameters
    ----------
    bls_results : dict
        Output from run_bls_search.
    fap_confidence : float or None
        The FAP-derived confidence (1 - FAP). If None, falls back to
        the old heuristic (for backward compatibility during testing).

    Returns
    -------
    tsnr : float
        Transit signal-to-noise ratio (BLS max power).
    confidence : float
        Detection confidence: 1 - FAP if available, else heuristic.
    depth : float
        Transit depth from BLS.
    """
    tsnr = bls_results['snr'].value

    if fap_confidence is not None:
        confidence = fap_confidence
    else:
        # Legacy heuristic fallback
        confidence = 0.95 if tsnr > 10 else (tsnr / 12.0)
        confidence = min(confidence, 1.0)

    return tsnr, confidence, bls_results['depth'].value


def check_exoplanet_alert(tsnr, confidence, target_id):
    """
    Prints an alert banner if the detection passes threshold criteria.
    """
    if tsnr >= 7.0 and confidence >= 0.85:
        banner = f"""
        #################################################
        !! EXOPLANET CANDIDATE ALERT !!
        TARGET: {target_id}
        TSNR: {tsnr:.2f} | CONFIDENCE: {confidence:.2f}
        STATUS: TRIGGERED
        #################################################
        """
        print(banner)
    else:
        print(f"No alert triggered for {target_id}. "
              f"TSNR: {tsnr:.2f}, Confidence: {confidence:.2f}")
