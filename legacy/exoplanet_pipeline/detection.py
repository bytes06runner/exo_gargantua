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

def resolve_fundamental_period(lc_bls, p_max, max_period, min_period):
    """
    Evaluates a bidirectional alias grid of the dominant BLS period to find
    the true fundamental period using max SNR and Odd/Even consistency.
    """
    time_val = lc_bls.time.value
    flux_val = lc_bls.flux.value
    valid = ~np.isnan(flux_val)
    time_val = time_val[valid]
    flux_val = flux_val[valid]
    
    trial_multipliers = [
        1/7, 1/6, 1/5, 1/4.5, 1/4, 1/3.5, 1/3, 1/2.5, 1/2, 1/1.5,
        1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0
    ]
    
    class EvaluatedCandidate:
        def __init__(self, pg, snr, depth, period, multiplier):
            self.pg = pg
            self.snr = snr
            self.depth = depth
            self.period = period
            self.multiplier = multiplier
            
    evaluated_candidates = []
    
    for m in trial_multipliers:
        trial_p = p_max * m
        if trial_p < min_period or trial_p > max_period:
            continue
            
        narrow_p = np.linspace(trial_p * 0.95, trial_p * 1.05, 2000)
        try:
            pg_sub = lc_bls.to_periodogram(method='bls', period=narrow_p, frequency_factor=5000)
        except Exception:
            continue
            
        best_p = pg_sub.period_at_max_power.value
        best_t0 = pg_sub.transit_time_at_max_power.value
        best_dur = pg_sub.duration_at_max_power.value
        
        max_idx = np.argmax(pg_sub.power)
        raw_snr = pg_sub.snr[max_idx]
        
        # Calculate epoch for each cadence
        E = np.round((time_val - best_t0) / best_p)
        in_transit = np.abs(time_val - (best_t0 + E * best_p)) < (best_dur / 2)
        
        n_transits = len(np.unique(E[in_transit]))
        if n_transits > 0:
            best_depth = 1.0 - np.nanmean(flux_val[in_transit])
            out_std = np.nanstd(flux_val[~in_transit])
            if out_std > 0:
                best_snr = (best_depth * np.sqrt(n_transits)) / out_std
            else:
                best_snr = 0.0
        else:
            best_depth = 0.0
            best_snr = 0.0
        
        # Epoch Gate Check
        E = np.round((time_val - best_t0) / best_p)
        in_transit = np.abs(time_val - (best_t0 + E * best_p)) < (best_dur / 2)
        unique_epochs = len(np.unique(E[in_transit]))
        
        if unique_epochs < 3:
            continue
            
        # Odd/Even Depth Check (calculated per-transit to properly account for red noise)
        E_in_transit = E[in_transit]
        
        odd_epochs = np.unique(E_in_transit[(E_in_transit % 2) != 0])
        even_epochs = np.unique(E_in_transit[(E_in_transit % 2) == 0])
        
        if len(odd_epochs) >= 2 and len(even_epochs) >= 2:
            odd_transit_depths = [1.0 - np.nanmean(flux_val[in_transit & (E == ep)]) for ep in odd_epochs]
            even_transit_depths = [1.0 - np.nanmean(flux_val[in_transit & (E == ep)]) for ep in even_epochs]
            
            odd_mean = np.nanmean(odd_transit_depths)
            even_mean = np.nanmean(even_transit_depths)
            
            var_odd = np.nanvar(odd_transit_depths)
            var_even = np.nanvar(even_transit_depths)
            
            if var_odd > 0 or var_even > 0:
                pooled_se = np.sqrt(var_odd / len(odd_epochs) + var_even / len(even_epochs))
                if pooled_se > 0:
                    diff_sigma = np.abs(odd_mean - even_mean) / pooled_se
                    rel_diff = np.abs(odd_mean - even_mean) / max(np.abs(odd_mean), np.abs(even_mean), 1e-6)
                    if diff_sigma > 2.0 and rel_diff > 0.1:
                        continue  # Failed odd/even check (likely EB)
                        
        evaluated_candidates.append(EvaluatedCandidate(
            pg=pg_sub, snr=raw_snr, depth=best_depth, period=best_p, multiplier=m
        ))
        
    if not evaluated_candidates:
        return None, 1.0
        
    D_max = max(c.depth for c in evaluated_candidates)
    
    # ── TESS Perigee Veto ────────────────────────────────────────────
    # TESS orbits Earth every ~13.7 days. At perigee it pauses for data
    # downlink, creating thermal settling dips and momentum dump artefacts.
    # These are deep, periodic, and perfectly mimic transits to BLS.
    #
    # Danger zones (period ranges dominated by spacecraft systematics):
    #   Primary perigee gap:  12.5 – 15.5 d
    #   Half-orbit dump:       6.5 –  7.5 d
    # ─────────────────────────────────────────────────────────────────
    PERIGEE_ZONES = [(10.0, 17.0), (6.5, 7.5)]
    PENALTY_FACTOR = 0.01

    for c in evaluated_candidates:
        in_danger = any(lo <= c.period <= hi for lo, hi in PERIGEE_ZONES)
        if in_danger:
            c.snr = c.snr * PENALTY_FACTOR
            print(f"  TESS Perigee Veto: P={c.period:.4f}d is in systematic danger zone, SNR penalised ×{PENALTY_FACTOR}")

    print("\n--- DEBUG: Harmonic Validator Candidates ---")
    for c in evaluated_candidates:
        print(f"P={c.period:.4f}d (m={c.multiplier:.2f}): depth={c.depth:.6f}, snr={c.snr:.1f} (thresh={0.95*D_max:.6f})")
    
    # Final selection: highest (possibly penalised) raw BLS SNR wins.
    best_period_obj = max(evaluated_candidates, key=lambda x: x.snr)
    
    return best_period_obj.pg, best_period_obj.multiplier


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
    
    # Dynamic frequency grid calculation
    t_baseline = lc_bls.time.value[-1] - lc_bls.time.value[0]
    df = 1.0 / (3.0 * t_baseline)
    freq_min = 1.0 / max_period
    freq_max = 1.0 / min_period
    if (freq_max - freq_min) / df > 200000:
        df = (freq_max - freq_min) / 200000.0
    freqs = np.arange(freq_min, freq_max, df)
    periods = 1.0 / freqs
    
    periodogram = lc_bls.to_periodogram(
        method='bls', 
        period=periods,
        frequency_factor=5000
    )
    
    # Module 3: SDE Normalization
    import pandas as pd
    power_s = pd.Series(periodogram.power.value)
    roll_med = power_s.rolling(window=501, min_periods=1, center=True).median()
    roll_std = power_s.rolling(window=501, min_periods=1, center=True).std()
    
    # Avoid div by zero
    roll_std_mean = np.nanmean(roll_std)
    if roll_std_mean == 0 or np.isnan(roll_std_mean):
        roll_std_mean = 1.0
    roll_std[roll_std == 0] = roll_std_mean
    
    sde = (power_s - roll_med) / roll_std
    sde = sde.fillna(0).values
    
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(sde)
    if len(peaks) == 0:
        peaks = np.argsort(sde)[-50:]
        
    sorted_peaks = peaks[np.argsort(sde[peaks])][::-1]
    
    # Module 2: Distinct Epoch Gating
    time_val = lc_bls.time.value
    flux_val = lc_bls.flux.value
    valid = ~np.isnan(flux_val)
    time_val = time_val[valid]
    
    best_period = None
    best_t0 = None
    best_depth = None
    best_duration = None
    best_snr = None
    
    PERIGEE_ZONES = [(10.0, 17.0), (6.5, 7.5)]
    PENALTY_FACTOR = 0.01
    
    valid_peaks = []
    for pk in sorted_peaks:
        P = periodogram.period.value[pk]
        t0 = periodogram.transit_time.value[pk]
        dur = periodogram.duration.value[pk]
        snr = periodogram.power.value[pk]
        
        # Calculate epoch for each cadence
        E = np.round((time_val - t0) / P)
        
        # In-transit points
        in_transit = np.abs(time_val - (t0 + E * P)) < (dur / 2)
        
        unique_epochs = len(np.unique(E[in_transit]))
        
        if unique_epochs >= 3:
            in_danger = any(lo <= P <= hi for lo, hi in PERIGEE_ZONES)
            if in_danger:
                snr = snr * PENALTY_FACTOR
                print(f"  TESS Perigee Veto (BLS init): P={P:.4f}d in danger zone, SNR penalised ×{PENALTY_FACTOR}")
            
            valid_peaks.append({
                'pk': pk,
                'period': periodogram.period[pk],
                't0': periodogram.transit_time[pk],
                'depth': periodogram.depth[pk],
                'duration': periodogram.duration[pk],
                'snr': snr
            })
            
    if valid_peaks:
        best_peak = max(valid_peaks, key=lambda x: x['snr'])
        best_period = best_peak['period']
        best_t0 = best_peak['t0']
        best_depth = best_peak['depth']
        best_duration = best_peak['duration']
        best_snr = best_peak['snr']
            
    # Fallback if no peaks pass the strict gate
    if best_period is None:
        best_period = periodogram.period_at_max_power
        best_t0 = periodogram.transit_time_at_max_power
        best_depth = periodogram.depth_at_max_power
        best_duration = periodogram.duration_at_max_power
        # Reject by setting power to 0
        best_snr = periodogram.max_power * 0.0

    # Removed old Systematic Veto check that was just 13.2-14.2

    # Harmonic Validator Check
    p_val = best_period.value if hasattr(best_period, 'value') else best_period
    
    resolved_pg, multiplier = resolve_fundamental_period(
        lc_bls, p_val, max_period, min_period
    )
    
    if resolved_pg is not None:
        if multiplier != 1.0:
            print(f"  Bidirectional Validator: P={p_val:.5f}d -> Adopting P={resolved_pg.period_at_max_power.value:.5f}d (multiplier {multiplier})")
        
        best_period = resolved_pg.period_at_max_power
        best_t0 = resolved_pg.transit_time_at_max_power
        best_depth = resolved_pg.depth_at_max_power
        best_duration = resolved_pg.duration_at_max_power
        best_snr = resolved_pg.max_power
        periodogram = resolved_pg

    # Returning best parameters based strictly on SDE + unique epoch gating + Harmonic Validator
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
