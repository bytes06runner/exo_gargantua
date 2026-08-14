"""
vetting.py — Multi-Test Vetting: Odd/Even, Secondary Eclipse, Centroid
======================================================================

Phase 3 vetting with three quantitative checks:

1. Odd/even transit depth comparison (existing, unchanged)
2. Secondary eclipse significance (Fix 4: quantitative sigma test)
3. Centroid shift analysis (Fix 3: new pixel-level vetting)

Fix 3: run_centroid_vetting downloads target pixel files, computes
flux-weighted centroids, and compares in-transit vs out-of-transit
positions to detect blended false positives.

Fix 4: run_phase_3_vetting now computes the binned flux depth in the
phase window [0.45, 0.55] and reports it as a sigma significance,
replacing the visual-only "the green line looks flat" judgment.
"""

import numpy as np
import lightkurve as lk
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for CLI usage
import matplotlib.pyplot as plt


def run_phase_3_vetting(lc, period, t0):
    """
    Runs Phase 3 vetting tests on a folded light curve.

    Tests performed:
    1. Odd/even transit depth comparison — flags eclipsing binaries
       if the depth difference exceeds 0.005.
    2. Secondary eclipse significance (Fix 4) — computes the mean
       binned flux depression in the phase window [0.45, 0.55]
       relative to the out-of-eclipse baseline, and reports how
       many standard errors that depression is from zero.

    Parameters
    ----------
    lc : lightkurve.LightCurve
        The filtered light curve.
    period : astropy.units.Quantity
        Best-fit orbital period from BLS.
    t0 : astropy.units.Quantity
        Best-fit transit epoch from BLS.

    Returns
    -------
    dict with keys:
        'folded', 'binned' : LightCurve objects for plotting
        'odd', 'even' : binned odd/even transit LightCurves
        'depth_diff' : float, absolute odd-even depth difference
        'secondary_eclipse_sigma' : float, significance of secondary
            eclipse detection in sigma (Fix 4). Values > 3 suggest
            a real secondary eclipse (possible eclipsing binary or
            self-luminous companion).
        'secondary_eclipse_depth' : float, measured depth in the
            [0.45, 0.55] phase window (Fix 4).
    """
    folded_lc = lc.fold(period=period, epoch_time=t0)
    binned_lc = folded_lc.bin(time_bin_size=period / 100)

    # Odd/even comparison (unchanged logic)
    transit_indices = np.floor(
        (lc.time.value - t0.value) / period.value
    ).astype(int)
    odd_mask = (transit_indices % 2 != 0)
    even_mask = (transit_indices % 2 == 0)

    lc_odd = lc[odd_mask].fold(period=period, epoch_time=t0)
    lc_even = lc[even_mask].fold(period=period, epoch_time=t0)

    odd_depth = 1.0 - np.nanmin(
        lc_odd.bin(time_bin_size=period / 50).flux.value
    )
    even_depth = 1.0 - np.nanmin(
        lc_even.bin(time_bin_size=period / 50).flux.value
    )
    depth_diff = abs(odd_depth - even_depth)

    # Fix 4: Quantitative secondary eclipse test
    # Compute the mean flux depression in the phase window [0.45, 0.55]
    # (centered on phase 0.5, where a secondary eclipse would occur for
    # a circular orbit). Compare to the out-of-eclipse baseline.
    # Fix 4: Quantitative secondary eclipse test using continuum-referenced baseline subtraction
    # Must use the binned light curve (e.g. 100 phase bins) to avoid inflating the significance
    # due to highly correlated red noise in the millions of full-resolution cadences.
    phase = (binned_lc.time.value / period.value) % 1.0
    eclipse_mask = (phase >= 0.45) & (phase <= 0.55)
    baseline_mask = ((phase >= 0.15) & (phase <= 0.35)) | ((phase >= 0.65) & (phase <= 0.85))

    if np.sum(eclipse_mask) >= 2 and np.sum(baseline_mask) >= 2:
        # Sort phase and flux to ensure contiguous rolling windows
        sort_idx = np.argsort(phase)
        phase_sorted = phase[sort_idx]
        flux_sorted = binned_lc.flux.value[sort_idx]

        # Masks on sorted arrays
        eclipse_sorted = (phase_sorted >= 0.45) & (phase_sorted <= 0.55)
        mask1 = (phase_sorted >= 0.15) & (phase_sorted <= 0.35)
        mask2 = (phase_sorted >= 0.65) & (phase_sorted <= 0.85)

        # Depth is simple mean of eclipse vs baseline
        f_out = np.nanmean(flux_sorted[mask1 | mask2])
        f_sec = np.nanmean(flux_sorted[eclipse_sorted])
        eclipse_depth = f_out - f_sec

        # Empirical Standard Error via Rolling Mean:
        # We compute rolling means of the baseline chunks with a window size
        # equal to the eclipse duration (0.1 phase). The standard deviation 
        # of these rolling means precisely captures the integrated red noise.
        # At 100 bins per phase, 0.1 phase = 10 bins.
        n_bins_total = len(flux_sorted)
        window_size = max(1, int(0.1 * n_bins_total))
        
        rolling_means = []
        for m in [mask1, mask2]:
            chunk_flux = flux_sorted[m]
            chunk_flux = chunk_flux[~np.isnan(chunk_flux)]
            if len(chunk_flux) >= window_size:
                rm = np.convolve(chunk_flux, np.ones(window_size)/window_size, mode='valid')
                rolling_means.extend(rm)

        if len(rolling_means) > 0:
            se_eclipse = np.nanstd(rolling_means)
        else:
            # Fallback if baseline chunk is too small
            se_eclipse = np.nanstd(flux_sorted[mask1 | mask2]) / np.sqrt(np.sum(eclipse_sorted))

        if se_eclipse > 0:
            secondary_eclipse_sigma = eclipse_depth / se_eclipse
        else:
            secondary_eclipse_sigma = 0.0
    else:
        eclipse_depth = 0.0
        secondary_eclipse_sigma = 0.0
        print("  Warning: insufficient data points in eclipse/baseline windows")

    return {
        'folded': folded_lc,
        'binned': binned_lc,
        'odd': lc_odd.bin(time_bin_size=period / 50),
        'even': lc_even.bin(time_bin_size=period / 50),
        'depth_diff': depth_diff,
        'secondary_eclipse_sigma': secondary_eclipse_sigma,
        'secondary_eclipse_depth': eclipse_depth,
    }


def visualize_vetting(vetting_results, target_id):
    """
    Generates Phase 3 vetting visualizations and prints quantitative results.

    Fix 4: Now prints the secondary eclipse sigma significance with a
    pass/fail determination instead of relying on visual inspection.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    vetting_results['folded'].scatter(
        ax=ax1, c='gray', alpha=0.3, label='Raw Cadences'
    )
    vetting_results['binned'].plot(
        ax=ax1, c='red', lw=2, label='Binned Average'
    )
    ax1.set_title(f"Folded Transit: {target_id}")
    ax1.set_xlim(-0.5, 0.5)

    vetting_results['odd'].plot(
        ax=ax2, c='orange', label='Odd Transits', lw=2
    )
    vetting_results['even'].plot(
        ax=ax2, c='blue', label='Even Transits', lw=2
    )
    ax2.set_title(
        f"Odd/Even Comparison (Diff: {vetting_results['depth_diff']:.5f})"
    )
    ax2.set_xlim(-0.2, 0.2)

    vetting_results['folded'].scatter(ax=ax3, c='gray', alpha=0.2)
    vetting_results['binned'].plot(ax=ax3, c='green', lw=2)
    ax3.set_xlim(0.3, 0.7)
    ax3.set_ylim(0.998, 1.002)

    # Fix 4: Display sigma value on the secondary eclipse plot
    sigma = vetting_results.get('secondary_eclipse_sigma', 0.0)
    depth = vetting_results.get('secondary_eclipse_depth', 0.0)
    ax3.set_title(
        f"Secondary Eclipse (depth={depth:.6f}, {sigma:.1f}σ)"
    )

    plt.tight_layout()
    plt.savefig('vetting_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Vetting plots saved to vetting_plots.png")

    # Odd/even depth check
    if vetting_results['depth_diff'] > 0.005:
        print("  !!! VETTING WARNING: Significant Odd/Even depth difference "
              "detected. Possible Eclipsing Binary.")
    else:
        print("  --- VETTING PASSED: Transit depths are consistent "
              "across cycles. ---")

    # Fix 4: Quantitative secondary eclipse verdict
    # Fix 4: Quantitative secondary eclipse verdict
    print(f"  Secondary eclipse depth: {depth:.6f}")
    print(f"  Secondary eclipse significance: {sigma:.2f} sigma")
    if abs(sigma) > 3:
        print("  !!! VETTING WARNING: Secondary eclipse flagged at "
              f"{sigma:.1f}σ. Possible eclipsing binary or companion "
              "signal.")
    else:
        print("  --- SECONDARY ECLIPSE TEST PASSED: No significant "
              f"secondary eclipse ({abs(sigma):.1f}σ < 3σ threshold). ---")


def run_centroid_vetting(target_star_id, period, t0, duration):
    """
    Fix 3: Pixel-level centroid vetting to detect blended false positives.

    Downloads target pixel files (TPFs) from TESS SPOC, computes
    flux-weighted centroid positions per cadence, and compares the
    mean centroid during in-transit cadences vs out-of-transit cadences.

    A significant centroid shift during transit indicates the dimming
    source is offset from the target star — a hallmark of a background
    eclipsing binary blended in the photometric aperture.

    Method
    ------
    For each cadence, the flux-weighted centroid is:
        col_centroid = Σ(flux * col_index) / Σ(flux)
        row_centroid = Σ(flux * row_index) / Σ(flux)
    where the sum is over all pixels in the TPF aperture.

    In-transit cadences are those within duration/2 of any predicted
    transit time (given period and t0). The shift is the Euclidean
    distance between the mean in-transit and out-of-transit centroids.

    Parameters
    ----------
    target_star_id : str
        TESS target identifier (e.g., "TIC 25155310").
    period : astropy.units.Quantity
        Orbital period from BLS (in days).
    t0 : astropy.units.Quantity
        Transit epoch from BLS (in BTJD).
    duration : astropy.units.Quantity
        Transit duration from BLS (in days).

    Returns
    -------
    dict with keys:
        'centroid_shift' : float, Euclidean shift in pixels between
            in-transit and out-of-transit mean centroids.
        'centroid_vetting_passed' : bool, True if shift < threshold.
        'in_transit_centroid' : tuple (col, row) mean in-transit centroid.
        'out_transit_centroid' : tuple (col, row) mean out-of-transit centroid.
        'n_in_transit' : int, number of in-transit cadences used.
        'n_out_transit' : int, number of out-of-transit cadences used.

    Notes
    -----
    The threshold of 1/3 TESS pixel (0.333 pix) is a reasonable starting
    point but should be tuned per target based on the star's brightness,
    crowding metric, and the number of cadences available. Bright,
    isolated targets can use a tighter threshold; faint, crowded targets
    may need a looser one.
    """
    # Threshold: 1/3 of a TESS pixel (~7 arcsec)
    # This should be tuned per target based on brightness and crowding.
    SHIFT_THRESHOLD = 0.333  # pixels

    print(f"  Downloading target pixel files for {target_star_id}...")
    search_result = None
    try:
        search_result = lk.search_targetpixelfile(
            target_star_id, mission='TESS', author='SPOC'
        )
    except Exception as e:
        print(f"  MAST API search failed or timed out: {e}")

    tpf_list = []
    
    if search_result:
        # Limit to first 5 sectors to avoid excessive download time
        max_tpfs = min(5, len(search_result))
        print(f"  Found {len(search_result)} TPFs, downloading first {max_tpfs}...")

        for idx in range(max_tpfs):
            try:
                tpf = search_result[idx].download()
                if tpf is not None:
                    tpf_list.append(tpf)
            except Exception as e:
                print(f"  Warning: failed to load TPF #{idx}: {e}")
                continue
    else:
        print("  Attempting to load TPFs from local cache...")
        import os
        import glob
        cache_dir = os.path.expanduser("~/.lightkurve/cache/mastDownload/TESS")
        clean_id = target_star_id.replace("TIC", "").strip()
        pattern1 = os.path.join(cache_dir, f"**/*{clean_id.zfill(16)}*tp.fits")
        pattern2 = os.path.join(cache_dir, f"**/*{clean_id}*tp.fits")
        
        files = glob.glob(pattern1, recursive=True)
        if not files:
            files = glob.glob(pattern2, recursive=True)
            
        if files:
            files = files[:5]  # limit to 5
            print(f"  Found {len(files)} cached TPFs.")
            for f in files:
                try:
                    tpf_list.append(lk.read(f))
                except Exception as read_err:
                    print(f"    Failed to read {f}: {read_err}")

    if len(tpf_list) == 0:
        print(f"  Failed to download any TPFs for {target_star_id}.")
        return {
            'centroid_shift': np.nan,
            'centroid_vetting_passed': False,
            'error': 'All TPF downloads failed'
        }

    print(f"  Successfully loaded {len(tpf_list)} TPFs.")

    period_val = period.value if hasattr(period, 'value') else float(period)
    t0_val = t0.value if hasattr(t0, 'value') else float(t0)
    dur_val = duration.value if hasattr(duration, 'value') else float(duration)

    all_col_in, all_row_in = [], []
    all_col_out, all_row_out = [], []
    
    in_frames_sum = None
    out_frames_sum = None
    n_in_frames = 0
    n_out_frames = 0

    for tpf in tpf_list:
        if tpf is None:
            continue

        # Build pixel coordinate grids
        n_times, n_rows, n_cols = tpf.flux.shape
        col_grid, row_grid = np.meshgrid(
            np.arange(n_cols), np.arange(n_rows)
        )

        times = tpf.time.value  # BTJD

        # Determine which cadences are in-transit
        # Phase = fractional distance from nearest transit center
        phase_from_transit = ((times - t0_val) % period_val) / period_val
        # Wrap to [-0.5, 0.5]
        phase_from_transit = np.where(
            phase_from_transit > 0.5,
            phase_from_transit - 1.0,
            phase_from_transit
        )
        # In-transit if within duration/2 of the transit center (phase ~ 0)
        half_dur_phase = (dur_val / 2.0) / period_val
        in_transit = np.abs(phase_from_transit) < half_dur_phase

        for i in range(n_times):
            frame = tpf.flux.value[i]
            if np.all(np.isnan(frame)):
                continue

            total_flux = np.nansum(frame)
            if total_flux <= 0:
                continue

            # Flux-weighted centroid
            col_c = np.nansum(frame * col_grid) / total_flux
            row_c = np.nansum(frame * row_grid) / total_flux

            if np.isfinite(col_c) and np.isfinite(row_c):
                if in_transit[i]:
                    all_col_in.append(col_c)
                    all_row_in.append(row_c)
                    if in_frames_sum is None:
                        in_frames_sum = frame.copy()
                    else:
                        in_frames_sum += frame
                    n_in_frames += 1
                else:
                    all_col_out.append(col_c)
                    all_row_out.append(row_c)
                    if out_frames_sum is None:
                        out_frames_sum = frame.copy()
                    else:
                        out_frames_sum += frame
                    n_out_frames += 1

    n_in = len(all_col_in)
    n_out = len(all_col_out)

    if n_in < 3 or n_out < 10:
        print(f"  Warning: insufficient cadences for centroid test "
              f"(in={n_in}, out={n_out})")
        return {
            'centroid_shift': np.nan,
            'centroid_vetting_passed': False,
            'n_in_transit': n_in,
            'n_out_transit': n_out,
            'error': 'Insufficient cadences'
        }

    mean_col_in = np.nanmean(all_col_in)
    mean_row_in = np.nanmean(all_row_in)
    mean_col_out = np.nanmean(all_col_out)
    mean_row_out = np.nanmean(all_row_out)

    shift = np.sqrt(
        (mean_col_in - mean_col_out)**2 + (mean_row_in - mean_row_out)**2
    )
    passed = shift < SHIFT_THRESHOLD

    print(f"  Centroid shift: {shift:.4f} pixels "
          f"(threshold: {SHIFT_THRESHOLD:.3f} pix)")
    print(f"  In-transit centroid:  col={mean_col_in:.4f}, row={mean_row_in:.4f} "
          f"({n_in} cadences)")
    print(f"  Out-of-transit centroid: col={mean_col_out:.4f}, row={mean_row_out:.4f} "
          f"({n_out} cadences)")

    if passed:
        print("  --- CENTROID VETTING PASSED: No significant centroid shift. ---")
    else:
        print("  !!! CENTROID VETTING WARNING: Significant centroid shift "
              "detected. Possible blended false positive.")
        print("  --- CENTROID VETTING FAILED: Centroid shift is too large. ---")

    result = {
        'centroid_shift': shift,
        'centroid_vetting_passed': passed,
        'n_in_transit': n_in,
        'n_out_transit': n_out,
        'mean_col_in': mean_col_in,
        'mean_row_in': mean_row_in,
        'mean_col_out': mean_col_out,
        'mean_row_out': mean_row_out
    }

    if n_in_frames > 0 and n_out_frames > 0:
        result['in_transit_img'] = in_frames_sum / n_in_frames
        result['out_transit_img'] = out_frames_sum / n_out_frames

    return result
