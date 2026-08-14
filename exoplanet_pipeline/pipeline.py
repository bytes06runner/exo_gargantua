"""
pipeline.py — Full Pipeline Orchestrator
=========================================

Wraps all phases into a single function, run_full_pipeline(target_star_id),
that calls each module in order and threads outputs through explicit
function arguments. Returns a single dict containing every intermediate
result.

This replaces the notebook global-variable pattern with proper function
composition (addresses Fix 7 from the original scope).
"""

import numpy as np
import warnings

from exoplanet_pipeline.ingestion import preprocess_tess_data, fetch_stellar_parameters
from exoplanet_pipeline.denoise import decompose_and_filter
from exoplanet_pipeline.detection import (
    run_bls_search, compute_fap_bootstrap,
    calculate_metrics_v2, check_exoplanet_alert
)
from exoplanet_pipeline.vetting import (
    run_phase_3_vetting, visualize_vetting, run_centroid_vetting
)
from exoplanet_pipeline.estimation import run_mcmc_estimation, derive_physical_parameters
from exoplanet_pipeline.report import generate_summary_report, generate_publication_figure


def run_full_pipeline(target_star_id, run_mcmc=True, run_centroid=True,
                      run_fap=True, n_fap_trials=200, n_mcmc_steps=2000,
                      run_report=False):
    """
    Runs the complete TESS exoplanet transit-detection pipeline from
    data ingestion through vetting and parameter estimation.

    Each phase's outputs are threaded explicitly as function arguments
    to the next phase — no global state is used.

    Parameters
    ----------
    target_star_id : str
        TESS target identifier (e.g., "TIC 25155310").
    run_mcmc : bool
        Whether to run MCMC parameter estimation (Fix 5). Disable for
        faster validation runs.
    run_centroid : bool
        Whether to run centroid vetting (Fix 3). Requires downloading
        target pixel files, which is slow.
    run_fap : bool
        Whether to run FAP bootstrap (Fix 2). Takes ~5-10 minutes.
    n_fap_trials : int
        Number of bootstrap permutations for FAP calculation.
    n_mcmc_steps : int
        Number of MCMC steps for parameter estimation.

    Returns
    -------
    results : dict or None
        Dictionary containing all intermediate results:
        - 'target': target_star_id
        - 'stitched_lc': the cleaned, normalized light curve
        - 'raw_flux_err': pre-normalization flux errors
        - 'noise_stats': photon noise, systematic/residual std
        - 'filtered_lc': the filtered light curve
        - 'residual_flux': flux minus systematic drift
        - 'bls_results': BLS best-fit parameters
        - 'tsnr', 'confidence', 'depth': detection metrics
        - 'fap', 'null_powers': FAP bootstrap results (if run)
        - 'vetting_results': odd/even, secondary eclipse results
        - 'centroid_results': centroid shift analysis (if run)
        - 'posteriors': MCMC parameter posteriors (if run)
        Returns None if no data is found or pipeline fails early.
    """
    results = {'target': target_star_id}

    # ========================================
    # PHASE 1: Data Ingestion
    # ========================================
    print(f"\n{'='*60}")
    print(f"PHASE 1: Data Ingestion — {target_star_id}")
    print(f"{'='*60}")

    stitched_lc, lc_collection, raw_flux_err = preprocess_tess_data(
        target_star_id
    )

    if stitched_lc is None:
        print(f"No data available for {target_star_id}. Pipeline aborted.")
        return None

    results['stitched_lc'] = stitched_lc
    results['lc_collection'] = lc_collection
    results['raw_flux_err'] = raw_flux_err

    print(f"  Stitched light curve: {len(stitched_lc)} cadences")

    # ========================================
    # PHASE 2: Noise Decomposition & Filtering
    # ========================================
    print(f"\n{'='*60}")
    print(f"PHASE 2: Noise Decomposition — {target_star_id}")
    print(f"{'='*60}")

    filtered_lc, residual_flux, noise_stats = decompose_and_filter(
        stitched_lc, raw_flux_err=raw_flux_err
    )

    results['filtered_lc'] = filtered_lc
    results['residual_flux'] = residual_flux
    results['noise_stats'] = noise_stats

    # Fix 1 verification: print the photon noise value
    print(f"  Photon noise (avg): {noise_stats['photon_noise_avg']:.6f}")
    print(f"  Systematic std:     {noise_stats['systematic_std']:.6f}")
    print(f"  Residual std:       {noise_stats['residual_std']:.6f}")

    # ========================================
    # PHASE 3: BLS Transit Search
    # ========================================
    print(f"\n{'='*60}")
    print(f"PHASE 3: BLS Transit Search — {target_star_id}")
    print(f"{'='*60}")

    bls_results = run_bls_search(filtered_lc)
    results['bls_results'] = bls_results

    print(f"  Best-fit period:   {bls_results['period'].value:.6f} d")
    print(f"  Best-fit t0:       {bls_results['t0'].value:.4f}")
    print(f"  Best-fit depth:    {bls_results['depth'].value:.6f}")
    print(f"  Best-fit duration: {bls_results['duration'].value:.6f} d")
    print(f"  BLS SNR (power):   {bls_results['snr'].value:.4f}")

    # Fix 2: FAP bootstrap confidence
    if run_fap:
        print(f"\n  Computing FAP via {n_fap_trials}-trial bootstrap...")
        observed_power = bls_results['snr'].value
        fap, confidence, null_powers = compute_fap_bootstrap(
            filtered_lc, observed_power, n_trials=n_fap_trials
        )
        results['fap'] = fap
        results['null_powers'] = null_powers
    else:
        confidence = None

    tsnr, confidence_final, depth = calculate_metrics_v2(
        bls_results, fap_confidence=confidence if run_fap else None
    )
    results['tsnr'] = tsnr
    results['confidence'] = confidence_final
    results['depth'] = depth

    check_exoplanet_alert(tsnr, confidence_final, target_star_id)

    # ========================================
    # PHASE 3b: Vetting
    # ========================================
    print(f"\n{'='*60}")
    print(f"PHASE 3b: Vetting — {target_star_id}")
    print(f"{'='*60}")

    vetting_results = run_phase_3_vetting(
        filtered_lc, bls_results['period'], bls_results['t0']
    )
    results['vetting_results'] = vetting_results

    visualize_vetting(vetting_results, target_star_id)

    # Fix 3: Centroid vetting
    if run_centroid:
        print(f"\n  --- Centroid Vetting ---")
        centroid_results = run_centroid_vetting(
            target_star_id,
            bls_results['period'],
            bls_results['t0'],
            bls_results['duration']
        )
        results['centroid_results'] = centroid_results
    else:
        results['centroid_results'] = {'centroid_vetting_passed': True,
                                        'skipped': True}

    # ========================================
    # PHASE 4: MCMC Parameter Estimation
    # ========================================
    if run_mcmc:
        print(f"\n{'='*60}")
        print(f"PHASE 4: MCMC Parameter Estimation — {target_star_id}")
        print(f"{'='*60}")

        try:
            posteriors, sampler = run_mcmc_estimation(
                filtered_lc, bls_results, raw_flux_err=raw_flux_err,
                n_steps=n_mcmc_steps
            )
            results['posteriors'] = posteriors
            results['sampler'] = sampler
        except Exception as e:
            print(f"  MCMC failed: {e}")
            results['posteriors'] = None
            results['mcmc_error'] = str(e)
            
            # Use dummy posteriors for reporting if MCMC fails
            posteriors = None
    else:
        posteriors = None
        sampler = None

    # ========================================
    # PHASE 4b: Physical Parameters
    # ========================================
    stellar_params = fetch_stellar_parameters(target_star_id)
    results['stellar_params'] = stellar_params

    derived_params = None
    if posteriors is not None:
        try:
            derived_params = derive_physical_parameters(posteriors, stellar_params)
            results['derived_physical_parameters'] = derived_params
        except Exception as e:
            print(f"  Error deriving physical params: {e}")

    # ========================================
    # PHASE 4c: Reporting
    # ========================================
    if run_report:
        print(f"\n{'='*60}")
        print(f"PHASE 4c: Candidate Reporting — {target_star_id}")
        print(f"{'='*60}")
        
        generate_summary_report(
            target_star_id, 
            bls_results, 
            vetting_results, 
            centroid_results if run_centroid else {}, 
            posteriors, 
            derived_params
        )
        
        generate_publication_figure(
            target_star_id, 
            filtered_lc, 
            sampler, 
            vetting_results, 
            centroid_results if run_centroid else {}
        )

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE — {target_star_id}")
    print(f"{'='*60}")
    
    print(f"  TSNR:       {tsnr:.2f}")
    if confidence_final is not None:
        print(f"  Confidence: {confidence_final:.4f}")
    print(f"  Depth:      {depth:.6f}")
    print(f"  Odd/Even diff: {vetting_results.get('depth_diff', 0):.5f}")
    print(f"  Secondary eclipse: {vetting_results.get('secondary_eclipse_sigma', 0):.2f}σ")
    
    if run_centroid and 'centroid_shift' in centroid_results:
        shift = centroid_results['centroid_shift']
        passed = centroid_results['centroid_vetting_passed']
        print(f"  Centroid shift: {shift:.4f} pix (passed: {passed})")

    if posteriors:
        print(f"  MCMC posteriors available for: {list(posteriors.keys())}")

    return results
