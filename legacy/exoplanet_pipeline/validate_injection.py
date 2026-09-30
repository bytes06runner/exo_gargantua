"""
validate_injection.py — Injection-Recovery Validation Suite
===========================================================

Builds a synthetic dataset by injecting transits into real TESS light curves
and evaluates the pipeline's detection capability.
"""

import numpy as np
import lightkurve as lk
import pandas as pd
import batman
import os

from exoplanet_pipeline.ingestion import preprocess_tess_data
from exoplanet_pipeline.denoise import decompose_and_filter
from exoplanet_pipeline.detection import run_bls_search, calculate_metrics_v2

def inject_transit(time, base_flux, period, t0, rp_rs, a_rs, inc):
    """
    Injects a synthetic planetary transit into a base flux array.
    """
    params = batman.TransitParams()
    params.t0 = t0
    params.per = period
    params.rp = rp_rs
    params.a = a_rs
    params.inc = inc
    params.ecc = 0.0
    params.w = 90.0
    params.u = [0.3, 0.1]
    params.limb_dark = "quadratic"
    
    m = batman.TransitModel(params, time)
    transit_flux = m.light_curve(params)
    return base_flux * transit_flux

def run_injection_recovery(tic_list, n_injections_per_star=5, output_csv="injection_results.csv"):
    """
    Runs the injection-recovery suite.
    Outputs a DataFrame measuring recovery rate vs. injected depth/SNR.
    """
    results = []
    
    for tic in tic_list:
        print(f"\nFetching base data for {tic}...")
        stitched_lc, _, raw_flux_err = preprocess_tess_data(tic)
        
        if stitched_lc is None:
            print(f"Skipping {tic} due to missing data.")
            continue
            
        time = stitched_lc.time.value
        base_flux = stitched_lc.flux.value
        
        for i in range(n_injections_per_star):
            print(f"  Running injection {i+1}/{n_injections_per_star} for {tic}...")
            # Randomly sample transit parameters
            period = np.random.uniform(1.0, 15.0)
            t0 = time[0] + np.random.uniform(0.1, period)
            rp_rs = np.random.uniform(0.01, 0.15) # Depths ~ 100 to 22500 ppm
            a_rs = np.random.uniform(5.0, 25.0)
            inc = np.random.uniform(85.0, 90.0)
            
            injected_flux = inject_transit(time, base_flux, period, t0, rp_rs, a_rs, inc)
            
            # Create a new lightkurve object for the pipeline
            lc_inj = lk.LightCurve(time=stitched_lc.time, flux=injected_flux, flux_err=stitched_lc.flux_err)
            
            try:
                filtered_lc, _, noise_stats = decompose_and_filter(lc_inj, raw_flux_err)
                bls_results = run_bls_search(filtered_lc)
                tsnr, _, _ = calculate_metrics_v2(bls_results, fap_confidence=None)
                
                # Check recovery (within 1% in period, close in t0)
                recovered_period = bls_results['period'].value
                recovered_t0 = bls_results['t0'].value
                
                p_match = abs(recovered_period - period) / period < 0.01
                # Account for phase wrap
                t0_diff = abs((recovered_t0 - t0) % period)
                t0_match = t0_diff < 0.1 or t0_diff > (period - 0.1)
                
                recovered = p_match and t0_match
                
                results.append({
                    'tic': tic,
                    'inj_period': period,
                    'inj_t0': t0,
                    'inj_rp_rs': rp_rs,
                    'inj_depth': rp_rs**2,
                    'rec_period': recovered_period,
                    'rec_t0': recovered_t0,
                    'recovered': int(recovered),
                    'bls_snr': bls_results['snr'].value,
                    'tsnr': tsnr
                })
                print(f"    -> Recovered: {recovered} (inj_P={period:.2f}, rec_P={recovered_period:.2f})")
            except Exception as e:
                print(f"    -> Injection failed or pipeline crashed: {e}")
                
    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv(output_csv, index=False)
        print(f"\nInjection-recovery results saved to {output_csv}")
        
        # Calculate summary statistics
        recovery_rate = df['recovered'].mean()
        print(f"Overall Recovery Rate: {recovery_rate:.1%}")
    
    return df

if __name__ == "__main__":
    # Example "quiet" stars for a quick run
    quiet_stars = ["TIC 270383187", "TIC 25155310"]
    run_injection_recovery(quiet_stars, n_injections_per_star=2)
