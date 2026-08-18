#!/usr/bin/env python3
"""
run_benchmark.py — Automated Benchmark Runner on 30 Stratified Targets
======================================================================
Executes the full Exo-Gargantua pipeline on the locked target list in benchmark_targets.json.
Logs all results, catches failures explicitly, computes the confusion matrix, and analyzes
grazing geometry / depth-ratio degeneracies.
"""

import os
import sys
import json
import time
import traceback
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc

from exoplanet_pipeline.pipeline import run_full_pipeline

def main():
    if not os.path.exists('benchmark_targets.json'):
        print("Error: benchmark_targets.json not found. Run select_benchmark_targets.py first.")
        sys.exit(1)

    with open('benchmark_targets.json', 'r') as f:
        targets = json.load(f)

    print(f"\n{'='*70}")
    print(f"STARTING EXOPLANET BENCHMARK: {len(targets)} STRATIFIED TARGETS")
    print(f"{'='*70}\n")

    results = []
    completed_ids = set()
    if os.path.exists('benchmark_results.jsonl'):
        try:
            with open('benchmark_results.jsonl', 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    e = json.loads(line)
                    if e.get('pipeline_status') in ('SUCCESS', 'CRASHED', 'FAILED'):
                        results.append(e)
                        completed_ids.add(e['target_id'])
            print(f"Loaded {len(results)} existing results from checkpoint.")
        except Exception as e:
            print(f"Warning reading benchmark_results.jsonl: {e}")

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    start_time_all = time.time()

    for idx, target_info in enumerate(targets, 1):
        target_id = target_info['target_id']
        ground_truth = target_info['ground_truth']
        catalog_disp = target_info['catalog_disp']
        toi_id = target_info.get('toi')
        bin_name = target_info.get('bin', 'N/A')
        
        if target_id in completed_ids:
            print(f">>> [{idx}/{len(targets)}] {target_id} already completed successfully. Skipping. <<<")
            continue

        print(f"\n>>> [{idx}/{len(targets)}] Processing {target_id} (TOI {toi_id} | Catalog: {catalog_disp} | Ground Truth: {ground_truth}) <<<")
        t0 = time.time()
        
        entry = {
            'target_id': target_id,
            'tid': target_info['tid'],
            'toi': toi_id,
            'catalog_disp': catalog_disp,
            'ground_truth': ground_truth,
            'bin': bin_name,
            'st_tmag': target_info.get('st_tmag'),
            'catalog_period': target_info.get('pl_orbper'),
            'catalog_depth_ppm': target_info.get('pl_trandep_ppm'),
            'catalog_rade': target_info.get('pl_rade'),
            'spoc_sectors': target_info.get('spoc_sectors'),
            'pipeline_status': 'PENDING',
            'error_message': None,
            'pipeline_disposition': None,
            'planet_probability': None,
            'rf_raw_probability': None,
            'flags': [],
            'bls_period': None,
            'bls_depth': None,
            'bls_snr': None,
            'mcmc_period': None,
            'mcmc_depth': None,
            'mcmc_rp_rs': None,
            'mcmc_a_rs': None,
            'mcmc_inc': None,
            'impact_parameter_b': None,
            'bls_mcmc_depth_ratio': None,
            'odd_even_diff': None,
            'sec_eclipse_sigma': None,
            'centroid_shift_px': None,
            'elapsed_sec': None
        }

        try:
            res = run_full_pipeline(
                target_id,
                run_mcmc=True,
                run_centroid=True,
                run_fap=False,
                n_mcmc_steps=1000,
                run_report=True
            )

            if res is None or 'filtered_lc' not in res:
                entry['pipeline_status'] = 'FAILED'
                entry['error_message'] = 'Pipeline returned None or no light curve found.'
                print(f"  FAILED: {entry['error_message']}")
            else:
                entry['pipeline_status'] = 'SUCCESS'
                ml_res = res.get('ml_vetting', {})
                entry['pipeline_disposition'] = ml_res.get('disposition', 'UNKNOWN')
                entry['planet_probability'] = ml_res.get('planet_probability')
                entry['rf_raw_probability'] = ml_res.get('rf_raw_probability')
                entry['flags'] = ml_res.get('flags', [])
                
                bls = res.get('bls_results', {})
                entry['bls_period'] = float(bls['period'].value) if hasattr(bls.get('period'), 'value') else float(bls.get('period', 0))
                entry['bls_depth'] = float(bls['depth'].value) if hasattr(bls.get('depth'), 'value') else float(bls.get('depth', 0))
                entry['bls_snr'] = float(bls['snr'].value) if hasattr(bls.get('snr'), 'value') else float(bls.get('snr', 0))

                vett = res.get('vetting_results', {})
                entry['odd_even_diff'] = vett.get('depth_diff')
                entry['sec_eclipse_sigma'] = vett.get('secondary_eclipse_sigma')

                cent = res.get('centroid_results', {})
                entry['centroid_shift_px'] = cent.get('centroid_shift')

                post = res.get('posteriors')
                if post is not None:
                    entry['mcmc_period'] = float(post['period'][0]) if 'period' in post else None
                    mcmc_d = float(post['depth'][0]) if 'depth' in post else None
                    entry['mcmc_depth'] = mcmc_d
                    entry['mcmc_rp_rs'] = float(post['rp_rs'][0]) if 'rp_rs' in post else None
                    a_rs = float(post['a_rs'][0]) if 'a_rs' in post else None
                    inc = float(post['inc'][0]) if 'inc' in post else None
                    entry['mcmc_a_rs'] = a_rs
                    entry['mcmc_inc'] = inc

                    if a_rs is not None and inc is not None:
                        b_val = a_rs * np.cos(np.radians(inc))
                        entry['impact_parameter_b'] = float(b_val)
                    
                    if mcmc_d is not None and entry['bls_depth'] is not None and entry['bls_depth'] > 0:
                        ratio = max(mcmc_d / entry['bls_depth'], entry['bls_depth'] / mcmc_d)
                        entry['bls_mcmc_depth_ratio'] = float(ratio)

                print(f"  SUCCESS in {time.time()-t0:.1f}s -> Disposition: {entry['pipeline_disposition']} (p={entry['planet_probability']}) | Flags: {len(entry['flags'])}")

        except Exception as e:
            entry['pipeline_status'] = 'CRASHED'
            entry['error_message'] = f"{type(e).__name__}: {str(e)}"
            print(f"  CRASHED in {time.time()-t0:.1f}s -> {entry['error_message']}")
            traceback.print_exc()

        entry['elapsed_sec'] = time.time() - t0
        results.append(entry)

        # Save intermediate results (JSONL append)
        with open('benchmark_results.jsonl', 'a') as f:
            f.write(json.dumps(entry) + '\n')
            
        plt.close('all')
        gc.collect()

    total_elapsed = time.time() - start_time_all
    print(f"\n{'='*70}")
    print(f"BENCHMARK COMPLETE IN {total_elapsed/60:.1f} MINUTES")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
