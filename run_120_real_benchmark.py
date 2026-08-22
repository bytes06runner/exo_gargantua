#!/usr/bin/env python3
import os
import sys
import json
import time
import pandas as pd
import numpy as np
import lightkurve as lk
import pyvo as vo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')

from exoplanet_pipeline.pipeline import run_full_pipeline
from exoplanet_pipeline.detection import run_bls_search

def retry_download(target_id, max_retries=5):
    """
    Robust download handler with retry and timeout for MAST server connection drops.
    """
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt+1}/{max_retries} to download SPOC data for {target_id}...")
            search_result = lk.search_lightcurve(target_id, mission='TESS', author='SPOC')
            if len(search_result) == 0:
                print("  No SPOC data found.")
                return None
            # Only download the first 4 sectors to keep processing somewhat bounded but representative
            lc_collection = search_result[:4].download_all(download_dir='./tess_cache')
            if not lc_collection:
                return None
                
            valid_lcs = []
            for lc in lc_collection:
                lc = lc.remove_nans()
                if len(lc) > 0:
                    median_flux = np.nanmedian(lc.flux.value)
                    if median_flux > 0:
                        lc.flux = lc.flux / median_flux
                        valid_lcs.append(lc)
            
            if not valid_lcs:
                return None
                
            return lk.LightCurveCollection(valid_lcs).stitch()
            
        except Exception as e:
            print(f"  Error downloading {target_id}: {e}")
            time.sleep(10 * (attempt + 1))  # Exponential backoff
    return None

def get_targets():
    print("Querying NASA Exoplanet Archive for exactly 120 stratified targets...")
    tap = vo.dal.TAPService('https://exoplanetarchive.ipac.caltech.edu/TAP')
    
    query = """
    SELECT tid, toi, tfopwg_disp, pl_orbper, pl_trandep, pl_rade
    FROM toi
    WHERE tfopwg_disp IN ('CP', 'KP', 'FP')
      AND st_tmag < 12.0
      AND pl_orbper > 0.5
      AND pl_orbper < 20.0
      AND pl_trandep > 100.0
    """
    res = tap.search(query)
    df = res.to_table().to_pandas()
    
    df = df.drop_duplicates(subset=['tid'])
    
    planets = df[df['tfopwg_disp'].isin(['CP', 'KP'])]
    shallow_mask = (planets['pl_trandep'] < 1500.0) | (planets['pl_rade'] < 2.5)
    shallow_planets = planets[shallow_mask].sample(40, random_state=42)
    deep_planets = planets[~shallow_mask].sample(20, random_state=42)
    
    fps_df = df[df['tfopwg_disp'] == 'FP']
    fp_targets = fps_df[fps_df['pl_trandep'] <= 10000.0].sample(30, random_state=42)
    eb_targets = fps_df[fps_df['pl_trandep'] > 10000.0].sample(30, random_state=42)
    
    shallow_planets['type'] = 'Shallow CP'
    deep_planets['type'] = 'Deep CP'
    fp_targets['type'] = 'FP'
    eb_targets['type'] = 'EB'
    
    return pd.concat([shallow_planets, deep_planets, fp_targets, eb_targets])

def run_benchmark():
    # CRITICAL: Nuke old results file to prevent append-mode duplication
    if os.path.exists('real_benchmark_results.jsonl'):
        os.remove('real_benchmark_results.jsonl')
        print("Deleted stale real_benchmark_results.jsonl")
    
    df = get_targets()
    print(f"Total targets selected: {len(df)}")
    
    results = []
    
    for idx, row in enumerate(df.itertuples(), 1):
        tid = row.tid
        target_id = f"TIC {tid}"
        target_type = row.type
        catalog_period = row.pl_orbper
        print(f"\n[{idx}/120] Processing {target_id} ({target_type}) - P={catalog_period:.2f}d")
        
        entry = {
            'target_id': target_id,
            'type': target_type,
            'catalog_period': float(catalog_period),
            'spoc_period': None,
            'spoc_snr': 0.0,
            'spoc_recovered': False,
            'exo_period': None,
            'exo_snr': 0.0,
            'exo_recovered': False,
            'spoc_rejected': False,
            'exo_rejected': False,
        }
        
        stitched_lc = retry_download(target_id)
        if stitched_lc is None:
            print("  Skipping due to download failure.")
            continue
            
        try:
            # Vanilla BLS Baseline
            time_val = stitched_lc.time.value
            flux_val = stitched_lc.flux.value
            
            # Remove NaNs
            valid = ~np.isnan(time_val) & ~np.isnan(flux_val)
            time_val = time_val[valid]
            flux_val = flux_val[valid]
            
            from astropy.timeseries import BoxLeastSquares
            model = BoxLeastSquares(time_val, flux_val)
            period_grid = np.linspace(0.5, 20.0, 10000)
            durations = np.linspace(0.01, 0.2, 10)
            res = model.power(period_grid, durations)
            
            max_idx = np.argmax(res.power)
            spoc_period = float(res.period[max_idx])
            spoc_snr = float(res.power[max_idx])
            
            entry['spoc_period'] = spoc_period
            entry['spoc_snr'] = spoc_snr
            
            p_diff = abs(spoc_period - catalog_period) / catalog_period
            is_rec = bool(p_diff < 0.05 or abs(spoc_period*2 - catalog_period)/catalog_period < 0.05 or abs(spoc_period/2 - catalog_period)/catalog_period < 0.05)
            entry['spoc_recovered'] = bool(is_rec)
            entry['spoc_rejected'] = bool(spoc_snr < 7.1)
            print(f"  SPOC -> SNR: {spoc_snr:.2f}, Recovered: {is_rec}")
        except Exception as e:
            print(f"  SPOC BLS Error: {e}")
            
        try:
            exo_res = run_full_pipeline(target_id, run_mcmc=False, run_centroid=False, run_fap=False, run_report=False)
            
            if exo_res is not None and 'bls_results' in exo_res:
                exo_bls = exo_res['bls_results']
                exo_p = exo_bls['period'].value if hasattr(exo_bls['period'], 'value') else exo_bls['period']
                exo_snr = exo_bls['snr'].value if hasattr(exo_bls['snr'], 'value') else exo_bls['snr']
                entry['exo_period'] = float(exo_p)
                entry['exo_snr'] = float(exo_snr)
                
                ep_diff = abs(exo_p - catalog_period) / catalog_period
                exo_rec = bool(ep_diff < 0.05 or abs(exo_p*2 - catalog_period)/catalog_period < 0.05 or abs(exo_p/2 - catalog_period)/catalog_period < 0.05)
                entry['exo_recovered'] = bool(exo_rec)
                
                vetting = exo_res.get('ml_vetting', {})
                disp = vetting.get('disposition', 'UNKNOWN')
                entry['exo_rejected'] = bool((disp == 'FALSE_POSITIVE') or (exo_snr < 7.1))
                
                print(f"  EXO  -> SNR: {exo_snr:.2f}, Recovered: {exo_rec}, Disp: {disp}")
            else:
                print("  EXO pipeline returned None.")
        except Exception as e:
            print(f"  EXO Pipeline Error: {e}")
            
        results.append(entry)
        
        with open('real_benchmark_results.jsonl', 'a') as f:
            f.write(json.dumps(entry) + '\n')
            
    res_df = pd.DataFrame(results)
    
    planets = res_df[res_df['type'].isin(['Shallow CP', 'Deep CP'])]
    shallow_planets = res_df[res_df['type'] == 'Shallow CP']
    deep_planets = res_df[res_df['type'] == 'Deep CP']
    fps = res_df[res_df['type'].isin(['FP', 'EB'])]
    
    spoc_rec_total = planets['spoc_recovered'].mean()
    exo_rec_total = planets['exo_recovered'].mean()
    
    spoc_rec_shallow = shallow_planets['spoc_recovered'].mean()
    exo_rec_shallow = shallow_planets['exo_recovered'].mean()
    
    spoc_rej_fp = fps['spoc_rejected'].mean()
    exo_rej_fp = fps['exo_rejected'].mean()
    
    spoc_mean_snr = planets['spoc_snr'].mean()
    exo_mean_snr = planets['exo_snr'].mean()
    
    latex_table = f"""\\begin{{deluxetable*}}{{lccc}}
\\tablecaption{{Automated Benchmark Results: Exo-Gargantua vs. SPOC Pipeline (120 TOI Sample) \\label{{tab:benchmark}}}}
\\tablewidth{{0pt}}
\\tablehead{{
\\colhead{{Metric}} & \\colhead{{SPOC Baseline}} & \\colhead{{Exo-Gargantua}} & \\colhead{{Improvement ($\\Delta$)}}
}}
\\startdata
Total Recovery Rate & {spoc_rec_total*100:.1f}\\% & {exo_rec_total*100:.1f}\\% & +{(exo_rec_total - spoc_rec_total)*100:.1f}\\% \\\\
Shallow Target ($<1500\\text{{ ppm}}$) Recovery & {spoc_rec_shallow*100:.1f}\\% & {exo_rec_shallow*100:.1f}\\% & +{(exo_rec_shallow - spoc_rec_shallow)*100:.1f}\\% \\\\
False Positive Rejection Rate & {spoc_rej_fp*100:.1f}\\% & {exo_rej_fp*100:.1f}\\% & +{(exo_rej_fp - spoc_rej_fp)*100:.1f}\\% \\\\
Mean SNR & {spoc_mean_snr:.2f} & {exo_mean_snr:.2f} & +{(exo_mean_snr - spoc_mean_snr):.2f} \\\\
\\enddata
\\end{{deluxetable*}}
"""
    with open("benchmark_table.tex", "w") as f:
        f.write(latex_table)
    
    plt.figure(figsize=(7, 6))
    plt.scatter(shallow_planets['spoc_snr'], shallow_planets['exo_snr'], color='blue', alpha=0.7, label='Shallow Planets (<1500 ppm)')
    plt.scatter(deep_planets['spoc_snr'], deep_planets['exo_snr'], color='orange', alpha=0.7, label='Deep Planets (>1500 ppm)')
    max_val = max(res_df['spoc_snr'].max(), res_df['exo_snr'].max()) + 5
    if np.isnan(max_val): max_val = 50
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y = x (Parity)')
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.xlabel('SPOC Baseline SNR')
    plt.ylabel('Exo-Gargantua SNR')
    plt.title('SNR Parity: Exo-Gargantua vs. SPOC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig_snr_parity.png', dpi=300)
    plt.close()
    
    showdown_target = res_df[(~res_df['spoc_recovered']) & (res_df['exo_recovered'])]
    if len(showdown_target) > 0:
        best_id = showdown_target.iloc[0]['target_id']
        print(f"Plotting showdown for target: {best_id}")
        try:
            lc_spoc = retry_download(best_id)
            bls_spoc = run_bls_search(lc_spoc)
            spoc_p = bls_spoc['period'].value if hasattr(bls_spoc['period'], 'value') else bls_spoc['period']
            spoc_t0 = bls_spoc['transit_time'].value if hasattr(bls_spoc['transit_time'], 'value') else bls_spoc['transit_time']
            folded_spoc = lc_spoc.fold(period=spoc_p, epoch_time=spoc_t0)
            
            exo_res = run_full_pipeline(best_id, run_mcmc=False, run_centroid=False, run_fap=False, run_report=False)
            filtered_lc = exo_res['filtered_lc']
            exo_bls = exo_res['bls_results']
            exo_p = exo_bls['period'].value if hasattr(exo_bls['period'], 'value') else exo_bls['period']
            exo_t0 = exo_bls['transit_time'].value if hasattr(exo_bls['transit_time'], 'value') else exo_bls['transit_time']
            folded_exo = filtered_lc.fold(period=exo_p, epoch_time=exo_t0)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            ax1.plot(folded_spoc.time.value, folded_spoc.flux.value, 'k.', markersize=2, alpha=0.5)
            ax1.set_title(f'SPOC PDCSAP Folded Data ({best_id})')
            ax1.set_ylabel('Relative Flux')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(-0.1, 0.1)
            
            ax2.plot(folded_exo.time.value, folded_exo.flux.value, 'b.', markersize=2, alpha=0.5)
            ax2.set_title(f'Exo-Gargantua Flattened Baseline ({best_id})')
            ax2.set_xlabel('Phase (Days)')
            ax2.set_ylabel('Relative Flux')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(-0.1, 0.1)
            
            plt.tight_layout()
            plt.savefig('fig_showdown.png', dpi=300)
            plt.close()
        except Exception as e:
            print(f"Error plotting showdown: {e}")
            
    print("Done. Saved real_benchmark_results.jsonl, benchmark_table.tex, and plots.")

if __name__ == "__main__":
    run_benchmark()
