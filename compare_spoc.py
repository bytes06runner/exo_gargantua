import os
import json
import numpy as np
import lightkurve as lk
from exoplanet_pipeline.detection import run_bls_search
import warnings
warnings.simplefilter('ignore')

def main():
    if not os.path.exists('benchmark_results.jsonl'):
        print("benchmark_results.jsonl not found.")
        return
        
    results = []
    with open('benchmark_results.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
                
    spoc_results = []
    
    print(f"Starting SPOC baseline comparison for {len(results)} targets...")
    for idx, res in enumerate(results, 1):
        target_id = res['target_id']
        catalog_period = res.get('catalog_period')
        if not catalog_period:
            continue
            
        print(f"[{idx}/{len(results)}] {target_id}")
        
        try:
            # Download SPOC light curves
            search_result = lk.search_lightcurve(target_id, mission='TESS', author='SPOC')
            if not search_result:
                print(f"  No SPOC data found for {target_id}.")
                continue
                
            lc_collection = search_result.download_all(download_dir='./tess_cache')
            if not lc_collection:
                print(f"  Failed to download SPOC data for {target_id}.")
                continue
                
            # Clean and stitch SPOC pdcsap_flux
            valid_lcs = []
            for lc in lc_collection:
                lc = lc.remove_nans()
                if len(lc) > 0:
                    # By default author='SPOC' returns PDCSAP flux in lc.flux
                    # We normalize it so BLS works correctly
                    median_flux = np.nanmedian(lc.flux.value)
                    if median_flux > 0:
                        lc.flux = lc.flux / median_flux
                        valid_lcs.append(lc)
                        
            if not valid_lcs:
                continue
                
            stitched_lc = lk.LightCurveCollection(valid_lcs).stitch()
            
            # Run BLS
            bls_res = run_bls_search(stitched_lc)
            spoc_period = bls_res['period'].value if hasattr(bls_res['period'], 'value') else bls_res['period']
            spoc_snr = bls_res['snr'].value if hasattr(bls_res['snr'], 'value') else bls_res['snr']
            
            # Check recovery
            period_diff = abs(spoc_period - catalog_period) / catalog_period
            # consider recovered if fundamental, half, or double
            is_recovered = bool(period_diff < 0.05 or abs(spoc_period*2 - catalog_period)/catalog_period < 0.05 or abs(spoc_period/2 - catalog_period)/catalog_period < 0.05)
            
            exo_period = res.get('bls_period', 0)
            exo_snr = res.get('bls_snr', 0)
            
            exo_recovered = False
            if exo_period is not None and exo_period > 0:
                exo_diff = abs(exo_period - catalog_period) / catalog_period
                exo_recovered = bool(exo_diff < 0.05 or abs(exo_period*2 - catalog_period)/catalog_period < 0.05 or abs(exo_period/2 - catalog_period)/catalog_period < 0.05)
            
            spoc_results.append({
                'target_id': target_id,
                'spoc_period': spoc_period,
                'spoc_snr': spoc_snr,
                'spoc_recovered': is_recovered,
                'exo_period': exo_period,
                'exo_snr': exo_snr,
                'exo_recovered': exo_recovered,
                'catalog_period': catalog_period
            })
            
            print(f"  SPOC SNR: {spoc_snr:.2f} | Recovered: {is_recovered}")
            print(f"  EXO SNR : {exo_snr:.2f} | Recovered: {exo_recovered}")
            
        except Exception as e:
            print(f"  Error processing {target_id}: {e}")

    # Summary
    if not spoc_results:
        print("No results to compare.")
        return
        
    spoc_rec = sum(1 for r in spoc_results if r['spoc_recovered'])
    exo_rec = sum(1 for r in spoc_results if r['exo_recovered'])
    
    print("\n--- COMPARISON SUMMARY ---")
    print(f"Total Targets Evaluated: {len(spoc_results)}")
    print(f"SPOC Pipeline Recovery: {spoc_rec}/{len(spoc_results)} ({(spoc_rec/len(spoc_results))*100:.1f}%)")
    print(f"Exo-Gargantua Recovery: {exo_rec}/{len(spoc_results)} ({(exo_rec/len(spoc_results))*100:.1f}%)")
    
    spoc_mean_snr = np.mean([r['spoc_snr'] for r in spoc_results if r['spoc_snr'] is not None])
    exo_mean_snr = np.mean([r['exo_snr'] for r in spoc_results if r['exo_snr'] is not None])
    
    print(f"SPOC Mean SNR: {spoc_mean_snr:.2f}")
    print(f"Exo-Gargantua Mean SNR: {exo_mean_snr:.2f}")

    with open('baseline_comparison_results.json', 'w') as f:
        json.dump(spoc_results, f, indent=2)
        
    # Write summary artifact text
    with open('baseline_summary.md', 'w') as f:
        f.write("# Baseline Comparison vs SPOC\n\n")
        f.write(f"- **Total Targets Evaluated**: {len(spoc_results)}\n")
        f.write(f"- **SPOC Pipeline Recovery**: {spoc_rec}/{len(spoc_results)} ({(spoc_rec/len(spoc_results))*100:.1f}%)\n")
        f.write(f"- **Exo-Gargantua Recovery**: {exo_rec}/{len(spoc_results)} ({(exo_rec/len(spoc_results))*100:.1f}%)\n")
        f.write(f"- **SPOC Mean SNR**: {spoc_mean_snr:.2f}\n")
        f.write(f"- **Exo-Gargantua Mean SNR**: {exo_mean_snr:.2f}\n")

if __name__ == '__main__':
    main()
