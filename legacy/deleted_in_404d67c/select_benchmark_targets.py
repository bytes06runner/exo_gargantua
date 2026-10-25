#!/usr/bin/env python3
"""
select_benchmark_targets.py — Stratified Sample Selection for Benchmarking
==========================================================================
Queries the NASA Exoplanet Archive TOI table for:
- 500 Confirmed / Known Planets (CP / KP), stratified across depths (shallow, medium, deep)
  and periods, including sub-Neptunes and super-Earths.
- 500 Certified False Positives (FP), varied across astrophysical mechanisms.

Filters out targets with > 10 SPOC sectors to keep execution fast and bounded.
Locks in ground truth and saves to benchmark_targets.json and benchmark_targets.csv.
"""

import os
import json
import pyvo as vo
import pandas as pd
import lightkurve as lk
import warnings
warnings.simplefilter("ignore")

def main():
    print("Querying NASA Exoplanet Archive TOI table via TAP...")
    tap = vo.dal.TAPService('https://exoplanetarchive.ipac.caltech.edu/TAP')
    
    query = """
    SELECT tid, toi, toipfx, tfopwg_disp, st_tmag, pl_orbper, pl_trandurh, pl_trandep, pl_rade, pl_eqt, st_teff, st_rad, sectors
    FROM toi
    WHERE tfopwg_disp IN ('CP', 'KP', 'FP')
      AND st_tmag < 13.0
      AND pl_orbper > 0.5
      AND pl_orbper < 20.0
      AND pl_trandep > 100.0
    """
    res = tap.search(query)
    df = res.to_table().to_pandas()
    print(f"Retrieved {len(df)} candidate TOI records.")

    # Remove duplicates on TID to have unique host stars
    df = df.sort_values(by=['st_tmag']).drop_duplicates(subset=['tid'])

    # Separate CP/KP and FP
    planets_df = df[df['tfopwg_disp'].isin(['CP', 'KP'])].copy()
    fps_df = df[df['tfopwg_disp'] == 'FP'].copy()
    
    print(f"Unique Host Stars -> Confirmed/Known Planets: {len(planets_df)}, False Positives: {len(fps_df)}")

    import time
    def _api_retry(func, *args, **kwargs):
        retries = [5, 15, 60]
        for delay in retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"  API Error: {e}. Retrying in {delay}s...")
                time.sleep(delay)
        return func(*args, **kwargs)

    # Function to check SPOC sectors
    def get_spoc_sector_count(tid):
        try:
            sr = _api_retry(lk.search_lightcurve, f"TIC {tid}", mission='TESS', author='SPOC')
            if len(sr) == 0:
                return 0
            # Get unique sectors
            if hasattr(sr, 'table') and 'sequence_number' in sr.table.colnames:
                unique_secs = len(set(sr.table['sequence_number']))
            else:
                unique_secs = len(sr)
            return unique_secs
        except Exception as e:
            return 0

    # Stratify Confirmed Planets:
    # 1. Shallow transits / smaller planets: depth < 1500 ppm (e.g. sub-Neptunes / super-Earths) -> 5 targets
    # 2. Medium transits: depth 1500 - 6000 ppm (e.g. Neptunes / sub-Saturns) -> 5 targets
    # 3. Deep transits / hot Jupiters: depth > 6000 ppm -> 5 targets
    
    selected_planets = []
    
    print("\n--- Selecting 500 Confirmed Planets (CP/KP) with 1 <= SPOC sectors <= 10 ---")
    
    # Bins
    bins = [
        ("Shallow (<1500 ppm)", planets_df[planets_df['pl_trandep'] < 1500.0]),
        ("Medium (1500-6000 ppm)", planets_df[(planets_df['pl_trandep'] >= 1500.0) & (planets_df['pl_trandep'] < 6000.0)]),
        ("Deep (>6000 ppm)", planets_df[planets_df['pl_trandep'] >= 6000.0])
    ]
    
    for bin_idx, (bin_name, bin_df) in enumerate(bins):
        target_count = 167 if bin_idx < 2 else 166
        count = 0
        print(f"Searching in bin: {bin_name} (Pool size: {len(bin_df)}, Target: {target_count})...")
        # Shuffle with fixed seed for reproducibility
        sampled = bin_df.sample(frac=1.0, random_state=42)
        for _, row in sampled.iterrows():
            tid = int(row['tid'])
            n_sec = get_spoc_sector_count(tid)
            if 1 <= n_sec <= 10:
                selected_planets.append({
                    'target_id': f"TIC {tid}",
                    'tid': tid,
                    'toi': float(row['toi']),
                    'catalog_disp': str(row['tfopwg_disp']),
                    'ground_truth': 'PLANET',
                    'bin': bin_name,
                    'st_tmag': float(row['st_tmag']),
                    'pl_orbper': float(row['pl_orbper']),
                    'pl_trandep_ppm': float(row['pl_trandep']),
                    'pl_rade': float(row['pl_rade']) if not pd.isna(row['pl_rade']) else None,
                    'spoc_sectors': n_sec
                })
                count += 1
                if count % 10 == 0:
                    print(f"  Selected {count}/{target_count} in {bin_name}")
                if count >= target_count:
                    break
        if count < target_count:
            print(f"  Warning: Only found {count} targets for {bin_name}")

    # Stratify False Positives:
    # 1. Shallow FPs (< 2500 ppm) -> 5 targets
    # 2. Medium FPs (2500 - 10000 ppm) -> 5 targets
    # 3. Deep FPs (> 10000 ppm, e.g. EBs) -> 5 targets
    selected_fps = []
    print("\n--- Selecting 500 Certified False Positives (FP) with 1 <= SPOC sectors <= 10 ---")
    fp_bins = [
        ("Shallow (<2500 ppm)", fps_df[fps_df['pl_trandep'] < 2500.0]),
        ("Medium (2500-10000 ppm)", fps_df[(fps_df['pl_trandep'] >= 2500.0) & (fps_df['pl_trandep'] < 10000.0)]),
        ("Deep (>10000 ppm)", fps_df[fps_df['pl_trandep'] >= 10000.0])
    ]
    
    for bin_idx, (bin_name, bin_df) in enumerate(fp_bins):
        target_count = 167 if bin_idx < 2 else 166
        count = 0
        print(f"Searching in bin: {bin_name} (Pool size: {len(bin_df)}, Target: {target_count})...")
        sampled = bin_df.sample(frac=1.0, random_state=42)
        for _, row in sampled.iterrows():
            tid = int(row['tid'])
            n_sec = get_spoc_sector_count(tid)
            if 1 <= n_sec <= 10:
                selected_fps.append({
                    'target_id': f"TIC {tid}",
                    'tid': tid,
                    'toi': float(row['toi']),
                    'catalog_disp': str(row['tfopwg_disp']),
                    'ground_truth': 'FALSE_POSITIVE',
                    'bin': bin_name,
                    'st_tmag': float(row['st_tmag']),
                    'pl_orbper': float(row['pl_orbper']),
                    'pl_trandep_ppm': float(row['pl_trandep']),
                    'pl_rade': float(row['pl_rade']) if not pd.isna(row['pl_rade']) else None,
                    'spoc_sectors': n_sec
                })
                count += 1
                if count % 10 == 0:
                    print(f"  Selected {count}/{target_count} in {bin_name}")
                if count >= target_count:
                    break

    all_targets = selected_planets + selected_fps
    print(f"\nTotal targets locked in: {len(all_targets)} ({len(selected_planets)} Planets, {len(selected_fps)} False Positives)")
    
    # Save to JSON and CSV
    with open('benchmark_targets.json', 'w') as f:
        json.dump(all_targets, f, indent=4)
        
    df_out = pd.DataFrame(all_targets)
    df_out.to_csv('benchmark_targets.csv', index=False)
    print("Saved benchmark targets to benchmark_targets.json and benchmark_targets.csv.")

if __name__ == "__main__":
    main()
