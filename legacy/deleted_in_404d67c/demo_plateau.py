import json
import numpy as np
import lightkurve as lk
from exoplanet_pipeline.data_loader import build_stitched_lightcurve
from exoplanet_pipeline.preprocessing import preprocess_lightcurve

target = "TIC 100100827"
with open('benchmark_targets_full.json', 'r') as f:
    targets = json.load(f)
    
t_info = next(t for t in targets if t['target_id'] == target)
lc = build_stitched_lightcurve(target, t_info['spoc_sectors'])
clean_lc = preprocess_lightcurve(lc)

lc_bls = clean_lc
if len(lc_bls) > 100000:
    step = max(1, len(lc_bls) // 50000)
    lc_bls = lc_bls[::step]

time_val = lc_bls.time.value
flux_val = lc_bls.flux.value
valid = ~np.isnan(flux_val)
time_val = time_val[valid]
flux_val = flux_val[valid]

p_max = 0.94145
trial_multipliers = [1.0, 5.0]

for m in trial_multipliers:
    trial_p = p_max * m
    narrow_p = np.linspace(trial_p * 0.95, trial_p * 1.05, 2000)
    pg_sub = lc_bls.to_periodogram(method='bls', period=narrow_p, frequency_factor=5000)
    
    best_p = pg_sub.period_at_max_power.value
    best_t0 = pg_sub.transit_time_at_max_power.value
    best_dur = pg_sub.duration_at_max_power.value
    
    E = np.round((time_val - best_t0) / best_p)
    in_transit = np.abs(time_val - (best_t0 + E * best_p)) < (best_dur / 2)
    n_in = np.sum(in_transit)
    
    best_depth = 1.0 - np.nanmean(flux_val[in_transit])
    out_std = np.nanstd(flux_val[~in_transit])
    best_snr = (best_depth * np.sqrt(n_in)) / out_std
    
    print(f"Multiplier {m}: P={best_p:.4f}, Depth={best_depth:.6f}, N_in={n_in}, SNR={best_snr:.2f}")
