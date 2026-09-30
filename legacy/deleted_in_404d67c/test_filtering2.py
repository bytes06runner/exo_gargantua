import json
import numpy as np
import lightkurve as lk
from pipeline.data_loader import build_stitched_lightcurve
from pipeline.preprocessing import preprocess_lightcurve

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
trial_multipliers = [1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0]

for m in trial_multipliers:
    trial_p = p_max * m
    narrow_p = np.linspace(trial_p * 0.95, trial_p * 1.05, 2000)
    pg_sub = lc_bls.to_periodogram(method='bls', period=narrow_p, frequency_factor=5000)
    
    best_p = pg_sub.period_at_max_power.value
    best_t0 = pg_sub.transit_time_at_max_power.value
    best_dur = pg_sub.duration_at_max_power.value
    
    E = np.round((time_val - best_t0) / best_p)
    in_transit = np.abs(time_val - (best_t0 + E * best_p)) < (best_dur / 2)
    unique_epochs = len(np.unique(E[in_transit]))
    
    if unique_epochs < 3:
        print(f"m={m:.2f} P={best_p:.4f}: FAILED EPOCH GATE (epochs={unique_epochs})")
        continue
        
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
                    print(f"m={m:.2f} P={best_p:.4f}: FAILED ODD/EVEN (diff_sigma={diff_sigma:.2f}, rel_diff={rel_diff:.2f})")
                    continue
                    
    print(f"m={m:.2f} P={best_p:.4f}: PASSED ALL GATES")
