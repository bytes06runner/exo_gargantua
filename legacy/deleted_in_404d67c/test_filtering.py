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
m = 1.0
trial_p = p_max * m
narrow_p = np.linspace(trial_p * 0.95, trial_p * 1.05, 2000)
pg_sub = lc_bls.to_periodogram(method='bls', period=narrow_p, frequency_factor=5000)

best_p = pg_sub.period_at_max_power.value
best_t0 = pg_sub.transit_time_at_max_power.value
best_dur = pg_sub.duration_at_max_power.value

E = np.round((time_val - best_t0) / best_p)
in_transit = np.abs(time_val - (best_t0 + E * best_p)) < (best_dur / 2)
unique_epochs = len(np.unique(E[in_transit]))

print(f"Unique Epochs: {unique_epochs}")

E_in_transit = E[in_transit]
flux_in_transit = flux_val[in_transit]

odd_mask = (E_in_transit % 2) != 0
even_mask = (E_in_transit % 2) == 0

odd_depths = 1.0 - flux_in_transit[odd_mask]
even_depths = 1.0 - flux_in_transit[even_mask]

odd_mean = np.nanmean(odd_depths)
even_mean = np.nanmean(even_depths)

var_odd = np.nanvar(odd_depths)
var_even = np.nanvar(even_depths)

pooled_se = np.sqrt(var_odd / len(odd_depths) + var_even / len(even_depths))
diff_sigma = np.abs(odd_mean - even_mean) / pooled_se if pooled_se > 0 else 0
rel_diff = np.abs(odd_mean - even_mean) / max(np.abs(odd_mean), np.abs(even_mean), 1e-6)

print(f"Odd Mean: {odd_mean:.6f}, Even Mean: {even_mean:.6f}")
print(f"Diff Sigma: {diff_sigma:.2f}, Rel Diff: {rel_diff:.2f}")

if diff_sigma > 2.0 and rel_diff > 0.1:
    print("FAILED ODD/EVEN CHECK")
else:
    print("PASSED ODD/EVEN CHECK")
