"""Diagnose why the TESS Perigee Veto didn't fire for TIC 456945304."""
import warnings
warnings.simplefilter('ignore')
import numpy as np
from exoplanet_pipeline.ingestion import preprocess_tess_data
from exoplanet_pipeline.denoise import decompose_and_filter

print("=== LOADING DATA ===")
lc, lc_collection, raw_flux_err = preprocess_tess_data("TIC 456945304")
filtered_lc, _, _ = decompose_and_filter(lc, raw_flux_err)

time_val = filtered_lc.time.value
flux_val = filtered_lc.flux.value
valid = ~np.isnan(flux_val)
time_val = time_val[valid]
flux_val = flux_val[valid]

print(f"\nTotal cadences: {len(time_val)}")
print(f"Time range: {time_val[0]:.2f} to {time_val[-1]:.2f} ({time_val[-1]-time_val[0]:.1f} days)")

# Check gaps
dt = np.diff(time_val)
print(f"\nCadence time-step statistics:")
print(f"  Median dt: {np.median(dt):.6f} days ({np.median(dt)*24*60:.1f} min)")
print(f"  Max dt:    {np.max(dt):.4f} days")

# All gaps > various thresholds
for thresh in [0.1, 0.2, 0.3, 0.5, 1.0]:
    n = np.sum(dt > thresh)
    print(f"  Gaps > {thresh}d: {n}")

# List all gaps > 0.1 days
gap_mask = dt > 0.1
gap_indices = np.where(gap_mask)[0]
print(f"\nAll gaps > 0.1 days:")
for gi in gap_indices:
    gap_size = dt[gi]
    t_before = time_val[gi]
    t_after = time_val[gi + 1]
    print(f"  Gap at t={t_before:.2f}-{t_after:.2f} ({gap_size:.3f} days = {gap_size*24:.1f} hrs)")

# Now simulate the perigee veto logic with the 15.05d candidate
print("\n=== SIMULATING VETO FOR P=15.0577d ===")
P_test = 15.0577
# Need to find best t0 for this period
lc_bls = filtered_lc
if len(filtered_lc) > 100000:
    step = max(1, len(filtered_lc) // 50000)
    lc_bls = filtered_lc[::step]

narrow_p = np.linspace(P_test * 0.95, P_test * 1.05, 2000)
pg = lc_bls.to_periodogram(method='bls', period=narrow_p, frequency_factor=5000)
best_p = pg.period_at_max_power.value
best_t0 = pg.transit_time_at_max_power.value
best_dur = pg.duration_at_max_power.value
print(f"Best P={best_p:.4f}d, t0={best_t0:.4f}, dur={best_dur:.4f}d")

E_all = np.round((time_val - best_t0) / best_p)
in_tr = np.abs(time_val - (best_t0 + E_all * best_p)) < (best_dur / 2)
transit_epochs = np.unique(E_all[in_tr])
print(f"Transit epochs: {transit_epochs}")

# Check each epoch against gap edges
GAP_THRESHOLD_DAYS = 0.5
EDGE_WINDOW_DAYS = 0.5
gap_mask2 = dt > GAP_THRESHOLD_DAYS
gap_indices2 = np.where(gap_mask2)[0]
gap_edges = []
for gi in gap_indices2:
    gap_edges.append(time_val[gi])
    gap_edges.append(time_val[gi + 1])
gap_edges = np.array(gap_edges) if len(gap_edges) else np.array([])
print(f"\nGap edges (threshold={GAP_THRESHOLD_DAYS}d): {len(gap_edges)} edges from {len(gap_indices2)} gaps")
for ge in gap_edges:
    print(f"  Edge at t={ge:.2f}")

n_near_edge = 0
for ep in transit_epochs:
    t_mid = best_t0 + ep * best_p
    if len(gap_edges) > 0:
        min_dist = np.min(np.abs(gap_edges - t_mid))
    else:
        min_dist = 999
    near = min_dist < EDGE_WINDOW_DAYS
    if near:
        n_near_edge += 1
    print(f"  Epoch {ep:.0f}: t_mid={t_mid:.2f}, nearest gap edge dist={min_dist:.3f}d {'*** NEAR ***' if near else ''}")

edge_frac = n_near_edge / len(transit_epochs) if len(transit_epochs) > 0 else 0
print(f"\nResult: {n_near_edge}/{len(transit_epochs)} near edge = {edge_frac:.0%} (threshold: >50%)")
