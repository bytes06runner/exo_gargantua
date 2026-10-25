import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
import os
import scipy.optimize as opt
from exoplanet_pipeline.ingestion import preprocess_tess_data
from exoplanet_pipeline.denoise import decompose_and_filter
from exoplanet_pipeline.detection import run_bls_search
from exoplanet_pipeline.estimation import run_mcmc_estimation, _batman_model
from astropy.timeseries import BoxLeastSquares

os.makedirs('figures', exist_ok=True)

# -------------------------------------------------------------------------
# Figure 1: The Thermal Hook Fix (TIC 36724087)
# -------------------------------------------------------------------------
print("Generating Figure 1: Thermal Hook Fix (TIC 36724087)")
lc, _, _ = preprocess_tess_data("TIC 36724087")
t = lc.time.value
f = lc.flux.value

# Grab the first segment (until a gap > 0.5d)
dt = np.diff(t)
gap_indices = np.where(dt > 0.5)[0] + 1
seg_idx = np.split(np.arange(len(t)), gap_indices)[0]

t_seg = t[seg_idx]
f_seg = f[seg_idx]

# Only keep first 4 days of that segment
first_4_days = (t_seg - t_seg[0]) <= 4.0
t_seg = t_seg[first_4_days]
f_seg = f_seg[first_4_days]

def thermal_model(t, A, tau, c):
    return A * np.exp(-(t - t[0]) / max(tau, 0.001)) + c

med = np.nanmedian(f_seg)
std = np.nanstd(f_seg)
valid = (f_seg - med) < (3.0 * std)

A_guess = np.nanpercentile(f_seg[valid][:20], 95) - med
if A_guess < 0: A_guess = 0.001

popt, _ = opt.curve_fit(
    thermal_model, t_seg[valid], f_seg[valid],
    p0=[A_guess, 0.5, med],
    bounds=([0, 0.01, -np.inf], [np.inf, 3.0, np.inf]),
    maxfev=2000
)

model_fit = thermal_model(t_seg, *popt)
residual = f_seg / (model_fit / np.nanmedian(model_fit))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax1.scatter(t_seg - t_seg[0], f_seg, color='black', s=2, alpha=0.5, label='Raw Data')
ax1.plot(t_seg - t_seg[0], model_fit, color='red', lw=2, label=r'Exponential Fit ($A e^{-t/\tau} + c$)')
ax1.set_ylabel("Relative Flux")
ax1.legend()
ax1.set_title("Figure 1: Convexity Bias Fix (TIC 36724087)")

ax2.scatter(t_seg - t_seg[0], residual, color='black', s=2, alpha=0.5, label='Residual')
ax2.axhline(1.0, color='red', lw=2, linestyle='--')
ax2.set_xlabel("Time since segment start (days)")
ax2.set_ylabel("Normalized Flux")
ax2.legend()
plt.tight_layout()
plt.savefig('figures/fig1_thermal_hook.png', dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Figure 2: The Alias Trap (TIC 142937186)
# -------------------------------------------------------------------------
print("Generating Figure 2: The Alias Trap (TIC 142937186)")
lc2, _, _ = preprocess_tess_data("TIC 142937186")
lc2_filtered, _, _ = decompose_and_filter(lc2)

# The pipeline Bidirectional Validator found 17.6878d to be the true fundamental.
best_p = 17.6878

multipliers = [1/3, 1/2, 1.0, 2.0, 3.0]
depths = []
for m in multipliers:
    trial_p = best_p * m
    narrow_p = np.linspace(trial_p * 0.95, trial_p * 1.05, 1000)
    pg_sub = lc2_filtered.to_periodogram(method='bls', period=narrow_p)
    depths.append(pg_sub.depth_at_max_power.value * 100) # percent

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar([str(round(m, 2)) + "x" for m in multipliers], depths, color=['skyblue', 'skyblue', 'salmon', 'skyblue', 'skyblue'])
ax.axhline(y=max(depths)*0.95, color='red', linestyle='--', label='>95% Depth Plateau')
ax.set_xlabel("Harmonic Multiplier")
ax.set_ylabel("Transit Depth (%)")
ax.set_title("Figure 2: The Alias Trap (Bidirectional Validator) - TIC 142937186")
ax.legend()
plt.tight_layout()
plt.savefig('figures/fig2_alias_trap.png', dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Figure 3: The Imposter (Eclipsing Binary Correction) (TIC 336267424)
# -------------------------------------------------------------------------
print("Generating Figure 3: The Imposter (TIC 336267424)")
lc3, _, _ = preprocess_tess_data("TIC 336267424")
lc3_filtered, _, _ = decompose_and_filter(lc3)
wrong_period = 1.48 # Catalog's wrong period
model3 = BoxLeastSquares(lc3_filtered.time.value, lc3_filtered.flux.value)
res3 = model3.power([wrong_period], [0.1])
t0_3 = res3.transit_time[0]

time_val = lc3_filtered.time.value
flux_val = lc3_filtered.flux.value

E = np.round((time_val - t0_3) / wrong_period)
phase = ((time_val - t0_3 + 0.5*wrong_period) % wrong_period) - 0.5*wrong_period

odd_mask = (E % 2 != 0)
even_mask = (E % 2 == 0)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(phase[odd_mask], flux_val[odd_mask], s=2, color='blue', alpha=0.5, label='Odd Transits')
ax.scatter(phase[even_mask], flux_val[even_mask], s=2, color='red', alpha=0.5, label='Even Transits')
ax.set_xlim(-0.2, 0.2)
ax.set_xlabel("Phase (days) folded at P=1.48d")
ax.set_ylabel("Normalized Flux")
ax.set_title("Figure 3: Eclipsing Binary Odd/Even Depth Discrepancy (TIC 336267424)")
ax.legend()
plt.tight_layout()
plt.savefig('figures/fig3_the_imposter.png', dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Figure 4: The Grazing Planet (TIC 425561347)
# -------------------------------------------------------------------------
print("Generating Figure 4: The Grazing Planet (TIC 425561347)")
lc4, _, _ = preprocess_tess_data("TIC 425561347")
lc4_filtered, _, _ = decompose_and_filter(lc4)
results4 = run_bls_search(lc4_filtered)
best_p4 = results4['period'].value
best_t0_4 = results4['t0'].value
best_dur4 = results4['duration'].value

posteriors, sampler = run_mcmc_estimation(lc4_filtered, results4)
best_rp_rs = np.mean(posteriors['rp_rs'])
best_a_rs = np.mean(posteriors['a_rs'])
best_inc = np.mean(posteriors['inc'])
best_period = np.mean(posteriors['period'])
best_t0 = np.mean(posteriors['t0'])

# Fold data
time_val = lc4_filtered.time.value
phase = ((time_val - best_t0 + 0.5 * best_period) % best_period) - 0.5 * best_period
sort_idx = np.argsort(phase)

model_flux = _batman_model(time_val, best_period, best_t0, best_rp_rs, best_a_rs, best_inc, [0.3, 0.1])
model_phase = ((time_val - best_t0 + 0.5 * best_period) % best_period) - 0.5 * best_period

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(phase, lc4_filtered.flux.value, color='gray', s=5, alpha=0.5, label='Folded Data')
ax.plot(model_phase[sort_idx], model_flux[sort_idx], color='red', lw=2, label=f"MCMC Fit (V-shaped, b={best_a_rs * np.cos(np.radians(best_inc)):.3f})")
ax.set_xlim(-0.2, 0.2)
ax.set_xlabel("Phase (days)")
ax.set_ylabel("Normalized Flux")
ax.set_title("Figure 4: The Grazing Planet MCMC Fit (TIC 425561347)")
ax.legend()
plt.tight_layout()
plt.savefig('figures/fig4_grazing_planet.png', dpi=300)
plt.close()

print("All figures generated in 'figures/' directory.")
