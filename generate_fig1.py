import matplotlib.pyplot as plt
import numpy as np
import batman
from exoplanet_pipeline.ingestion import preprocess_tess_data
from exoplanet_pipeline.denoise import decompose_and_filter

# 1. Get the real detrended light curve
print("Ingesting and preprocessing TESS data for TIC 281408474...")
stitched_lc, _, raw_flux_err = preprocess_tess_data("TIC 281408474")
print("Denoising / detrending light curve...")
filtered_lc, _, _ = decompose_and_filter(stitched_lc, raw_flux_err)

# 2. Phase-fold using the MCMC median posteriors from the JSON
period = 3.4094465897465627
t0 = 3668.3924370493487
phase = ((filtered_lc.time.value - t0 + period/2) % period) / period - 0.5

# 3. Build the fitted batman model using the same MCMC medians
print("Building batman model...")
params = batman.TransitParams()
params.t0, params.per = 0.0, period
params.rp = 0.07553868962080082          # rp_rs median
params.a  = 7.148967123031735            # a_rs median
params.inc = 87.27295508397438
params.ecc, params.w = 0.0, 90.0
params.u, params.limb_dark = [0.3, 0.1], "quadratic"

phase_model = np.linspace(-0.1, 0.1, 500)
m = batman.TransitModel(params, phase_model * period)
model_flux = m.light_curve(params)

# 4. Single clean panel — RNAAS-ready
print("Plotting and saving figure...")
fig, ax = plt.subplots(figsize=(7, 5))
mask = np.abs(phase) < 0.1
ax.plot(phase[mask], filtered_lc.flux.value[mask], '.', ms=1.5, color='gray', alpha=0.4, label='TIC 281408474 (raw cadences)')
ax.plot(phase_model, model_flux, color='crimson', lw=2, label='Best-fit transit model')
ax.set_xlabel("Orbital Phase")
ax.set_ylabel("Normalized Flux")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("fig1_tic281408474.png", dpi=300)
print("Saved fig1_tic281408474.png successfully!")
