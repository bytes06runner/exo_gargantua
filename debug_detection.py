import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from exoplanet_pipeline.ingestion import preprocess_tess_data
from exoplanet_pipeline.denoise import decompose_and_filter
from exoplanet_pipeline.detection import run_bls_search

target = "TIC 180695581"
print(f"Testing {target}...")
lc, coll, err = preprocess_tess_data(target)
filtered_lc, res, stats = decompose_and_filter(lc, raw_flux_err=err)

# Run same BLS setup
lc_bls = filtered_lc
if len(filtered_lc) > 100000:
    step = max(1, len(filtered_lc) // 50000)
    lc_bls = filtered_lc[::step]

t_baseline = lc_bls.time.value[-1] - lc_bls.time.value[0]
df = 1.0 / (3.0 * t_baseline)
freq_min = 1.0 / 20.0
freq_max = 1.0 / 0.5
if (freq_max - freq_min) / df > 200000:
    df = (freq_max - freq_min) / 200000.0
freqs = np.arange(freq_min, freq_max, df)
periods = 1.0 / freqs

periodogram = lc_bls.to_periodogram(
    method='bls', 
    period=periods,
    frequency_factor=5000
)

power_s = pd.Series(periodogram.power.value)
roll_med = power_s.rolling(window=501, min_periods=1, center=True).median()
roll_std = power_s.rolling(window=501, min_periods=1, center=True).std()
roll_std_mean = np.nanmean(roll_std)
if roll_std_mean == 0 or np.isnan(roll_std_mean):
    roll_std_mean = 1.0
roll_std[roll_std == 0] = roll_std_mean

sde = (power_s - roll_med) / roll_std
sde = sde.fillna(0).values

peaks, _ = find_peaks(sde)
if len(peaks) == 0:
    peaks = np.argsort(sde)[-50:]
    
sorted_peaks = peaks[np.argsort(sde[peaks])][::-1]

print("Top 10 peaks by SDE:")
time_val = lc_bls.time.value
flux_val = lc_bls.flux.value
valid = ~np.isnan(flux_val)
time_val = time_val[valid]

for i, pk in enumerate(sorted_peaks[:10]):
    P = periodogram.period.value[pk]
    t0 = periodogram.transit_time.value[pk]
    dur = periodogram.duration.value[pk]
    power = periodogram.power.value[pk]
    sde_val = sde[pk]
    
    E = np.round((time_val - t0) / P)
    in_transit = np.abs(time_val - (t0 + E * P)) < (dur / 2)
    unique_epochs = len(np.unique(E[in_transit]))
    
    print(f"  {i+1}: P={P:.5f} d, t0={t0:.4f}, dur={dur:.4f}, raw_pow={power:.2f}, sde={sde_val:.2f}, epochs={unique_epochs}")

