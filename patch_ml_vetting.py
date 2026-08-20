import json
import numpy as np
import os
from exoplanet_pipeline.ml_vetting import MLVetter
from exoplanet_pipeline.ingestion import fetch_stellar_parameters

results = []
with open('benchmark_results.jsonl', 'r') as f:
    for line in f:
        if not line.strip(): continue
        results.append(json.loads(line))

vetter = MLVetter.load('ml_vetter.joblib')

for entry in results:
    if entry.get('pipeline_status') != 'SUCCESS':
        continue
    
    # fetch stellar params
    stellar_params = fetch_stellar_parameters(entry['target_id'])
    approx_rp = np.sqrt(max(float(entry.get('bls_depth', 0)), 1e-6)) * stellar_params['Rs'] * 109.28
    
    # tsnr is not explicitly saved, it's roughly bls_snr
    tsnr = entry.get('bls_snr', 0)
    
    features = {
        'bls_power': entry.get('bls_snr', 0),
        'depth_diff': entry.get('odd_even_diff', 0) or 0,
        'secondary_eclipse_sigma': entry.get('sec_eclipse_sigma', 0) or 0,
        'centroid_shift': entry.get('centroid_shift_px', 0) or 0,
        'snr': tsnr,
        'depth': entry.get('bls_depth', 0) or 0,
        'rp_earth': approx_rp
    }
    
    ml_pred = vetter.predict(features)
    entry['pipeline_disposition'] = ml_pred['disposition']
    entry['planet_probability'] = ml_pred['planet_probability']
    entry['rf_raw_probability'] = ml_pred.get('rf_raw_probability')
    entry['flags'] = ml_pred.get('flags', [])
    
    # physical flags that pipeline applies
    # 1. Radius veto
    # we don't have full posteriors here easily, let's use approx_rp for a basic check
    if approx_rp > 25.0:
        entry['pipeline_disposition'] = 'FALSE_POSITIVE'
        entry['planet_probability'] = 0.0
    
    # 2. Depth mismatch flag
    bls_d = entry.get('bls_depth', 0) or 0
    mcmc_d = entry.get('mcmc_depth', 0) or bls_d
    ratio = max(mcmc_d/max(bls_d, 1e-6), bls_d/max(mcmc_d, 1e-6))
    if ratio >= 1.5:
        entry['flags'].append('BLS_MCMC_DEPTH_MISMATCH')
        
    # 3. Grazing geometry flag
    a_rs = entry.get('mcmc_a_rs')
    inc = entry.get('mcmc_inc')
    rp_rs = entry.get('mcmc_rp_rs')
    if a_rs is not None and inc is not None and rp_rs is not None:
        b = a_rs * np.cos(np.radians(inc))
        if b + rp_rs >= 0.95:
            entry['flags'].append('NEAR_GRAZING_GEOMETRY')
            if entry['pipeline_disposition'] == 'CANDIDATE':
                entry['pipeline_disposition'] = 'AMBIGUOUS'
                entry['planet_probability'] = min(entry.get('planet_probability', 1.0), 0.5)

with open('benchmark_results.jsonl', 'w') as f:
    for entry in results:
        f.write(json.dumps(entry) + '\n')
print("Patched ML vetting results successfully.")
