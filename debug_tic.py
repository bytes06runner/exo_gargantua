import json
import numpy as np
import lightkurve as lk
from exoplanet_pipeline.pipeline import run_full_pipeline
from exoplanet_pipeline.detection import run_bls_search

target = "TIC 100100827"
print(f"Running partial pipeline for {target}")

from exoplanet_pipeline.data_loader import build_stitched_lightcurve
from exoplanet_pipeline.preprocessing import preprocess_lightcurve

with open('benchmark_targets_full.json', 'r') as f:
    targets = json.load(f)
    
t_info = next(t for t in targets if t['target_id'] == target)
lc = build_stitched_lightcurve(target, t_info['spoc_sectors'])
clean_lc = preprocess_lightcurve(lc)

from exoplanet_pipeline.detection import compute_fap_bootstrap
res = run_bls_search(clean_lc)
print(f"Found period: {res['period']}")
