import json
import numpy as np
import pandas as pd
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from exoplanet_pipeline.ingestion import fetch_stellar_parameters

results = []
with open('real_benchmark_results.jsonl', 'r') as f:
    for line in f:
        if not line.strip(): continue
        results.append(json.loads(line))

for entry in results:
    if not entry.get('exo_rejected'):
        tid = entry['target_id']
        try:
            stellar_params = fetch_stellar_parameters(tid)
            if stellar_params is not None and stellar_params.get('Rs') is not None:
                depth = entry.get('exo_depth')
                if depth is None: # In run_120_real_benchmark, depth is not saved, but we can compute it if needed, wait...
                    pass
        except Exception as e:
            pass
