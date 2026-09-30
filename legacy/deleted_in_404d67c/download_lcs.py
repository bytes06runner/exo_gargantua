import json
import lightkurve as lk
import os

with open('benchmark_targets.json', 'r') as f:
    targets = json.load(f)

for target in targets:
    tic = target['target_id']
    print(f"Downloading {tic}...")
    try:
        res = lk.search_lightcurve(tic, mission='TESS', author='SPOC')
        if res:
            res.download_all(download_dir='./tess_cache')
    except Exception as e:
        print(f"Failed {tic}: {e}")
