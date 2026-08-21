import json
import lightkurve as lk
import os
import socket
import glob
import time

socket.setdefaulttimeout(30.0)

with open('benchmark_targets.json', 'r') as f:
    targets = json.load(f)

for target in targets:
    tic = target['target_id']
    
    # Check if already downloaded
    existing = glob.glob(f"./tess_cache/**/*{tic}*lc.fits", recursive=True)
    if existing:
        print(f"{tic} already downloaded ({len(existing)} files).")
        continue
        
    print(f"Downloading {tic}...")
    success = False
    for attempt in range(5):
        try:
            res = lk.search_lightcurve(tic, mission='TESS', author='SPOC')
            if res:
                res.download_all(download_dir='./tess_cache')
            success = True
            break
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(5)
    if not success:
        print(f"FAILED completely to download {tic}.")
