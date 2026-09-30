import sys
import numpy as np
import lightkurve as lk
import pandas as pd
from exoplanet_pipeline.pipeline import run_full_pipeline

for tid in ["TIC 36724087", "TIC 142937186"]:
    res = run_full_pipeline(tid, run_mcmc=False, run_centroid=False, run_fap=False, run_report=False)
    if res:
        print(f"{tid} Final BLS P: {res['bls_results']['period']}")
