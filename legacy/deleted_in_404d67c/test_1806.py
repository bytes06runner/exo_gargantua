import sys
import numpy as np
import lightkurve as lk
import pandas as pd
from exoplanet_pipeline.pipeline import run_full_pipeline

res = run_full_pipeline("TIC 180695581", run_mcmc=False, run_centroid=False, run_fap=False, run_report=False)
if res:
    print(f"Final BLS P: {res['bls_results']['period']}")
