import sys
import numpy as np
import lightkurve as lk
from pipeline import run_pipeline

res = run_pipeline("TIC 180695581", "Deep (>10000 ppm)", max_sectors=1)
if res:
    print("Done")
