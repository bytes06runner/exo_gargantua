import sys
sys.path.append('exoplanet_pipeline')
from pipeline import run_pipeline

res = run_pipeline("TIC 142937186", "Deep (>10000 ppm)", max_sectors=1)
if res:
    print(f"Final MCMC P: {res.get('mcmc_period')}")
