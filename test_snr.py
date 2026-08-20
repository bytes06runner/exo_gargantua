import lightkurve as lk
from exoplanet_pipeline.pipeline import run_full_pipeline

for target in ["TIC 100100827", "TIC 456945304", "TIC 36724087"]:
    res = run_full_pipeline(target, run_mcmc=False, run_centroid=False, run_fap=False)
    if res and 'bls_results' in res:
        print(f"{target} period: {res['bls_results']['period']}")
    else:
        print(f"{target} failed")
