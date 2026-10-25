import lightkurve as lk
from exoplanet_pipeline.pipeline import run_full_pipeline

# Test the worst offenders from the last benchmark
test_targets = [
    "TIC 456945304",   # catalog P=2.14d, was locking 15.05d
    "TIC 437704321",   # catalog P=2.85d, was locking 12.97d
    "TIC 36724087",    # catalog P=0.77d, was locking 13.77d
    "TIC 118327550",   # catalog P=7.40d, was locking 12.56d
    "TIC 425561347",   # catalog P=5.78d, was locking 15.06d
]

for target in test_targets:
    res = run_full_pipeline(target, run_mcmc=False, run_centroid=False, run_fap=False)
    if res and 'bls_results' in res:
        print(f"{target} period: {res['bls_results']['period']}")
    else:
        print(f"{target} failed")
