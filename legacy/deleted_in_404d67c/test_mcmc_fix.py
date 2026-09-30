import os
import json
from exoplanet_pipeline.pipeline import run_full_pipeline

# The three problematic targets that failed previously
test_targets = ["TIC 180695581", "TIC 36724087", "TIC 142937186"]

print("Starting verification on failed targets...")
for tic in test_targets:
    print(f"\n--- Testing {tic} ---")
    try:
        res = run_full_pipeline(tic, run_mcmc=True, n_mcmc_steps=1000)
        if res is None:
            print("  Pipeline failed.")
            continue
            
        bls = res.get('bls_results', {})
        bls_period = bls.get('period', 0)
        
        post = res.get('posteriors', {})
        mcmc_period = post.get('period', [0])[0] if post else 0
        
        print(f"  BLS Period:  {getattr(bls_period, 'value', bls_period):.6f} d")
        print(f"  MCMC Period: {mcmc_period:.6f} d")
    except Exception as e:
        print(f"  Error: {e}")
