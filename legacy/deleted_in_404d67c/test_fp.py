import sys
from exoplanet_pipeline.pipeline import run_full_pipeline

def test_fp():
    target_id = "TIC 336267424"
    print(f"Running pipeline on {target_id}...")
    res = run_full_pipeline(target_id, run_mcmc=False, run_centroid=False, run_fap=False, run_report=False)
    
    if res:
        print("\nFINAL PIPELINE RESULTS:")
        print(f"ML Disposition: {res.get('ml_vetting', {}).get('disposition', 'UNKNOWN')}")
        print(f"Planet Prob: {res.get('ml_vetting_score', 0.0)}")

if __name__ == "__main__":
    test_fp()
