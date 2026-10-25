"""Re-run the benchmark-era v1 pipeline (commit ac43691) on one target, exactly as
legacy/deleted_in_404d67c/run_120_real_benchmark.py called it, and print which override fired.

Usage (from a checkout of ac43691):
    git worktree add ../legacy_ac43691 ac43691 && cd ../legacy_ac43691
    python <this repo>/scripts/legacy_audit/rerun_legacy_target.py "TIC 406941612"

Light curves are NOT pinned: lightkurve downloads whatever sectors MAST serves today, so
targets observed since August 2026 can give different results from the v1 run.
"""
import sys, time, io, contextlib, json, re
sys.path.insert(0, '.')
import warnings; warnings.simplefilter('ignore')
from exoplanet_pipeline.pipeline import run_full_pipeline
tid = sys.argv[1]
t = time.time(); buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    res = run_full_pipeline(tid, run_mcmc=False, run_centroid=False, run_fap=False, run_report=False)
log = buf.getvalue()
ov = [l.strip() for l in log.splitlines() if 'OVERRIDE' in l or 'Flag' in l or 'ML Vetting Probability' in l or 'Bidirectional' in l or 'Using' in l and 'sector' in l]
print(json.dumps({'tic': tid, 'secs': round(time.time()-t), 'period': float(res['bls_results']['period'].value) if res else None,
  'power': float(res['bls_results']['snr'].value) if res else None, 'lines': ov, 'disp': (res or {}).get('ml_vetting', {}).get('disposition')}))
open(f"/tmp/legacy_{tid.replace(' ','_')}.log",'w').write(log)
