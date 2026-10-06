"""Cost/time estimate for the v2 calibration data (decision V2-b): ~2,000 fresh training-host injections, each
searched with GPU BLS and with TLS and resolved from both, scaled from the measured B1(ii) runs (no truth read).
Output: results/v2_cost_estimate.json."""

import glob
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from b1_run import read_records  # noqa: E402

N_NEW = 2000
CPU_SESSIONS = 5
SESSION_LIMIT_H = 12.0
SAFETY = 0.75  # plan shards to use <= 75 % of the session limit


def main():
    tls = [r["wall_s"] for p in glob.glob(str(ROOT / "results/kaggle/b1/*/b1/b1_search_tls_*.jsonl.gz")) for r in read_records(p)]
    bls_done = [json.loads(Path(f).read_text()) for f in glob.glob(str(ROOT / "results/kaggle/b1/*/b1/done_search_bls_gpu_*.json"))]
    tls_core_h = float(np.sum(tls) / 3600) * N_NEW / len(tls)             # one TLS thread per injection
    shard_core_h = SESSION_LIMIT_H * SAFETY * 4                            # 4 workers per session
    n_shards = math.ceil(tls_core_h / shard_core_h)
    waves = math.ceil(n_shards / CPU_SESSIONS)
    gpu_h = max(d["wall_s"] for d in bls_done) / 3600 * N_NEW / sum(d["n"] for d in bls_done)  # two devices in parallel
    out = {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "basis": {"tls_runs": len(tls), "tls_wall_s_median": float(np.median(tls)), "tls_wall_s_p99": float(np.percentile(tls, 99)),
                     "bls_sessions": len(bls_done), "bls_session_wall_h": [round(d["wall_s"] / 3600, 2) for d in bls_done]},
           "n_injections": N_NEW,
           "tls": {"core_hours": tls_core_h, "shards": n_shards, "planned_h_per_shard": tls_core_h / n_shards / 4,
                   "waves_of_5_sessions": waves, "wall_h": waves * tls_core_h / n_shards / 4},
           "gpu_bls": {"gpu_quota_h": gpu_h, "wall_h": gpu_h},
           "note": "TLS and GPU BLS run in parallel; total wall ~ max of the two. CPU sessions have no quota; GPU quota 30 h/week."}
    (ROOT / "results" / "v2_cost_estimate.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
