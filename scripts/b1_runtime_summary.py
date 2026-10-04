"""Per-run resolver time in B1(i) (no truth read). Output: results/b1_runtime.json."""

import glob
import gzip
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    wall, rt, npts, err = [], [], [], 0
    for p in glob.glob(str(ROOT / "results/kaggle/b1/*/b1/b1_wrong_*.jsonl.gz")):
        with gzip.open(p, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                if "error" in r:
                    err += 1
                    continue
                wall.append(r["wall_s"]); rt.append(r["summary"]["runtime_s"]); npts.append(r["summary"]["n_points"])
    done = [json.loads(Path(f).read_text()) for f in glob.glob(str(ROOT / "results/kaggle/b1/*/b1/done_wrong_*.json"))]
    q = lambda a: {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(np.max(a))}  # noqa: E731
    out = {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "run_commits": sorted({d["git_commit"] for d in done}),
           "hardware": "Kaggle CPU session, 4 cores; 4 worker processes, 1 BLAS thread each",
           "runs": len(wall) + err, "errors": err, "per_run_wall_s": q(wall), "resolver_runtime_s": q(rt), "n_points": q(npts),
           "shard_wall_h": sorted(round(d["wall_s"] / 3600, 2) for d in done)}
    (ROOT / "results" / "b1_runtime.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
