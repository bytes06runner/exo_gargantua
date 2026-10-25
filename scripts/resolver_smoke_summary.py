"""Runtime/health summary of the resolver execution test on the 20 smoke targets (sealed B2 members).
Reads ONLY status, runtime and size columns -- never the resolved periods, probabilities or any truth (A4).
Output: results/resolver_smoke_runtime.json."""

import glob
import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    f = glob.glob(str(ROOT / "results/kaggle/resolver_smoke/*/resolver_smoke/resolver_smoke.csv"))[0]
    d = pd.read_csv(f, usecols=lambda c: c in ("toi", "seed", "status", "runtime_s", "n_points", "load_s", "n_hypotheses"))
    ok = d[d["status"] == "ok"]
    done = json.loads(Path(f).with_name("done.json").read_text())
    out = {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "run_commit": done["git_commit"], "hardware": "Kaggle CPU session (4 cores), one resolver process",
           "targets": int(d["toi"].nunique()), "runs": int(len(d)), "ok": int(len(ok)),
           "status_counts": d["status"].value_counts().to_dict(),
           "runtime_s": {"median": float(ok["runtime_s"].median()), "p95": float(ok["runtime_s"].quantile(0.95)),
                         "max": float(ok["runtime_s"].max())},
           "n_points": {"median": int(ok["n_points"].median()), "max": int(ok["n_points"].max())},
           "hypotheses_per_run": sorted(ok["n_hypotheses"].unique().astype(int).tolist()),
           "note": "Execution test only; resolver outputs are sealed (no truth comparison before resolver-frozen-v1)."}
    (ROOT / "results" / "resolver_smoke_runtime.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
