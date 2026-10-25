"""B4 runtime summary (docs/b4_definition.md): per-light-curve times from the b4 stage outputs, joined by TIC; TLS
per-light-curve time quoted from the measured B2 TLS seed runs (results/kaggle/b2/*/b2/seeds_tls_*.csv, tls_s,
Kaggle CPU session, TLS use_threads = all 4 cores). Output: results/b4_runtime.json."""

import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from b1_run import read_records  # noqa: E402


def q(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return {"n": int(a.size), "median_s": float(np.median(a)), "p95_s": float(np.percentile(a, 95)), "max_s": float(a.max())} if a.size else {"n": 0}


def main():
    base = ROOT / "results/kaggle/b4"
    bls = pd.DataFrame([r for p in glob.glob(str(base / "*/b4/b4_bls_*.jsonl.gz")) for r in read_records(p)])
    res = pd.DataFrame([r for p in glob.glob(str(base / "*/b4/b4_resolver_*.jsonl.gz")) for r in read_records(p)])
    done = [json.loads(Path(f).read_text()) for f in glob.glob(str(base / "*/b4/done_*.json"))]
    m = bls.merge(res, on="tic", how="outer", suffixes=("_gpu", "_cpu"))
    ok = m["bls_s"].notna() & m["resolver_s"].notna() & m.get("error_gpu", pd.Series(np.nan, index=m.index)).isna() & m.get("error_cpu", pd.Series(np.nan, index=m.index)).isna()
    tot = m.loc[ok, "load_s_gpu"] + m.loc[ok, "prepare_s"] + m.loc[ok, "bls_s"] + m.loc[ok, "resolver_s"]
    tls = pd.concat(pd.read_csv(f) for f in glob.glob(str(ROOT / "results/kaggle/b2/*/b2/seeds_tls_*.csv")))
    rep = {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "sample": "TESS Sector 49, all SPOC 120-s light curves in the frozen listing", "pinned": 19999,
           "light_curves_with_bls": int(m["bls_s"].notna().sum()), "light_curves_with_resolver": int(m["resolver_s"].notna().sum()),
           "complete_pipeline": int(ok.sum()),
           "failures": {"bls_stage": int(m.get("error_gpu", pd.Series(dtype=object)).notna().sum()),
                        "resolver_stage": int(m.get("error_cpu", pd.Series(dtype=object)).notna().sum())},
           "per_light_curve": {"load": q(m["load_s_gpu"]), "prepare": q(m["prepare_s"]), "gpu_bls": q(m["bls_s"]),
                               "resolver": q(m["resolver_s"]), "total": q(tot)},
           "hardware": {d["stage"]: d["hardware"] for d in done}, "workers": {d["stage"]: d["workers"] for d in done},
           "stage_wall_h": {f"{d['stage']}_{i}": round(d["wall_s"] / 3600, 2) for i, d in enumerate(done)},
           "tls_quoted_from_B2": {**q(tls["tls_s"]), "hardware": "Kaggle CPU session, 4 cores, TLS use_threads=4",
                                  "note": "early-sector B2 light curves (1-13 sectors), not Sector 49 single-sector light curves"}}
    (ROOT / "results" / "b4_runtime.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep["per_light_curve"], indent=1))


if __name__ == "__main__":
    main()
