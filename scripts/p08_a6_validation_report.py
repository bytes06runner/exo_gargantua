"""P08 validation under amendment A6 (and the raw rule, V1) from the A6 re-run of the 50 validation stars
(kaggle/jobs/p08_a6val_*). Per rule: GPU-vs-astropy flips across SDE = 9, max |dSDE|, peak-period agreement;
per engine: the pool projection by sector-count stratum (Wilson 95%). Diagnostics (cell-minimum, dense grid) are
summarised but decide nothing. Output: results/p08_a6_validation_report.json.
"""

from __future__ import annotations

import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exogargantua import p08io  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
from p08_validation_report import wilson  # noqa: E402

SDE_MAX = 9.0
STRATUM_SIZE = 750
RULES = {"raw_V1": ("sde", "peak_period", "p08"), "A6": ("sde_a6", "peak_period_a6", "p08_a6")}


def load(engine):
    files = glob.glob(str(ROOT / f"results/kaggle/p08a6val/*/p08a6val/p08_{engine}_*.csv"))
    return pd.concat(p08io.read_rows(f) for f in files) if files else None


def projection(df, col):
    out = {}
    k, n = int((df[col] == "pass").sum()), len(df)  # P07 failures count as not in the pool
    lo, hi = wilson(k, n)
    out["overall"] = {"pass": k, "n": n, "stars": round(4 * STRATUM_SIZE * k / n),
                      "stars_ci95": [round(4 * STRATUM_SIZE * lo), round(4 * STRATUM_SIZE * hi)]}
    out["by_stratum"] = {}
    for st, g in df.groupby("stratum"):
        kk, nn = int((g[col] == "pass").sum()), len(g)
        l2, h2 = wilson(kk, nn)
        out["by_stratum"][st] = {"pass": kk, "n": nn, "stars": round(STRATUM_SIZE * kk / nn),
                                 "stars_ci95": [round(STRATUM_SIZE * l2), round(STRATUM_SIZE * h2)]}
    out["injections_at_10_per_star"] = out["overall"]["stars"] * 10
    out["injections_ci95"] = [x * 10 for x in out["overall"]["stars_ci95"]]
    return out


def main():
    pool = pd.read_csv(ROOT / "data" / "injection_pool_log.csv")
    pool = pool[pool["decision"] == "include"][["tic", "stratum"]]
    gpu, cpu = load("gpu"), load("astropy")
    rep = {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "runs_commits": sorted({json.loads(Path(f).read_text())["git_commit"]
                                   for f in glob.glob(str(ROOT / "results/kaggle/p08a6val/*/p08a6val/done_*.json"))}),
           "engines_present": {"gpu": gpu is not None, "astropy": cpu is not None}}
    for name, df in (("gpu", gpu), ("astropy", cpu)):
        if df is None:
            continue
        df = df.merge(pool, on="tic", how="left")
        rep[name] = {"n_stars": int(len(df)), "p07_pass": int((df["p07"] == "pass").sum()),
                     "projection": {r: projection(df.assign(**{c: df[c].fillna("exclude")}), c) for r, (_, _, c) in RULES.items()}}
        s = df[df["p07"] == "pass"]
        rep[name]["diagnostics_excluded_at_9"] = {c: int((s[c] >= SDE_MAX).sum()) for c in
                                                  ("sde", "sde_a6", "sde_a6_cellmax", "sde_dense_tlspts", "sde_dense_scaled") if c in s}
        rep[name]["a6_vs_raw_decision_changes"] = {"raw_exclude_to_a6_pass": s.loc[(s["p08"] == "exclude") & (s["p08_a6"] == "pass"), "tic"].astype(int).tolist(),
                                                   "raw_pass_to_a6_exclude": s.loc[(s["p08"] == "pass") & (s["p08_a6"] == "exclude"), "tic"].astype(int).tolist()}
    if gpu is not None and cpu is not None:
        m = gpu.merge(cpu, on="tic", suffixes=("_gpu", "_cpu"))
        m = m[(m["p07_gpu"] == "pass") & (m["p07_cpu"] == "pass")]
        rep["gpu_vs_astropy"] = {"stars_compared": int(len(m)), "same_bls_grid": bool((m["n_periods_gpu"] == m["n_periods_cpu"]).all()),
                                 "same_tls_grid": bool((m["n_tls_periods_gpu"] == m["n_tls_periods_cpu"]).all())}
        for r, (sc, pc, _) in RULES.items():
            d = (m[f"{sc}_gpu"] - m[f"{sc}_cpu"]).abs()
            flips = m[(m[f"{sc}_gpu"] >= SDE_MAX) != (m[f"{sc}_cpu"] >= SDE_MAX)]
            dp = (m[f"{pc}_gpu"] - m[f"{pc}_cpu"]).abs()
            rep["gpu_vs_astropy"][r] = {"flips_across_9": flips["tic"].astype(int).tolist(), "max_abs_sde_diff": float(d.max()),
                                        "median_abs_sde_diff": float(d.median()), "peak_period_identical": int((dp == 0).sum()),
                                        "max_abs_peak_period_diff_d": float(dp.max()),
                                        "min_distance_to_9": float(np.minimum((m[f"{sc}_gpu"] - SDE_MAX).abs(), (m[f"{sc}_cpu"] - SDE_MAX).abs()).min())}
    out = ROOT / "results" / "p08_a6_validation_report.json"
    out.write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
