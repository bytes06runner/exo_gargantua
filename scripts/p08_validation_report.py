"""P08 validation report (decision V1): GPU vs astropy SDE on the 50 pre-drawn stars, the removal
pattern, and the projected pool. Reads the pulled Kaggle outputs; writes results/p08_validation_report.json.
"""

from __future__ import annotations

import glob
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exogargantua import p08io  # noqa: E402
SDE_MAX = 9.0


def wilson(k, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load(engine):
    files = glob.glob(str(ROOT / f"results/kaggle/p08val/*/p08val/p08_{engine}_*.csv"))
    return pd.concat(p08io.read_rows(f) for f in files) if files else None


def main():
    pool = pd.read_csv(ROOT / "data" / "injection_pool_log.csv")
    pool = pool[pool["decision"] == "include"]
    gpu = load("gpu").merge(pool[["tic", "stratum"]], on="tic", how="left")
    rep = {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "n_validation_stars": int(len(gpu)), "p07_pass": int((gpu["p07"] == "pass").sum())}
    s = gpu[gpu["p07"] == "pass"].copy()
    s["n_window_sectors"] = s["window_sectors"].str.count(";") + 1
    s["peak_rel_to_pmax"] = s["peak_period"] / np.minimum(30.0, 0.95 * s["span_d"])
    n_pass = int((s["p08"] == "pass").sum())
    rep["gpu"] = {
        "p08_pass": n_pass, "p08_exclude": int((s["p08"] == "exclude").sum()),
        "by_p08": {k: {"n": int(len(g)), "median_sde": float(g["sde"].median()), "median_window_sectors": float(g["n_window_sectors"].median()),
                       "median_span_d": float(g["span_d"].median()), "peak_below_1d": int((g["peak_period"] < 1).sum()),
                       "peak_within_10pct_of_pmax": int((g["peak_rel_to_pmax"] > 0.9).sum()),
                       "peak_12_to_15d": int(g["peak_period"].between(12, 15).sum())}
                   for k, g in s.groupby("p08")},
        "by_stratum": {k: {"p07_excl": int((g["p07"] != "pass").sum()), "p08_excl": int((g["p08"] == "exclude").sum()),
                           "pass": int((g["p08"] == "pass").sum())} for k, g in gpu.groupby("stratum")},
    }
    # projected pool: overall and per stratum (750 stars each), Wilson 95% intervals on the pass fraction
    k, n = n_pass, len(gpu)
    lo, hi = wilson(k, n)
    rep["projected_pool"] = {"overall_pass_fraction": k / n, "overall_ci95": [lo, hi],
                             "stars": round(len(pool) * k / n), "stars_ci95": [round(len(pool) * lo), round(len(pool) * hi)],
                             "by_stratum": {}}
    for st, g in gpu.groupby("stratum"):
        kk, nn = int((g["p08"] == "pass").sum()), len(g)
        l2, h2 = wilson(kk, nn)
        rep["projected_pool"]["by_stratum"][st] = {"pass": kk, "n": nn, "stars": round(750 * kk / nn),
                                                   "stars_ci95": [round(750 * l2), round(750 * h2)]}
    rep["injection_design_implication"] = {
        "injections_at_10_per_star": rep["projected_pool"]["stars"] * 10,
        "injections_ci95": [x * 10 for x in rep["projected_pool"]["stars_ci95"]],
        "design_headline": 30000,
    }
    cpu = load("astropy")
    if cpu is not None:
        m = gpu.merge(cpu, on="tic", suffixes=("_gpu", "_cpu"))
        m = m[(m["p07_gpu"] == "pass") & m["sde_cpu"].notna()]
        d = (m["sde_gpu"] - m["sde_cpu"]).abs()
        flips = m[(m["sde_gpu"] >= SDE_MAX) != (m["sde_cpu"] >= SDE_MAX)]
        grid_step = (m["peak_period_gpu"] - m["peak_period_cpu"]).abs()
        rep["gpu_vs_astropy"] = {"stars_compared": int(len(m)), "max_abs_sde_diff": float(d.max()),
                                 "median_abs_sde_diff": float(d.median()), "flips_across_9": flips["tic"].astype(int).tolist(),
                                 "same_grid_size": bool((m["n_periods_gpu"] == m["n_periods_cpu"]).all()),
                                 "peak_period_identical": int((grid_step == 0).sum()),
                                 "max_abs_peak_period_diff_d": float(grid_step.max())}
    out = ROOT / "results" / "p08_validation_report.json"
    out.write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
