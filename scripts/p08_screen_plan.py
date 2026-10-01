"""Plan the full P08 screen (A5 window, A6 + raw SDE) on Kaggle GPU (two T4 per session, one device-shard each).

Work per star follows p08_cost.py (sectors x bins x BLS periods over the A5 window, from pinned products; P07 is
applied on Kaggle and can only shorten this). Seconds per unit of work are calibrated on the 45 GPU A6-validation
stars (search_s + a6_s). Output: results/p08_screen_plan.json with the number of sessions and the estimated GPU
quota hours per session (= session wall time), used by the orchestrator's quota guard.
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
sys.path.insert(0, str(ROOT / "scripts"))
from exogargantua import p08io, sample as S  # noqa: E402
import estimate_costs as E1  # noqa: E402

TARGET_SESSION_H = 3.5   # per session; keeps each well inside Kaggle's 12 h limit and lets the quota guard pace them
SETUP_H = 0.1            # clone + pip install + cache mount, measured ~5 min on the validation runs
BINS_PER_SECTOR = 3275.0  # as scripts/p08_screen.py predicted_cost


def work(win, starts, nper_cache):
    sp = E1.window_span(win, starts)
    key = round(sp, 3)
    if key not in nper_cache:
        nper_cache[key] = E1.n_bls_periods(sp)
    return len(win) * BINS_PER_SECTOR * nper_cache[key]


def main():
    starts = S.sector_start_btjd(pd.read_csv(sorted((ROOT / "data" / "raw").glob("tess_orbit_times_*.csv"))[0]))
    nper = {}
    val = pd.concat(p08io.read_rows(f) for f in glob.glob(str(ROOT / "results/kaggle/p08a6val/*gpu*/p08a6val/p08_gpu_*.csv")))
    val = val[val["p07"] == "pass"]
    vw = np.array([work([int(x) for x in str(w).split(";")], starts, nper) for w in val["window_sectors"]])
    sec_per_work = float((val["search_s"] + val["a6_s"]).sum() / vw.sum())
    pool = pd.read_csv(ROOT / "data" / "injection_pool_log.csv")
    pool = pool[pool["decision"] == "include"]
    pins = pd.read_csv(ROOT / "data" / "pinned_products.csv", usecols=["tic", "sector", "role"])
    secs = pins[pins["role"] == "pool"].groupby("tic")["sector"].apply(sorted)
    w = []
    for tic in pool["tic"]:
        s = secs[tic]
        w.append(work([x for x in s if starts[x] < starts[s[0]] + 365.25][:10], starts, nper))
    w = np.array(w)
    device_h_total = w.sum() * sec_per_work / 3600
    n_sessions = math.ceil(device_h_total / 2 / (TARGET_SESSION_H - SETUP_H))
    # p08_screen.py balances greedily over 2 * n_sessions device-shards; the largest shard sets the session time
    load = np.zeros(2 * n_sessions)
    for c in sorted(w, reverse=True):
        load[int(np.argmin(load))] += c
    shard_h = load * sec_per_work / 3600
    sess_h = [float(max(shard_h[2 * k], shard_h[2 * k + 1]) + SETUP_H) for k in range(n_sessions)]
    out = {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "stars": int(len(w)), "calibration_stars": int(len(val)), "sec_per_work_unit": sec_per_work,
           "device_h_total": float(device_h_total), "n_sessions": n_sessions, "nshards": 2 * n_sessions,
           "est_session_h": sess_h, "est_quota_h_total": float(sum(sess_h)),
           "note": "P07 (applied on Kaggle) removes ~10% of stars in the validation sample, so these are upper estimates."}
    (ROOT / "results" / "p08_screen_plan.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
