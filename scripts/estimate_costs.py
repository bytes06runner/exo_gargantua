"""Gate 2 cost estimate, computed only from committed files (no typed-in numbers).

Inputs:
  --smoke  smoke_results.json from the Kaggle smoke run (measured costs, 4-core Kaggle CPU)
  --quota  results/kaggle/quota_<date>.json (saved `kaggle quota` output)
  data/sample_log.csv, data/pinned_products.csv, data/stellar_params.csv, data/raw/ orbit times
Output: results/cost_estimate.json

Model (checkable):
  * BLS and TLS cost per search = c * N_points * N_trial_periods, with one constant per method
    fitted (median ratio) on the smoke targets, where both N's were recorded.
  * For each target we reproduce the exact grid sizes the pre-registered searches would use:
    astropy `BoxLeastSquares.autoperiod` (same durations, minimum_n_transit=2,
    frequency_factor=1) and `transitleastsquares.period_grid` (TIC R*, M*), on the target's real
    early-window span; N_points = early sectors x median 10-min bins per sector (smoke).
  * Storage and download scale with the number of pinned files (mean MB per file, smoke).
  * B1: the injection grid is not fixed yet; costed per injection on the pool stars' own
    sector sets, using the first k pinned sectors with k uniform in 1..10.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exogargantua import sample as S  # noqa: E402

BLS_DURATIONS = np.array([0.04, 0.06, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30])
SECTOR_DAYS = 27.4


def n_bls_periods(span):
    from astropy.timeseries import BoxLeastSquares
    t = np.linspace(0, span, 200)
    pmax = min(30.0, 0.95 * span)
    return len(BoxLeastSquares(t, np.ones_like(t)).autoperiod(
        BLS_DURATIONS, minimum_period=0.5, maximum_period=pmax, minimum_n_transit=2, frequency_factor=1.0))


def n_tls_periods(span, r, m):
    from transitleastsquares import period_grid
    r = r if np.isfinite(r) else 1.0
    m = m if np.isfinite(m) else 1.0
    return len(period_grid(R_star=r, M_star=m, time_span=span, period_min=0.5,
                           period_max=min(30.0, 0.95 * span), oversampling_factor=3))


def window_span(sectors, starts):
    return starts[max(sectors)] + SECTOR_DAYS - starts[min(sectors)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", required=True)
    ap.add_argument("--quota", required=True)
    ap.add_argument("--b1-injections", type=int, default=30000, help="proposal, not fixed")
    args = ap.parse_args()

    smoke = json.loads(Path(args.smoke).read_text())
    rows = [r for r in smoke["targets"] if "search" in r]
    cores = smoke["env"]["cpu_count"]
    # per-(point x period) constants
    c_bls = float(np.median([r["search"]["bls_s"] / (r["search"]["n_binned"] * r["search"]["bls_n_periods"]) for r in rows]))
    c_tls = float(np.median([r["search"]["tls_s"] / (r["search"]["n_binned"] * r["search"]["tls_n_periods"]) for r in rows]))
    bins_per_sector = float(np.median([r["search"]["n_binned"] / len(str(r["sectors_early"]).split(";")) for r in rows]))
    files = [f for r in smoke["targets"] for f in r["files"] if "raw_bytes" in f]
    raw_mb = float(np.mean([f["raw_bytes"] for f in files])) / 1e6
    slim_mb = float(np.mean([f["slim_bytes"] for f in files if "slim_bytes" in f])) / 1e6
    dl_mb_s = sum(r["raw_bytes"] for r in smoke["targets"]) / 1e6 / max(1e-9, sum(r["download_s"] for r in smoke["targets"]))
    # model check on the smoke targets themselves (predicted vs measured, with real grid sizes)
    check = [{"toi": r["toi"], "measured_s": r["search"]["bls_s"] + r["search"]["tls_s"],
              "model_s": r["search"]["n_binned"] * (c_bls * r["search"]["bls_n_periods"] + c_tls * r["search"]["tls_n_periods"])}
             for r in rows]

    starts = S.sector_start_btjd(pd.read_csv(sorted((ROOT / "data" / "raw").glob("tess_orbit_times_*.csv"))[0]))
    log = pd.read_csv(ROOT / "data" / "sample_log.csv", dtype={"toi": str})
    stars = pd.read_csv(ROOT / "data" / "stellar_params.csv").drop_duplicates("tic").set_index("tic")
    pins = pd.read_csv(ROOT / "data" / "pinned_products.csv", usecols=["tic", "sector", "role"])
    inc = log[log["decision"] == "include"]
    b2 = inc[inc["b2_eligible"] == "pending_H3"]

    def cost(sectors, tic):
        span = window_span(sectors, starts)
        npts = len(sectors) * bins_per_sector
        r = stars.loc[tic, "rad"] if tic in stars.index else np.nan
        m = stars.loc[tic, "mass"] if tic in stars.index else np.nan
        return npts * c_bls * n_bls_periods(span), npts * c_tls * n_tls_periods(span, r, m), span

    b2_rows = []
    for _, r in b2.iterrows():
        secs = [int(x) for x in str(r["sectors_early"]).split(";")]
        bls_s, tls_s, span = cost(secs, int(r["tic"]))
        b2_rows.append({"toi": r["toi"], "n_early": len(secs), "span_d": span, "bls_s": bls_s, "tls_s": tls_s})
    b2df = pd.DataFrame(b2_rows)
    bins = pd.cut(b2df["span_d"], [0, 30, 60, 120, 240, 400], labels=["<=30d", "30-60d", "60-120d", "120-240d", "240-400d"])
    by_span = b2df.groupby(bins, observed=False).agg(n=("toi", "size"), bls_h=("bls_s", lambda s: s.sum() / 3600),
                                                      tls_h=("tls_s", lambda s: s.sum() / 3600)).reset_index()

    rng = np.random.default_rng(20260930)
    pool = pins[pins["role"] == "pool"].groupby("tic")["sector"].apply(sorted)
    b1 = {"first_k": [], "one_year_window": []}
    for tic in rng.choice(pool.index.to_numpy(), size=min(400, len(pool)), replace=False):
        k = int(rng.integers(1, 11))
        secs = pool[tic]
        b1["first_k"].append(sum(cost(secs[:k], int(tic))[:2]))
        # alternative: up to k sectors that all start within 365 d of the first one used
        win = [s_ for s_ in secs if starts[s_] < starts[secs[0]] + 365.25][:k]
        bls_s, tls_s, _ = cost(win, int(tic))
        b1["one_year_window"].append((bls_s, tls_s, len(win)))
    b1_mean = float(np.mean(b1["first_k"]))
    w = np.array(b1["one_year_window"])
    b1_win = {"mean_bls_s": float(w[:, 0].mean()), "mean_tls_s": float(w[:, 1].mean()),
              "mean_sectors": float(w[:, 2].mean())}

    n_files = {"toi_host": int((pins["role"] == "toi_host").sum()), "pool": int((pins["role"] == "pool").sum())}
    tot_files = sum(n_files.values())
    quota = json.loads(Path(args.quota).read_text())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    est = {
        "git_commit": commit, "smoke_commit": smoke["git_commit"], "smoke_env": smoke["env"],
        "measured": {"targets": len(smoke["targets"]), "searched": len(rows), "mean_raw_mb_per_file": raw_mb,
                     "mean_slim_mb_per_file": slim_mb, "download_mb_per_s": dl_mb_s,
                     "c_bls_s_per_point_period": c_bls, "c_tls_s_per_point_period": c_tls,
                     "bins_per_sector": bins_per_sector, "model_check": check},
        "full_sample": {
            "pinned_files": n_files, "download_gb": tot_files * raw_mb / 1e3, "slim_cache_gb": tot_files * slim_mb / 1e3,
            "download_wall_h_one_session": tot_files * raw_mb / dl_mb_s / 3600,
            "b2_targets_pending_h3": int(len(b2df)),
            "b2_bls_wall_h": float(b2df["bls_s"].sum() / 3600), "b2_tls_wall_h": float(b2df["tls_s"].sum() / 3600),
            "b2_by_early_span": by_span.to_dict("records"),
        },
        "b1_proposal": {"injections": args.b1_injections,
                        "first_k_sectors": {"mean_search_s_per_injection": b1_mean,
                                            "wall_h_one_session": args.b1_injections * b1_mean / 3600},
                        "one_year_window": {**b1_win,
                                            "wall_h_one_session_bls_serial": args.b1_injections * (b1_win["mean_bls_s"] + b1_win["mean_tls_s"]) / 3600,
                                            "wall_h_one_session_bls_parallel": args.b1_injections * (b1_win["mean_bls_s"] / cores + b1_win["mean_tls_s"]) / 3600},
                        "note": "BLS + TLS seed searches only; resolver cost unknown until Phase 3"},
        "b2_bls_parallel_across_cores_wall_h": float(b2df["bls_s"].sum() / 3600 / cores + b2df["tls_s"].sum() / 3600),
        "cores_per_session": cores, "kaggle_quota": quota,
    }
    out = ROOT / "results" / "cost_estimate.json"
    out.write_text(json.dumps(est, indent=1, default=float))
    print(json.dumps({k: est[k] for k in ("measured", "full_sample", "b1_proposal")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
