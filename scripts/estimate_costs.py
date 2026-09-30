"""Gate 2 cost estimate, computed only from committed files (no typed-in numbers).

Inputs:
  results/kaggle/smoke_run/<run_id>/smoke_results.json   measured per-target costs
  data/sample_log.csv, data/pinned_products.csv, data/injection_pool_log.csv   frozen sample
  results/kaggle/quota_<date>.json                        `kaggle quota` output saved at Gate 2
Output: results/cost_estimate.json (and a printed summary).

Scaling model (stated so it can be checked):
  * download and cache sizes scale with the number of pinned files (mean bytes per file);
  * download time scales with bytes (measured MB/s);
  * BLS and TLS CPU time per search are fitted as power laws in the number of 10-min bins,
    t = a * N^k (least squares in log-log), because both grow with points x trial periods;
  * a search on n sectors has N ~ n * (median bins per sector) points.
The B1 grid size is an input (default proposal below), not a measured quantity.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def fit_power(n, t):
    n, t = np.asarray(n, float), np.asarray(t, float)
    ok = (n > 0) & (t > 0)
    k, loga = np.polyfit(np.log(n[ok]), np.log(t[ok]), 1)
    return float(np.exp(loga)), float(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", required=True, help="path to smoke_results.json")
    ap.add_argument("--quota", required=True, help="path to saved kaggle quota json")
    ap.add_argument("--b1-injections", type=int, default=30000, help="proposed B1 grid size")
    args = ap.parse_args()

    smoke = json.loads(Path(args.smoke).read_text())
    rows = [r for r in smoke["targets"] if "search" in r]
    files = [f for r in smoke["targets"] for f in r["files"] if "raw_bytes" in f]
    raw_per_file = float(np.mean([f["raw_bytes"] for f in files]))
    slim_per_file = float(np.mean([f["slim_bytes"] for f in files if "slim_bytes" in f]))
    dl_bytes = sum(r["raw_bytes"] for r in smoke["targets"])
    dl_secs = sum(r["download_s"] for r in smoke["targets"])
    mb_per_s = dl_bytes / 1e6 / dl_secs
    nbin = [r["search"]["n_binned"] for r in rows]
    bls_a, bls_k = fit_power(nbin, [r["search"]["bls_s"] for r in rows])
    tls_a, tls_k = fit_power(nbin, [r["search"]["tls_s"] for r in rows])
    # bins per early sector, from the smoke targets
    per_sector = np.median([r["search"]["n_binned"] / len(str(r["sectors_early"]).split(";")) for r in rows])
    cores = smoke["env"]["cpu_count"]

    log = pd.read_csv(ROOT / "data" / "sample_log.csv", dtype={"toi": str})
    pins = pd.read_csv(ROOT / "data" / "pinned_products.csv")
    inc = log[log["decision"] == "include"]
    b2 = inc[inc["b2_eligible"].astype(str).str.startswith("pending")]
    n_files = {"toi_host": int((pins["role"] == "toi_host").sum()), "pool": int((pins["role"] == "pool").sum())}

    def search_s(n_sectors):
        n = n_sectors * per_sector
        return bls_a * n ** bls_k + tls_a * n ** tls_k

    b2_early = b2["sectors_early"].astype(str).str.count(";") + 1
    b2_search_s = float(sum(search_s(n) for n in b2_early))
    total_bytes = (n_files["toi_host"] + n_files["pool"]) * raw_per_file
    slim_bytes = (n_files["toi_host"] + n_files["pool"]) * slim_per_file
    download_h = total_bytes / 1e6 / mb_per_s / 3600

    # B1: injections spread uniformly over 1..10 sectors (proposal; the grid is fixed in Phase 4)
    b1_sectors = np.arange(1, 11)
    b1_mean_s = float(np.mean([search_s(n) for n in b1_sectors]))  # one BLS + one TLS search per injection
    b1_s = args.b1_injections * b1_mean_s

    quota = json.loads(Path(args.quota).read_text())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    est = {
        "git_commit": commit, "smoke_file": str(args.smoke), "smoke_commit": smoke["git_commit"],
        "measured": {"n_targets": len(smoke["targets"]), "mean_raw_mb_per_file": raw_per_file / 1e6,
                     "mean_slim_mb_per_file": slim_per_file / 1e6, "download_mb_per_s": mb_per_s,
                     "bls_fit_a_k": [bls_a, bls_k], "tls_fit_a_k": [tls_a, tls_k],
                     "median_bins_per_sector": float(per_sector), "cpu_count": cores, "gpu": smoke["env"]["gpu"]},
        "frozen": {"tois_included": int(len(inc)), "b2_pending_h3": int(len(b2)), "pinned_files": n_files},
        "full_sample": {"download_gb": total_bytes / 1e9, "slim_cache_gb": slim_bytes / 1e9,
                        "download_wall_h_single_session": download_h,
                        "b2_bls_tls_wall_h": b2_search_s / 3600,
                        "b2_bls_tls_cpu_core_h": b2_search_s / 3600 * cores},
        "b1_proposal": {"injections": args.b1_injections, "searches_per_injection": "BLS + TLS (resolver cost not yet known)",
                        "wall_h": b1_s / 3600, "cpu_core_h": b1_s / 3600 * cores},
        "kaggle_quota": quota,
    }
    out = ROOT / "results" / "cost_estimate.json"
    out.write_text(json.dumps(est, indent=1))
    print(json.dumps(est, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
