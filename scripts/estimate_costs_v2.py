"""Cost estimate for the amended design (A1, A2, A3, G1), from committed files only.

Inputs: smoke_results.json (measured Kaggle costs), gate2b_results.json (T4 probe, parallel BLS,
GPU BLS timings), results/kaggle/quota_*.json, and the frozen data/ files.
Output: results/cost_estimate_v2.json.

Model:
  * astropy BLS / TLS: c * N_points * N_trial_periods (constants fitted on the smoke run, as in
    estimate_costs.py); exact per-target grid sizes reproduced.
  * GPU BLS (if accepted in gate2b): c_gpu * N_points * N_trial_periods fitted on the T4 timings.
  * Every Kaggle hour is a GPU-quota hour (G1: GPU T4 sessions only).
  * Seed searches in B2 and B1(ii) use the early / one-year window; B1(i) runs no search.
  * B1(i) per-injection preparation = wotan detrending of the unbinned light curve, costed from the
    smoke run's measured detrend time per sector; the resolver's own cost is unknown until Phase 3.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from exogargantua import sample as S  # noqa: E402
import estimate_costs as E1  # noqa: E402

B1_TOTAL, B1_SEARCH = 30000, 2000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", required=True)
    ap.add_argument("--gate2b", required=True)
    ap.add_argument("--quota", required=True)
    args = ap.parse_args()
    smoke = json.loads(Path(args.smoke).read_text())
    g2 = json.loads(Path(args.gate2b).read_text())
    rows = [r for r in smoke["targets"] if "search" in r]
    cores = smoke["env"]["cpu_count"]

    c_bls = float(np.median([r["search"]["bls_s"] / (r["search"]["n_binned"] * r["search"]["bls_n_periods"]) for r in rows]))
    c_tls = float(np.median([r["search"]["tls_s"] / (r["search"]["n_binned"] * r["search"]["tls_n_periods"]) for r in rows]))
    nbin = {r["toi"]: r["search"]["n_binned"] for r in rows}
    gpu = g2["gpu_bls"]
    gpu_ok = bool(gpu.get("accepted"))
    c_gpu = float(np.median([t["gpu_s"] / (nbin[t["toi"]] * t["n_periods"]) for t in gpu["targets"]
                             if t["n_periods"] > 50000])) if gpu["targets"] else float("nan")
    bins_per_sector = float(np.median([r["search"]["n_binned"] / len(str(r["sectors_early"]).split(";")) for r in rows]))
    detrend_per_sector = float(np.median([r["search"]["detrend_bin_s"] / len(str(r["sectors_early"]).split(";")) for r in rows]))
    par_speedup = g2["bls_parallel"]["serial_wall_s_smoke"] / g2["bls_parallel"]["wall_s"]

    starts = S.sector_start_btjd(pd.read_csv(sorted((ROOT / "data" / "raw").glob("tess_orbit_times_*.csv"))[0]))
    log = pd.read_csv(ROOT / "data" / "sample_log.csv", dtype={"toi": str})
    stars = pd.read_csv(ROOT / "data" / "stellar_params.csv").drop_duplicates("tic").set_index("tic")
    pins = pd.read_csv(ROOT / "data" / "pinned_products.csv", usecols=["tic", "sector", "role"])
    b2 = log[(log["decision"] == "include") & (log["b2_eligible"] == "pending_H3")]

    def grids(sectors, tic):
        span = E1.window_span(sectors, starts)
        npts = len(sectors) * bins_per_sector
        r = stars.loc[tic, "rad"] if tic in stars.index else np.nan
        m = stars.loc[tic, "mass"] if tic in stars.index else np.nan
        return npts, E1.n_bls_periods(span), E1.n_tls_periods(span, r, m)

    def search_cost(items):
        """items: list of (npts, n_bls, n_tls). Returns wall hours per BLS mode + TLS."""
        per = [n * nb for n, nb, _ in items]
        bls = sum(per)
        tls = sum(n * nt for n, _, nt in items) * c_tls
        # 4 workers, largest first: wall = max(total / workers, longest single search)
        par = max(bls / cores, max(per) if per else 0.0)
        return {"bls_serial_h": bls * c_bls / 3600, "bls_parallel_h": par * c_bls / 3600,
                "bls_gpu_h": bls * c_gpu / 3600 if gpu_ok else None, "tls_h": tls / 3600}

    b2_items = [grids([int(x) for x in str(r["sectors_early"]).split(";")], int(r["tic"])) for _, r in b2.iterrows()]
    rng = np.random.default_rng(20260930)
    pool = pins[pins["role"] == "pool"].groupby("tic")["sector"].apply(sorted)
    b1_items, b1_nsec = [], []
    for tic in rng.choice(pool.index.to_numpy(), size=min(400, len(pool)), replace=False):
        k = int(rng.integers(1, 11))
        secs = [s_ for s_ in pool[tic] if starts[s_] < starts[pool[tic][0]] + 365.25][:k]
        b1_items.append(grids(secs, int(tic)))
        b1_nsec.append(len(secs))
    per_inj = search_cost(b1_items)
    b1_search = {k: (v / len(b1_items) * B1_SEARCH if v is not None else None) for k, v in per_inj.items()}
    b1_prep_h = B1_TOTAL * float(np.mean(b1_nsec)) * detrend_per_sector / 3600
    b2_cost = search_cost(b2_items)

    def total(c, bls_mode):
        b = c[bls_mode]
        return None if b is None else b + c["tls_h"]

    n_files = int(len(pins))
    raw_mb = float(np.mean([f["raw_bytes"] for r in smoke["targets"] for f in r["files"] if "raw_bytes" in f])) / 1e6
    slim32_mb = float(np.mean([f["slim_bytes"] for r in smoke["targets"] for f in r["files"] if "slim_bytes" in f])) / 1e6
    read_s_per_file = float(np.median([r["read_cache_s"] / max(1, len(r["files"])) for r in smoke["targets"]]))
    dl_mb_s = sum(r["raw_bytes"] for r in smoke["targets"]) / 1e6 / sum(r["download_s"] for r in smoke["targets"])
    probe = g2["probe"]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    est = {
        "git_commit": commit, "inputs": {"smoke": args.smoke, "gate2b": args.gate2b, "quota": args.quota},
        "constants": {"c_bls": c_bls, "c_tls": c_tls, "c_gpu_bls": c_gpu, "gpu_bls_accepted": gpu_ok,
                      "bls_parallel_speedup_measured": par_speedup, "bins_per_sector": bins_per_sector,
                      "detrend_s_per_sector": detrend_per_sector, "cores": cores},
        "download": {"files": n_files, "raw_gb": n_files * raw_mb / 1e3, "raw_mb_per_file": raw_mb,
                     "download_h": n_files * raw_mb / dl_mb_s / 3600, "read_compact_h": n_files * read_s_per_file / 3600,
                     "compact_float32_gb": n_files * slim32_mb / 1e3,
                     "compact_float64_flux_gb_estimate": n_files * slim32_mb * 20 / 16 / 1e3,
                     "probe_disk": {k: v for k, v in probe.items() if k.startswith("disk_")}},
        "b2": {"targets": len(b2_items), **b2_cost,
               "total_h_parallel_bls": total(b2_cost, "bls_parallel_h"), "total_h_gpu_bls": total(b2_cost, "bls_gpu_h")},
        "b1": {"injections_total": B1_TOTAL, "full_search_subsample": B1_SEARCH,
               "mean_sectors_per_injection": float(np.mean(b1_nsec)),
               "search_subsample": {**b1_search, "total_h_parallel_bls": total(b1_search, "bls_parallel_h"),
                                    "total_h_gpu_bls": total(b1_search, "bls_gpu_h")},
               "wrong_seed_runs_prep_h": b1_prep_h,
               "wrong_seed_runs_resolver_h": "unknown until Phase 3 (resolver not built yet)"},
        "kaggle_quota": json.loads(Path(args.quota).read_text()),
    }
    out = ROOT / "results" / "cost_estimate_v2.json"
    out.write_text(json.dumps(est, indent=1, default=float))
    print(json.dumps({k: est[k] for k in ("constants", "download", "b2", "b1")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
