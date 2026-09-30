"""B2 holdout seed searches (sealed, amendment A4): TLS on CPU sessions, GPU BLS on T4 sessions (G2).

Reads the compact cache (lc/<tic>.npz + manifest.csv from the chunk-4 output or the owner's dataset),
builds each TOI's early-sector light curve exactly as the smoke/gate2b checks did (pinned early
sectors minus X10/X11 exclusions, other TOIs on the host masked, wotan + 10-min bins), and runs:
  --mode tls      transitleastsquares (CPU, all cores), shard i of N (cost-balanced, deterministic)
  --mode bls_gpu  PyTorch BLS (gpu_bls.py, accepted under A2 iv) on one CUDA device, shard i of N
Output: seeds_<mode>_<i>of<N>.csv with seed periods ONLY -- no truth columns, no accuracy (A4).
H3 (>= 2 truth-ephemeris transits in the early data; a pre-registered eligibility rule, not a
comparison of any method's output) is written separately to h3_<i>of<N>.csv in --mode tls.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from exogargantua import sample as S, search as SE  # noqa: E402
import smoke_run as SR  # noqa: E402
import estimate_costs as E1  # noqa: E402


def say(m):
    print(f"[{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}] {m}", flush=True)


def b2_targets():
    log = pd.read_csv(ROOT / "data" / "sample_log.csv", dtype={"toi": str})
    return log[(log["decision"] == "include") & (log["b2_eligible"] == "pending_H3")].sort_values("toi")


def shard(targets: pd.DataFrame, i: int, n: int) -> pd.DataFrame:
    """Deterministic longest-processing-time partition by predicted search cost."""
    starts = S.sector_start_btjd(pd.read_csv(sorted((ROOT / "data" / "raw").glob("tess_orbit_times_*.csv"))[0]))
    cost = []
    for _, r in targets.iterrows():
        secs = [int(x) for x in str(r["sectors_early"]).split(";")]
        span = E1.window_span(secs, starts)
        cost.append(len(secs) * E1.n_tls_periods(span, 1.0, 1.0))
    order = np.argsort(-np.asarray(cost), kind="stable")
    load = np.zeros(n)
    owner = np.empty(len(targets), int)
    for j in order:
        k = int(np.argmin(load))
        owner[j] = k
        load[k] += cost[j]
    return targets.iloc[np.where(owner == i - 1)[0]]


def find_cache(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    for m in Path("/kaggle/input").rglob("manifest.csv"):
        return m.parent
    raise SystemExit("cache not found")


def early_lc(cache: Path, excluded: set, r, toi_table):
    tic, toi = int(r["tic"]), float(r["toi"])
    with np.load(cache / "lc" / f"{tic}.npz") as z:
        te, fe = [], []
        for s in [int(x) for x in str(r["sectors_early"]).split(";")]:
            if (tic, s) in excluded or f"s{s:04d}_time" not in z.files:
                continue
            te.append(z[f"s{s:04d}_time"]); fe.append(z[f"s{s:04d}_flux"])
    t, f = np.concatenate(te), np.concatenate(fe)
    m = SR.other_toi_mask(t, tic, toi, toi_table)
    return t, f, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tls", "bls_gpu"], required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--only-tois", default="", help="debug only: comma-separated TOIs")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = find_cache(args.cache)
    man = pd.read_csv(cache / "manifest.csv")
    excluded = set(zip(man.loc[man["excluded"].notna(), "tic"], man.loc[man["excluded"].notna(), "sector"]))
    snaps = json.loads((ROOT / "data" / "raw" / "snapshots.json").read_text())
    toi_table = S.normalise_toi(pd.read_csv(ROOT / snaps["S-TOI"]["file"], comment="#"))
    stars = pd.read_csv(ROOT / "data" / "stellar_params.csv").drop_duplicates("tic").set_index("tic")
    mine = shard(b2_targets(), args.shard, args.nshards)
    if args.only_tois:
        allb2 = b2_targets()
        mine = allb2[allb2["toi"].isin(args.only_tois.split(","))]
    say(f"{args.mode} shard {args.shard}/{args.nshards}: {len(mine)} TOIs, cache {cache}")

    tag = f"{args.shard}of{args.nshards}"
    seeds_path, h3_path = out / f"seeds_{args.mode}_{tag}.csv", out / f"h3_{tag}.csv"
    done = set(pd.read_csv(seeds_path, dtype={"toi": str})["toi"]) if seeds_path.exists() else set()
    if args.mode == "tls":  # Tschudi (2026a) correction applied to the same TLS result (docs/baselines.md §3)
        from exogargantua.baselines import tschudi as TS
        ts_mod = TS.load(TS.fetch(Path("/tmp/tschudi_repo") if Path("/tmp").exists() else out / "_tschudi"))
        ts_cfg = TS.config(ts_mod, os.cpu_count() or 1)
    if args.mode == "bls_gpu":
        import torch
        from exogargantua import gpu_bls
        dev = args.device or "cuda:0"
    for _, r in mine.iterrows():
        if r["toi"] in done:
            continue
        t0 = time.time()
        t, f, m = early_lc(cache, excluded, r, toi_table)
        tb, fb = SE.prepare(t[m], f[m])
        row = {"toi": r["toi"], "tic": int(r["tic"]), "n_binned": int(tb.size), "prepare_s": time.time() - t0}
        if args.mode == "tls":
            tic = int(r["tic"])
            rs = float(stars.loc[tic, "rad"]) if tic in stars.index else np.nan
            ms = float(stars.loc[tic, "mass"]) if tic in stars.index else np.nan
            tls_out, tls_res = SE.tls_peak(tb, fb, os.cpu_count() or 1, rs, ms, return_results=True)
            row.update(tls_out)
            row.update(TS.correct(ts_mod, ts_cfg, tls_res, tb, fb, rs, ms))
            # H3: pre-registered eligibility (truth-ephemeris coverage), kept in a separate file
            trow = toi_table[toi_table["toi"] == float(r["toi"])].iloc[0]
            n_tr = S.n_covered_transits(t, float(r["truth_period"]), float(trow["epoch_btjd"]),
                                        float(trow["duration_h"]) / 24.0)
            pd.DataFrame([{"toi": r["toi"], "h3_transits_in_early": n_tr, "b2_eligible": "yes" if n_tr >= 2 else "no (H3)"}]
                         ).to_csv(h3_path, mode="a", header=not h3_path.exists(), index=False)
        else:
            pmin, pmax, _ = SE.period_limits(tb)
            grid = SE.bls_grid(tb, pmin, pmax)
            t1 = time.time()
            pw = gpu_bls.bls_power(tb, fb, grid, SE.BLS_DURATIONS, device=dev)
            if str(dev).startswith("cuda"):
                torch.cuda.synchronize(dev)
            i = int(np.argmax(pw))
            row.update({"bls_period": float(grid[i]), "bls_index": i, "bls_n_periods": int(len(grid)),
                        "bls_power_max": float(pw[i]), "bls_s": time.time() - t1, "device": dev})
        row["wall_s"] = time.time() - t0
        pd.DataFrame([row]).to_csv(seeds_path, mode="a", header=not seeds_path.exists(), index=False)
        say(f"  {r['toi']}: {row['wall_s']:.0f}s")
    (out / f"done_{args.mode}_{tag}.json").write_text(json.dumps(
        {"git_commit": args.commit, "mode": args.mode, "shard": args.shard, "nshards": args.nshards,
         "n": int(len(mine)), "cache": str(cache), "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
