"""P07 + P08 injection-pool screen (docs/sample_definition.md §5, amendment A5).

Per pool star:
  P07  drop sectors with CROWDSAP < 0.9 (and X10/X11 exclusions from the cache manifest);
       the star is excluded if fewer than 2 sectors remain.
  A5   screening window = the first <= 10 remaining sectors that start within 365.25 d of the
       first remaining one (the data its injections use; docs/injection_design.md §1).
  P08  BLS on the uninjected window with the pre-registered §6 configuration (PDCSAP, wotan
       biweight 0.75 d, 10-min bins, autoperiod grid, likelihood objective);
       SDE = (max(power) - mean(power)) / std(power) over the full trial-period grid
       (Kovacs et al. 2002 definition, decision V1); star excluded if SDE >= 9.
  A6   (docs/decisions.md) the same spectrum, SDE computed after TLS's
       spectrum detrending (src/exogargantua/sde.py). Both values are recorded for every star: column
       `sde` / `p08` is the pre-registered raw rule, `sde_a6` / `p08_a6` the A6 rule; `sde_a6_cellmax` and
       `sde_dense_*` are diagnostics only and never decide anything.
Engines: --engine astropy (CPU) or --engine gpu (src/exogargantua/gpu_bls.py, one CUDA device).
Output rows: tic, p07 status, window sectors, grid size, peak period, power stats, SDE, timings.
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
from exogargantua import p08io, sample as S, search as SE, sde as SD  # noqa: E402
import b2_seeds as B2  # noqa: E402  (find_cache)
import estimate_costs as E1  # noqa: E402

CROWD_MIN = 0.9
SDE_MAX = 9.0


def say(m):
    print(f"[{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}] {m}", flush=True)


def sde(power: np.ndarray) -> float:
    p = np.asarray(power, float)
    p = p[np.isfinite(p)]
    return float((p.max() - p.mean()) / p.std())


def star_window(tic, man_by_tic, starts):
    m = man_by_tic.get(tic)
    if m is None:
        return None, "P07: no manifest rows"
    ok = m[(m["excluded"].isna()) & (pd.to_numeric(m["crowdsap"], errors="coerce") >= CROWD_MIN)]
    secs = sorted(int(s) for s in ok["sector"])
    if len(secs) < 2:
        return None, f"P07: {len(secs)} sector(s) with CROWDSAP >= {CROWD_MIN} after X10/X11"
    win = [s for s in secs if starts[s] < starts[secs[0]] + 365.25][:10]
    return win, ""


def predicted_cost(win, starts, bins_per_sector=3275.0):
    return len(win) * bins_per_sector * E1.n_bls_periods(E1.window_span(win, starts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["astropy", "gpu"], required=True)
    ap.add_argument("--tics-file", default=None, help="csv with a 'tic' column (validation list)")
    ap.add_argument("--shard", type=int, default=1)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=4, help="astropy: stars searched in parallel")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--dense-diagnostics", action="store_true", help="also record the diagnostic sde_dense_* columns")
    ap.add_argument("--save-spectra", action="store_true", help="write each star's power spectrum (float32 npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = B2.find_cache(args.cache)
    man = pd.read_csv(cache / "manifest.csv")
    man_by_tic = {int(k): g for k, g in man.groupby("tic")}
    starts = S.sector_start_btjd(pd.read_csv(sorted((ROOT / "data" / "raw").glob("tess_orbit_times_*.csv"))[0]))
    if args.tics_file:
        tics = [int(t) for t in pd.read_csv(ROOT / args.tics_file)["tic"]]
    else:
        pool = pd.read_csv(ROOT / "data" / "injection_pool_log.csv")
        tics = [int(t) for t in pool.loc[pool["decision"] == "include", "tic"]]
    # deterministic cost-balanced sharding
    wins = {t: star_window(t, man_by_tic, starts) for t in tics}
    cost = {t: (predicted_cost(w, starts) if w else 0.0) for t, (w, _) in wins.items()}
    load, owner = np.zeros(args.nshards), {}
    for t in sorted(tics, key=lambda t: (-cost[t], t)):
        k = int(np.argmin(load))
        owner[t] = k + 1
        load[k] += cost[t]
    mine = [t for t in tics if owner[t] == args.shard]
    tag = f"{args.engine}_{args.shard}of{args.nshards}"
    say(f"{tag}: {len(mine)} stars, cache {cache}")
    path = out / f"p08_{tag}.csv"
    stars = pd.read_csv(ROOT / "data" / "stellar_params.csv").drop_duplicates("tic").set_index("tic")
    spec_dir = out / "spectra"
    if args.save_spectra:
        spec_dir.mkdir(exist_ok=True)
    cols = p08io.A6_FULL_COLS if args.dense_diagnostics else p08io.A6_NODENSE_COLS
    done = set(p08io.read_rows(path)["tic"]) if path.exists() else set()

    def prep(t):
        win, why = wins[t]
        row = {"tic": t, "p07": "exclude" if why else "pass", "p07_reason": why}
        if why:
            return row, None, None
        with np.load(cache / "lc" / f"{t}.npz") as z:
            tt = np.concatenate([z[f"s{s:04d}_time"] for s in win])
            ff = np.concatenate([z[f"s{s:04d}_flux"] for s in win])
        tb, fb = SE.prepare(tt, ff)
        rs = float(stars.loc[t, "rad"]) if t in stars.index else np.nan
        ms = float(stars.loc[t, "mass"]) if t in stars.index else np.nan
        row.update({"window_sectors": ";".join(map(str, win)), "span_d": float(tb.max() - tb.min()), "n_binned": int(tb.size),
                    "rs": rs, "ms": ms})
        tb_of[t] = tb
        return row, tb, fb

    tb_of = {}

    def finish(row, grid, pw, iin, secs, fb):
        i = int(np.nanargmax(pw))
        s = sde(pw)
        t0 = time.time()
        ptls = SD.tls_periods(tb_of[row["tic"]], row.pop("rs"), row.pop("ms"))
        ok = np.isfinite(pw)
        s_a6, peak_a6 = SD.sde_a6(grid[ok], pw[ok], iin[ok], fb, ptls)
        diag = {"sde_a6_cellmax": SD.sde_a6_cellmax(grid[ok], pw[ok], iin[ok], fb, ptls)}
        if args.dense_diagnostics:
            diag.update({"sde_dense_tlspts": SD.sde_dense(pw[ok], iin[ok], fb, len(ptls), "tls_points"),
                         "sde_dense_scaled": SD.sde_dense(pw[ok], iin[ok], fb, len(ptls), "scaled")})
        row.update({"n_periods": int(len(grid)), "peak_period": float(grid[i]), "power_max": float(pw[i]),
                    "power_mean": float(np.nanmean(pw)), "power_std": float(np.nanstd(pw)), "sde": s,
                    "p08": "exclude" if s >= SDE_MAX else "pass",
                    "n_tls_periods": int(len(ptls)), "sde_a6": s_a6, "peak_period_a6": peak_a6,
                    "p08_a6": "exclude" if s_a6 >= SDE_MAX else "pass",
                    **diag,
                    "search_s": secs, "a6_s": time.time() - t0, "engine": args.engine})
        if args.save_spectra:
            np.savez_compressed(spec_dir / f"{row['tic']}_{args.engine}.npz", period=grid.astype(np.float64),
                                power=np.asarray(pw, np.float64), ivar_in=np.asarray(iin, np.float32),
                                periods_tls=ptls, flux=np.asarray(fb, np.float64))
        p08io.write_row(path, row, cols)
        say(f"  {row['tic']}: SDE raw {s:.4f}, A6 {s_a6:.4f} ({secs:.0f}s)")

    todo = [t for t in mine if t not in done]
    if args.engine == "gpu":
        import torch
        from exogargantua import gpu_bls
        for t in todo:
            row, tb, fb = prep(t)
            if tb is None:
                p08io.write_row(path, row, cols)
                continue
            pmin, pmax, _ = SE.period_limits(tb)
            grid = SE.bls_grid(tb, pmin, pmax)
            t0 = time.time()
            pw, iin = gpu_bls.bls_power(tb, fb, grid, SE.BLS_DURATIONS, device=args.device, return_ivar_in=True)
            if str(args.device).startswith("cuda"):
                torch.cuda.synchronize(args.device)
            finish(row, grid, pw, iin, time.time() - t0, fb)
    else:
        from concurrent.futures import ProcessPoolExecutor
        prepped = [prep(t) for t in todo]
        for row, tb, fb in prepped:
            if tb is None:
                p08io.write_row(path, row, cols)
        jobs = [(row, tb, fb) for row, tb, fb in prepped if tb is not None]
        with ProcessPoolExecutor(args.workers) as ex:
            for (row, _, fb), res in zip(jobs, ex.map(_astropy_power, [(tb, fb) for _, tb, fb in jobs])):
                grid, pw, iin, secs = res
                finish(row, grid, pw, iin, secs, fb)
    (out / f"done_{tag}.json").write_text(json.dumps({"git_commit": args.commit, "engine": args.engine, "n": len(mine),
                                                      "cache": str(cache), "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat()}))
    return 0


def _astropy_power(args):
    from astropy.timeseries import BoxLeastSquares
    tb, fb = args
    t0 = time.time()
    pmin, pmax, _ = SE.period_limits(tb)
    grid = SE.bls_grid(tb, pmin, pmax)
    res = BoxLeastSquares(tb, fb).power(grid, SE.BLS_DURATIONS, objective="likelihood")
    pw = np.asarray(res.power)
    return grid, pw, SD.ivar_in_from_depth(pw, np.asarray(res.depth)), time.time() - t0


if __name__ == "__main__":
    sys.exit(main())
