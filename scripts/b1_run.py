"""B1 runs on Kaggle (docs/injection_design.md §3-4, §6).

  --mode wrong      B1(i): every injection, resolver seeded with its pre-drawn wrong seed (CPU sessions).
  --mode search     B1(ii): the 2,000-injection subsample; seed searches exactly as in B2 (search.prepare, then
                    --engine tls: search.tls_peak on CPU, or --engine bls_gpu: gpu_bls on one T4), then the
                    resolver seeded with that search's period.

Each injection's light curve: the host's cached PDCSAP flux in the listed sectors (finite points), multiplied by
the injected signal (exogargantua.inject; quadratic Claret limb darkening from the host's Teff/logg).
Output (JSON lines): injection id, seed, resolver summary and the per-alias feature rows. NO truth columns and
NO correctness labels are written; training labels (fit/calib) and scoring (test, guarded) are computed later.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from exogargantua import inject as I  # noqa: E402
from exogargantua.resolver import Star, alias_rows, resolve  # noqa: E402

RHO_SUN = 1.41
CACHE = None


def light_curve(row, z):
    tt, ff = [], []
    for s in str(row["sectors"]).split(";"):
        t, f = z[f"s{int(s):04d}_time"], z[f"s{int(s):04d}_flux"]
        ok = np.isfinite(t) & np.isfinite(f)
        tt.append(t[ok])
        ff.append(f[ok] / np.median(f[ok]))
    t, f = np.concatenate(tt), np.concatenate(ff)
    u = I.limb_darkening(row["teff"], row["logg"])
    if row["cls"] == "planet":
        sig, _ = I.planet_flux(t, row["P_true"], row["t0_true"], row["depth"], row["b"], row["r_star"], row["m_star"], u)
    else:
        sig, _ = I.eb_flux(t, row["P_true"], row["t0_true"], row["depth"], row["depth2"], row["b"], row["r_star"], row["m_star"], u,
                           ecc=row["ecc"], omega=row["omega"] if row["cls"] == "eb_ecc" else np.pi / 2)
    return t, f * sig


def run_resolver(t, f, P_seed, row):
    s, hyps = resolve(t, f, P_seed, star=Star(row["rho_solar"] * RHO_SUN, row["e_rho_solar"] * RHO_SUN))
    probs = np.array([s["hyp_probs"][h.key] for h in hyps])
    return {"summary": {k: v for k, v in s.items() if k != "hyp_probs"},
            "rows": [{"r": k, "P": P, "x": x} for k, P, x in alias_rows(hyps, probs)]}


def do_host(args):
    tic, rows, mode, engine, device = args
    out = []
    with np.load(Path(CACHE) / "lc" / f"{tic}.npz") as z:
        for row in rows:
            t0 = time.time()
            rec = {"inj_id": row["inj_id"]}
            try:
                t, f = light_curve(row, z)
                if mode == "wrong":
                    rec["seed"] = {"engine": "wrong", "P_seed": row["P_seed"]}
                    rec.update(run_resolver(t, f, row["P_seed"], row))
                else:
                    from exogargantua import search as SE
                    tb, fb = SE.prepare(t, f)
                    if engine == "tls":
                        sd = SE.tls_peak(tb, fb, 1, row["r_star"], row["m_star"])
                        P_seed = sd["tls_period"]
                    else:
                        import torch
                        from exogargantua import gpu_bls
                        pmin, pmax, _ = SE.period_limits(tb)
                        grid = SE.bls_grid(tb, pmin, pmax)
                        pw = gpu_bls.bls_power(tb, fb, grid, SE.BLS_DURATIONS, device=device)
                        torch.cuda.synchronize(device)
                        sd = {"bls_period": float(grid[int(np.nanargmax(pw))]), "bls_n_periods": int(grid.size)}
                        P_seed = sd["bls_period"]
                    rec["seed"] = {"engine": engine, "P_seed": P_seed, **sd}
                    rec.update(run_resolver(t, f, P_seed, row))
            except Exception as e:  # recorded, never dropped
                rec["error"] = repr(e)[:300]
            rec["wall_s"] = time.time() - t0
            out.append(rec)
    return out


def main():
    global CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["wrong", "search"], required=True)
    ap.add_argument("--engine", choices=["tls", "bls_gpu"], default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=1)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", required=True)
    a = ap.parse_args()
    import b2_seeds as B2
    CACHE = str(B2.find_cache(a.cache))
    inj = pd.read_csv(ROOT / "data" / "b1_injections.csv")
    if a.mode == "search":
        inj = inj[inj["inj_id"].isin(pd.read_csv(ROOT / "data" / "b1_search_subsample.csv")["inj_id"])]
    # shard by host (each host's light curve is read once), longest-processing-time on total sectors
    cost = inj.groupby("tic")["n_sectors"].sum().sort_values(ascending=False, kind="stable")
    load, owner = np.zeros(a.nshards), {}
    for tic, c in cost.items():
        k = int(np.argmin(load))
        owner[tic] = k + 1
        load[k] += c
    mine = inj[inj["tic"].map(owner) == a.shard]
    tag = f"{a.mode}{'_' + a.engine if a.engine else ''}_{a.shard}of{a.nshards}"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"b1_{tag}.jsonl.gz"
    jobs = [(int(t), g.to_dict("records"), a.mode, a.engine, a.device) for t, g in mine.groupby("tic", sort=False)]
    print(f"{tag}: {len(mine)} injections on {len(jobs)} hosts, cache {CACHE}", flush=True)
    t0, n = time.time(), 0
    workers = 1 if a.engine == "bls_gpu" else a.workers
    with gzip.open(path, "wt") as fh:
        if workers == 1:
            it = map(do_host, jobs)
        else:
            ex = ProcessPoolExecutor(workers)
            it = ex.map(do_host, jobs, chunksize=1)
        for recs in it:
            for r in recs:
                fh.write(json.dumps(r, default=float) + "\n")
            n += len(recs)
            if n % 200 < len(recs):
                print(f"{n}/{len(mine)} {time.time() - t0:.0f}s", flush=True)
    (out / f"done_{tag}.json").write_text(json.dumps({"git_commit": a.commit, "mode": a.mode, "engine": a.engine, "n": int(len(mine)),
                                                       "workers": workers, "wall_s": time.time() - t0, "cache": CACHE}))


if __name__ == "__main__":
    main()
