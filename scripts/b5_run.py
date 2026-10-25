"""B5 runs (docs/b5_definition.md): early-sector light curve of each included EB (pinned files, X10/X11 applied),
H3 eligibility from the catalog ephemeris, seed search exactly as B2 (search.prepare, then TLS 1.32 or the GPU BLS
port on the pre-registered grid), then the frozen resolver from that seed. Writes no correctness.
Output: b5_<engine>_<shard>of<n>.jsonl.gz.
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
from exogargantua import sample as S, search as SE  # noqa: E402
from exogargantua.resolver import Star, alias_rows, resolve  # noqa: E402

RHO_SUN = 1.41
G = {}


def eclipse_duration(r):
    for c in ("prim_width_pf", "prim_width_2g"):
        w = r.get(c, np.nan)
        if np.isfinite(w) and w > 0:
            return float(w * r["period"])
    return 0.1


def light_curve(tic, sectors):
    with np.load(G["cache"] / "lc" / f"{tic}.npz") as z:
        tt, ff = [], []
        for s in sectors:
            if (tic, s) in G["excluded"] or f"s{s:04d}_time" not in z.files:
                continue
            tt.append(z[f"s{s:04d}_time"]); ff.append(z[f"s{s:04d}_flux"])
    if not tt:
        return None, None
    t, f = np.concatenate(tt), np.concatenate(ff)
    o = np.argsort(t)
    return t[o], f[o]


def one(args):
    r, engine, device = args
    tic = int(r["tic"])
    rec = {"tic": tic, "engine": engine}
    t0 = time.time()
    try:
        secs = [int(x) for x in str(r["sectors_early"]).split(";")]
        t, f = light_curve(tic, secs)
        if t is None:
            rec["error"] = "no usable early sectors (X10/X11)"
            return rec
        rec["h3_eclipses"] = int(S.n_covered_transits(t, float(r["period"]), float(r["bjd0"]), eclipse_duration(r)))
        tb, fb = SE.prepare(t, f)
        st = G["stars"].loc[tic] if tic in G["stars"].index else None
        rs = float(st["rad"]) if st is not None else np.nan
        ms = float(st["mass"]) if st is not None else np.nan
        if engine == "tls":
            sd = SE.tls_peak(tb, fb, 1, rs, ms)
            P = sd["tls_period"]
        else:
            import torch
            from exogargantua import gpu_bls
            pmin, pmax, _ = SE.period_limits(tb)
            grid = SE.bls_grid(tb, pmin, pmax)
            pw = gpu_bls.bls_power(tb, fb, grid, SE.BLS_DURATIONS, device=device)
            torch.cuda.synchronize(device)
            P = float(grid[int(np.nanargmax(pw))])
            sd = {"bls_period": P, "bls_power_max": float(np.nanmax(pw)), "bls_n_periods": int(grid.size)}
        rec["seed"] = {"engine": engine, "P_seed": float(P), **{k: float(v) for k, v in sd.items() if isinstance(v, (int, float))}}
        star = Star(float(st["rho"]) * RHO_SUN, float(st["e_rho"]) * RHO_SUN) if st is not None and np.isfinite(st["rho"]) else Star()
        s, hyps = resolve(t, f, float(P), star=star)
        probs = np.array([s["hyp_probs"][h.key] for h in hyps])
        rec.update({"summary": {k: v for k, v in s.items() if k != "hyp_probs"},
                    "rows": [{"r": k, "P": PP, "x": x} for k, PP, x in alias_rows(hyps, probs)]})
    except Exception as e:
        rec["error"] = repr(e)[:300]
    rec["wall_s"] = time.time() - t0
    return rec


def init(cache):
    import b2_seeds as B2
    G["cache"] = Path(cache)
    man = pd.read_csv(G["cache"] / "manifest.csv")
    G["excluded"] = set(zip(man.loc[man["excluded"].notna(), "tic"], man.loc[man["excluded"].notna(), "sector"]))
    G["stars"] = pd.read_csv(ROOT / "data" / "stellar_params.csv").drop_duplicates("tic").set_index("tic")


def main():
    import b2_seeds as B2
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["tls", "bls_gpu"], required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", required=True)
    a = ap.parse_args()
    cache = str(B2.find_cache(a.cache))
    log = pd.read_csv(ROOT / "data" / "b5_sample_log.csv")
    eb = pd.read_csv(ROOT / "data" / "raw" / "tess_ebs_prsa2022_v1.0_20260930.csv")
    inc = log[log["decision"] == "include"].merge(eb[["tess_id", "prim_width_pf", "prim_width_2g"]].rename(columns={"tess_id": "tic"}),
                                                   on="tic", how="left").sort_values("tic")
    n_early = inc["sectors_early"].astype(str).str.count(";") + 1
    order = np.argsort(-n_early.to_numpy(), kind="stable")  # longest first, round-robin: balanced shards
    mine = inc.iloc[[i for k, i in enumerate(order) if k % a.nshards == a.shard - 1]]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.engine}_{a.shard}of{a.nshards}"
    jobs = [(r, a.engine, a.device) for r in mine.to_dict("records")]
    t0, n = time.time(), 0
    workers = 1 if a.engine == "bls_gpu" else a.workers
    with gzip.open(out / f"b5_{tag}.jsonl.gz", "wt") as fh:
        if workers == 1:
            init(cache)
            it = map(one, jobs)
        else:
            it = ProcessPoolExecutor(workers, initializer=init, initargs=(cache,)).map(one, jobs, chunksize=1)
        for rec in it:
            fh.write(json.dumps(rec, default=float) + "\n")
            n += 1
            if n % 50 == 0:
                print(f"{n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
                fh.flush()
    (out / f"done_{tag}.json").write_text(json.dumps({"git_commit": a.commit, "engine": a.engine, "n": len(jobs),
                                                       "wall_s": time.time() - t0, "cache": cache}))


if __name__ == "__main__":
    main()
