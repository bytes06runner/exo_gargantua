"""False-alarm test (pre-registered criterion, docs/success_criteria.md: confident resolution on scrambled data < 5 %).

Design (fixed 2026-10-20 before running; docs/decisions.md FA1):
  light curves  early-sector light curves of the 1,198 B2 TOIs (other TOIs on the host masked), as in B2
  inverted      f -> 2 - f (transits become brightenings; a dip search finds no real transit)
  scrambled     within each contiguous segment (gaps > 0.5 d split segments) the 1-day blocks are permuted
                (numpy.random.default_rng([20261020, tic])); single events survive, periodicity does not
  seed          GPU BLS peak on the modified light curve (search.prepare + the pre-registered BLS grid)
  resolver      the frozen resolver from that seed; rows stored so v1 and v2 are both scored later
Output: fa_<variant>_<shard>of<n>.jsonl.gz (no truth involved).
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
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from exogargantua import search as SE  # noqa: E402
from exogargantua.resolver import Star, alias_rows, resolve  # noqa: E402
import b2_resolve as BR  # noqa: E402
import b2_seeds as B2  # noqa: E402

RHO_SUN = 1.41
SCRAMBLE_SEED = 20261020


def scramble(t, f, tic, block=1.0, gap=0.5):
    rng = np.random.default_rng([SCRAMBLE_SEED, int(tic)])
    out = f.copy()
    seg_edges = np.r_[0, np.where(np.diff(t) > gap)[0] + 1, t.size]
    for a, b in zip(seg_edges[:-1], seg_edges[1:]):
        ts = t[a:b]
        bid = np.floor((ts - ts[0]) / block).astype(int)
        blocks = [np.where(bid == k)[0] + a for k in np.unique(bid)]
        perm = rng.permutation(len(blocks))
        out[a:b] = np.concatenate([f[blocks[k]] for k in perm])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["inverted", "scrambled"], required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", required=True)
    a = ap.parse_args()
    import torch
    from exogargantua import gpu_bls
    cache = str(B2.find_cache(a.cache))
    BR.init(cache)
    targets = B2.b2_targets()
    mine = targets.iloc[[i for i in range(len(targets)) if i % a.nshards == a.shard - 1]]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.variant}_{a.shard}of{a.nshards}"
    t_start, n = time.time(), 0
    with gzip.open(out / f"fa_{tag}.jsonl.gz", "wt") as fh:
        for _, r in mine.iterrows():
            rec = {"toi": r["toi"], "tic": int(r["tic"]), "variant": a.variant}
            t0 = time.time()
            try:
                t, f, m = B2.early_lc(BR.G["cache"], BR.G["excluded"], r, BR.G["toi_table"])
                t, f = t[m], f[m]
                ok = np.isfinite(t) & np.isfinite(f)
                o = np.argsort(t[ok])
                t, f = t[ok][o], f[ok][o] / np.median(f[ok])
                f = 2.0 - f if a.variant == "inverted" else scramble(t, f, r["tic"])
                tb, fb = SE.prepare(t, f)
                pmin, pmax, _ = SE.period_limits(tb)
                grid = SE.bls_grid(tb, pmin, pmax)
                pw = gpu_bls.bls_power(tb, fb, grid, SE.BLS_DURATIONS, device=a.device)
                torch.cuda.synchronize(a.device)
                P = float(grid[int(np.nanargmax(pw))])
                st = BR.G["stars"].loc[int(r["tic"])] if int(r["tic"]) in BR.G["stars"].index else None
                star = Star(float(st["rho"]) * RHO_SUN, float(st["e_rho"]) * RHO_SUN) if st is not None else Star()
                s, hyps = resolve(t, f, P, star=star)
                probs = np.array([s["hyp_probs"][h.key] for h in hyps])
                rec.update({"seed": {"engine": "bls_gpu", "P_seed": P, "bls_power_max": float(np.nanmax(pw))},
                            "summary": {k: v for k, v in s.items() if k != "hyp_probs"},
                            "rows": [{"r": k, "P": PP, "x": x} for k, PP, x in alias_rows(hyps, probs)]})
            except Exception as e:
                rec["error"] = repr(e)[:300]
            rec["wall_s"] = time.time() - t0
            fh.write(json.dumps(rec, default=float) + "\n")
            n += 1
            if n % 50 == 0:
                print(f"{n}/{len(mine)} {time.time() - t_start:.0f}s", flush=True)
    (out / f"done_{tag}.json").write_text(json.dumps({"git_commit": a.commit, "variant": a.variant, "n": int(len(mine)),
                                                       "wall_s": time.time() - t_start, "cache": cache}))


if __name__ == "__main__":
    main()
