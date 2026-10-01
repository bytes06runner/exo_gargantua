"""Phase 3 synthetic check, run part (Kaggle CPU): fully synthetic TESS-like light curves
(exogargantua.resolver.synth; NOT the B1 benchmark), B1-style wrong seeds r ~ U(alias set), plus 10 %
signal-free light curves for the abstention check. Runs the resolver (principled combiner) and stores, per
target, the truth, the summary and the per-alias feature rows for the learned combiner.
Output: synth_<shard>of<n>.jsonl.gz (one JSON object per target).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exogargantua.resolver import Star, alias_rows, principled, resolve, synth  # noqa: E402

SEED = 20261001
P_NONE = 0.10


def one(args):
    i, seed = args
    rng = np.random.default_rng([seed, i])
    d = synth.make_target(rng, p_none=P_NONE)
    t0 = time.time()
    try:
        s, hyps = resolve(d["t"], d["f"], d["P_seed"], star=Star(d["star"]["rho"], d["star"]["rho_err"]))
    except Exception as e:  # recorded, never silently dropped
        return {"i": i, "error": repr(e), "cls": d["cls"]}
    probs = np.array([s["hyp_probs"][h.key] for h in hyps])
    rows = alias_rows(hyps, probs)
    truth = {k: d[k] for k in ("P_true", "t0_true", "depth", "b", "cls", "n_sectors", "seed_r", "P_seed", "snr_true", "n_in")}
    truth.update({"sigma_white": d["noise"]["sigma_white"], "red_amp": d["noise"]["red_amp"], "rho_true": d["star"]["rho_true"],
                  "rho": d["star"]["rho"]})
    return {"i": i, "truth": truth, "summary": {k: v for k, v in s.items() if k != "hyp_probs"},
            "rows": [{"r": k, "P": P, "x": x, "correct": bool(d["cls"] != "none" and abs(P / d["P_true"] - 1) < 1e-3)}
                     for k, P, x in rows], "wall_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--shard", type=int, default=1)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mine = [i for i in range(a.n) if i % a.nshards == a.shard - 1]
    path = out / f"synth_{a.shard}of{a.nshards}.jsonl.gz"
    t0 = time.time()
    with gzip.open(path, "wt") as fh, ProcessPoolExecutor(a.workers) as ex:
        for k, rec in enumerate(ex.map(one, [(i, a.seed) for i in mine], chunksize=4)):
            fh.write(json.dumps(rec, default=float) + "\n")
            if k % 100 == 0:
                print(f"{k}/{len(mine)} {time.time() - t0:.0f}s", flush=True)
    (out / f"done_{a.shard}of{a.nshards}.json").write_text(json.dumps({"git_commit": a.commit, "seed": a.seed, "n": a.n, "p_none": P_NONE,
                                                                        "targets": len(mine), "wall_s": time.time() - t0}))


if __name__ == "__main__":
    main()
