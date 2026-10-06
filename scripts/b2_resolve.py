"""B2: run the frozen resolver on every B2 TOI from every seed source (Kaggle CPU). Reads no truth.

Light curve: each TOI's pinned early sectors, other TOIs on the host masked -- exactly as the B2 seed runs
(scripts/b2_seeds.py::early_lc). Seeds (all committed and sealed, docs/baselines.md):
  tls, tschudi           results/kaggle/b2/*/b2/seeds_tls_*.csv (tls_period, tschudi_period)
  bls                    results/kaggle/b2/*/b2/seeds_bls_gpu_*.csv (bls_period)
  spoc1, spocE, qlp_hist, spoc_hist   results/b2/seeds_catalog.csv
Each (TOI, seed period) is resolved once; seed sources that give the same period share the run. The output keeps
the per-alias rows so v1 and v2 (which differ only in calibration, V2-a) are both scored from the same runs.
Output: b2res_<shard>of<n>.jsonl.gz.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import glob  # noqa: E402
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
from exogargantua import sample as S  # noqa: E402
from exogargantua.resolver import Star, alias_rows, resolve  # noqa: E402
import b2_seeds as B2  # noqa: E402

RHO_SUN = 1.41
SOURCES = ("tls", "tschudi", "bls", "spoc1", "spocE", "qlp_hist", "spoc_hist")
G = {}


def seed_table():
    tls = pd.concat(pd.read_csv(f, dtype={"toi": str}) for f in glob.glob(str(ROOT / "results/kaggle/b2/*/b2/seeds_tls_*.csv")))
    bls = pd.concat(pd.read_csv(f, dtype={"toi": str}) for f in glob.glob(str(ROOT / "results/kaggle/b2/*/b2/seeds_bls_gpu_*.csv")))
    cat = pd.read_csv(ROOT / "results" / "b2" / "seeds_catalog.csv", dtype={"toi": str})
    t = tls[["toi", "tls_period", "tschudi_period"]].merge(bls[["toi", "bls_period"]], on="toi", how="outer")
    t = t.merge(cat[["toi", "spoc1_period", "spocE_period", "qlp_hist_period", "spoc_hist_period"]], on="toi", how="left")
    return t.rename(columns={"tls_period": "tls", "tschudi_period": "tschudi", "bls_period": "bls", "spoc1_period": "spoc1",
                             "spocE_period": "spocE", "qlp_hist_period": "qlp_hist", "spoc_hist_period": "spoc_hist"}).set_index("toi")


def one(args):
    r, seeds = args
    out = {"toi": r["toi"], "tic": int(r["tic"]), "seeds": {}, "runs": {}}
    try:
        t, f, m = B2.early_lc(G["cache"], G["excluded"], r, G["toi_table"])
        t, f = t[m], f[m]
        ok = np.isfinite(t) & np.isfinite(f)
        t, f = t[ok], f[ok] / np.median(f[ok])
        st = G["stars"].loc[int(r["tic"])] if int(r["tic"]) in G["stars"].index else None
        star = Star(float(st["rho"]) * RHO_SUN, float(st["e_rho"]) * RHO_SUN) if st is not None else Star()
    except Exception as e:
        out["error"] = f"load: {e!r}"[:300]
        return out
    for src in SOURCES:
        P = seeds.get(src)
        out["seeds"][src] = float(P) if P is not None and np.isfinite(P) and P > 0 else None
    for P in sorted({v for v in out["seeds"].values() if v is not None}):
        key = f"{P:.10g}"
        t0 = time.time()
        try:
            s, hyps = resolve(t, f, P, star=star)
            probs = np.array([s["hyp_probs"][h.key] for h in hyps])
            out["runs"][key] = {"summary": {k: v for k, v in s.items() if k != "hyp_probs"},
                                "rows": [{"r": k, "P": PP, "x": x} for k, PP, x in alias_rows(hyps, probs)], "wall_s": time.time() - t0}
        except Exception as e:
            out["runs"][key] = {"error": repr(e)[:300]}
    return out


def init(cache):
    man = pd.read_csv(Path(cache) / "manifest.csv")
    G["cache"] = Path(cache)
    G["excluded"] = set(zip(man.loc[man["excluded"].notna(), "tic"], man.loc[man["excluded"].notna(), "sector"]))
    snaps = json.loads((ROOT / "data" / "raw" / "snapshots.json").read_text())
    G["toi_table"] = S.normalise_toi(pd.read_csv(ROOT / snaps["S-TOI"]["file"], comment="#"))
    G["stars"] = pd.read_csv(ROOT / "data" / "stellar_params.csv").drop_duplicates("tic").set_index("tic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", required=True)
    a = ap.parse_args()
    cache = str(B2.find_cache(a.cache))
    seeds = seed_table()
    mine = B2.shard(B2.b2_targets(), a.shard, a.nshards)
    jobs = [(r.to_dict(), seeds.loc[r["toi"]].to_dict() if r["toi"] in seeds.index else {}) for _, r in mine.iterrows()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.shard}of{a.nshards}"
    t0, n = time.time(), 0
    with gzip.open(out / f"b2res_{tag}.jsonl.gz", "wt") as fh, ProcessPoolExecutor(a.workers, initializer=init, initargs=(cache,)) as ex:
        for rec in ex.map(one, jobs, chunksize=1):
            fh.write(json.dumps(rec, default=float) + "\n")
            n += 1
            if n % 50 == 0:
                print(f"{n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    (out / f"done_{tag}.json").write_text(json.dumps({"git_commit": a.commit, "n": len(jobs), "wall_s": time.time() - t0, "cache": cache}))


if __name__ == "__main__":
    main()
