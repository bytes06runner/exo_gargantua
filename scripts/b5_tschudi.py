"""POST HOC baseline (decision B5-tsch): the Tschudi (2026a) harmonic correction on the B5 eclipsing binaries.
For each included EB: early-sector light curve and TLS exactly as in the B5 rerun (search.prepare, TLS 1.32 with TIC
stellar parameters from data/b5_stellar_params.csv), then their resolve_harmonic_alias on that same TLS result through
the adapter used in B2 (thresholds at their defaults, known-planet lookup emptied). Writes no correctness.
Output: b5_tschudi_<shard>of<n>.jsonl.gz."""

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
from exogargantua import search as SE  # noqa: E402
from exogargantua.baselines import tschudi as TS  # noqa: E402
import b5_run as B5  # noqa: E402

T = {}


def init(cache, stars, repo):
    B5.init(cache, stars)
    T["mod"] = TS.load(Path(repo))
    T["cfg"] = TS.config(T["mod"], 1)


def one(r):
    tic = int(r["tic"])
    rec = {"tic": tic}
    t0 = time.time()
    try:
        t, f = B5.light_curve(tic, [int(x) for x in str(r["sectors_early"]).split(";")])
        if t is None:
            rec["error"] = "no usable early sectors (X10/X11)"
            return rec
        tb, fb = SE.prepare(t, f)
        st = B5.G["stars"].loc[tic] if tic in B5.G["stars"].index else None
        rs = float(st["rad"]) if st is not None else np.nan
        ms = float(st["mass"]) if st is not None else np.nan
        out, res = SE.tls_peak(tb, fb, 1, rs, ms, return_results=True)
        rec.update({k: float(v) for k, v in out.items()})
        rec.update(TS.correct(T["mod"], T["cfg"], res, tb, fb, rs, ms))
    except Exception as e:
        rec["error"] = repr(e)[:300]
    rec["wall_s"] = time.time() - t0
    return rec


def main():
    import b2_seeds as B2
    ap = argparse.ArgumentParser()
    ap.add_argument("--stars", default="data/b5_stellar_params.csv")
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", required=True)
    a = ap.parse_args()
    cache = str(B2.find_cache(a.cache))
    repo = str(TS.fetch(Path("/tmp/tschudi_repo")))
    log = pd.read_csv(ROOT / "data" / "b5_sample_log.csv")
    inc = log[log["decision"] == "include"].sort_values("tic")
    n_early = inc["sectors_early"].astype(str).str.count(";") + 1
    order = np.argsort(-n_early.to_numpy(), kind="stable")  # same sharding as b5_run.py
    mine = inc.iloc[[i for k, i in enumerate(order) if k % a.nshards == a.shard - 1]]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"tschudi_{a.shard}of{a.nshards}"
    t0, n = time.time(), 0
    with gzip.open(out / f"b5_{tag}.jsonl.gz", "wt") as fh:
        for rec in ProcessPoolExecutor(a.workers, initializer=init, initargs=(cache, a.stars, repo)).map(one, mine.to_dict("records"), chunksize=1):
            fh.write(json.dumps(rec, default=float) + "\n")
            n += 1
            if n % 25 == 0:
                print(f"{n}/{len(mine)} {time.time() - t0:.0f}s", flush=True)
                fh.flush()
    (out / f"done_{tag}.json").write_text(json.dumps({"git_commit": a.commit, "n": len(mine), "wall_s": time.time() - t0, "cache": cache,
                                                       "tschudi_commit": TS.COMMIT}))


if __name__ == "__main__":
    main()
