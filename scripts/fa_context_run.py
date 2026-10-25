"""POST HOC false-alarm context (owner instruction 2026-10-26; not pre-registered).

On exactly the light curves of the false-alarm test FA1 (scripts/false_alarm_run.py: B2 TOI early-sector light
curves, inverted f -> 2 - f, or scrambled 1-day blocks with rng [20261020, tic]), record the detection statistics
of the searches themselves, so the paper can state how often the search crosses SDE >= 9 on these light curves:
  --engine tls      TLS 1.32 SDE (search.tls_peak with TIC stellar parameters), one thread per worker process
  --engine bls_gpu  A6 SDE of the GPU BLS spectrum (sde.sde_a6) and the raw BLS SDE (decision V1), one process per T4
Output: fac_<engine>_<variant>_<shard>of<n>.jsonl.gz.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from exogargantua import search as SE, sde as SD  # noqa: E402
import b2_resolve as BR  # noqa: E402
import b2_seeds as B2  # noqa: E402
from false_alarm_run import scramble  # noqa: E402


def modified_lc(r, variant):
    t, f, m = B2.early_lc(BR.G["cache"], BR.G["excluded"], r, BR.G["toi_table"])
    t, f = t[m], f[m]
    ok = np.isfinite(t) & np.isfinite(f)
    o = np.argsort(t[ok])
    t, f = t[ok][o], f[ok][o] / np.median(f[ok])
    return t, (2.0 - f if variant == "inverted" else scramble(t, f, r["tic"]))


def stellar(tic):
    st = BR.G["stars"].loc[tic] if tic in BR.G["stars"].index else None
    return (float(st["rad"]), float(st["mass"])) if st is not None else (np.nan, np.nan)


def tls_one(args):
    r, variant = args
    rec = {"toi": r["toi"], "tic": int(r["tic"]), "variant": variant}
    t0 = time.time()
    try:
        t, f = modified_lc(r, variant)
        tb, fb = SE.prepare(t, f)
        rs, ms = stellar(int(r["tic"]))
        out = SE.tls_peak(tb, fb, 1, rs, ms)
        rec.update({"tls_sde": out["tls_sde"], "tls_period": out["tls_period"]})
    except Exception as e:
        rec["error"] = repr(e)[:200]
    rec["wall_s"] = time.time() - t0
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["tls", "bls_gpu"], required=True)
    ap.add_argument("--variant", choices=["inverted", "scrambled"], required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", required=True)
    a = ap.parse_args()
    cache = str(B2.find_cache(None))
    BR.init(cache)
    targets = B2.b2_targets()
    mine = targets.iloc[[i for i in range(len(targets)) if i % a.nshards == a.shard - 1]]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.engine}_{a.variant}_{a.shard}of{a.nshards}"
    t_start = time.time()
    with gzip.open(out / f"fac_{tag}.jsonl.gz", "wt") as fh:
        if a.engine == "tls":
            with ProcessPoolExecutor(a.workers, initializer=BR.init, initargs=(cache,)) as ex:
                for k, rec in enumerate(ex.map(tls_one, [(r, a.variant) for r in mine.to_dict("records")], chunksize=1)):
                    fh.write(json.dumps(rec, default=float) + "\n")
                    if k % 50 == 0:
                        print(f"{k}/{len(mine)} {time.time() - t_start:.0f}s", flush=True)
                        fh.flush()
        else:
            import torch
            from exogargantua import gpu_bls
            for k, r in enumerate(mine.to_dict("records")):
                rec = {"toi": r["toi"], "tic": int(r["tic"]), "variant": a.variant}
                t0 = time.time()
                try:
                    t, f = modified_lc(r, a.variant)
                    tb, fb = SE.prepare(t, f)
                    pmin, pmax, _ = SE.period_limits(tb)
                    grid = SE.bls_grid(tb, pmin, pmax)
                    pw, iin = gpu_bls.bls_power(tb, fb, grid, SE.BLS_DURATIONS, device=a.device, return_ivar_in=True)
                    torch.cuda.synchronize(a.device)
                    ok = np.isfinite(pw)
                    rs, ms = stellar(int(r["tic"]))
                    s_a6, p_a6 = SD.sde_a6(grid[ok], pw[ok], iin[ok], fb, SD.tls_periods(tb, rs, ms))
                    rec.update({"sde_a6": s_a6, "peak_period_a6": p_a6, "sde_raw": SD.sde_raw_power(pw[ok]),
                                "bls_period": float(grid[int(np.nanargmax(pw))])})
                except Exception as e:
                    rec["error"] = repr(e)[:200]
                rec["wall_s"] = time.time() - t0
                fh.write(json.dumps(rec, default=float) + "\n")
                if k % 50 == 0:
                    print(f"{k}/{len(mine)} {time.time() - t_start:.0f}s", flush=True)
                    fh.flush()
    (out / f"done_{tag}.json").write_text(json.dumps({"git_commit": a.commit, "engine": a.engine, "variant": a.variant,
                                                       "n": int(len(mine)), "wall_s": time.time() - t_start}))


if __name__ == "__main__":
    main()
