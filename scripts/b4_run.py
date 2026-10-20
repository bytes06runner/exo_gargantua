"""B4 runtime on a sector-scale sample: all 19,999 SPOC 120-s light curves of Sector 49 (data/b4_pins.csv), from the
compact caches built by kaggle/jobs/b4_cache_* (attached as kernel sources).
  --stage bls       (GPU session, one process per T4): per light curve, time to load, prepare (wotan + 10-min bins)
                    and run GPU BLS on the pre-registered grid; seed = BLS peak
  --stage resolver  (CPU session, 4 workers, 1 BLAS thread each): per light curve, time of the frozen resolver from the
                    BLS seed (seeds read from the attached bls outputs)
Per-light-curve total = load + prepare + BLS + resolver (stages timed on their own hardware, reported separately).
Hardware recorded in each done file. TLS is not run (owner decision); its per-light-curve time is quoted from the
B2/B1 runs. Output: b4_<stage>_<shard>of<n>.jsonl.gz.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import glob  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from b1_run import read_records  # noqa: E402

SECTOR = 49


def caches():
    return sorted({Path(m).parent for m in glob.glob("/kaggle/input/**/manifest.csv", recursive=True)})


def lc_files():
    out = {}
    for c in caches():
        for f in (c / "lc").glob("*.npz"):
            out[int(f.stem)] = f
    return out


def load(path):
    with np.load(path) as z:
        if f"s{SECTOR:04d}_time" not in z.files:
            return None, None
        return z[f"s{SECTOR:04d}_time"], z[f"s{SECTOR:04d}_flux"]


def resolver_one(args):
    from exogargantua.resolver import resolve
    tic, path, P = args
    t0 = time.time()
    t, f = load(path)
    t_load = time.time() - t0
    if t is None or not np.isfinite(P):
        return {"tic": tic, "error": "no data or seed"}
    t1 = time.time()
    try:
        s, _ = resolve(t, f, P)
        return {"tic": tic, "load_s": t_load, "resolver_s": time.time() - t1, "n_points": int(t.size), "p_max": s["p_max"]}
    except Exception as e:
        return {"tic": tic, "error": repr(e)[:200], "resolver_s": time.time() - t1}


def hardware(device=None):
    hw = {"platform": platform.platform(), "cpu": platform.processor(), "cpu_count": os.cpu_count()}
    try:
        hw["cpu_model"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        pass
    if device:
        import torch
        hw["gpu"] = torch.cuda.get_device_name(device)
    return hw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["bls", "resolver"], required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", required=True)
    a = ap.parse_args()
    files = lc_files()
    tics = sorted(files)
    mine = tics[a.shard - 1::a.nshards]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.stage}_{a.shard}of{a.nshards}"
    t_start = time.time()
    with gzip.open(out / f"b4_{tag}.jsonl.gz", "wt") as fh:
        if a.stage == "bls":
            import torch
            from exogargantua import gpu_bls, search as SE
            for k, tic in enumerate(mine):
                rec = {"tic": tic}
                t0 = time.time()
                t, f = load(files[tic])
                rec["load_s"] = time.time() - t0
                if t is None:
                    rec["error"] = "no sector-49 data"
                else:
                    try:
                        t1 = time.time()
                        tb, fb = SE.prepare(t, f)
                        rec["prepare_s"] = time.time() - t1
                        t2 = time.time()
                        pmin, pmax, _ = SE.period_limits(tb)
                        grid = SE.bls_grid(tb, pmin, pmax)
                        pw = gpu_bls.bls_power(tb, fb, grid, SE.BLS_DURATIONS, device=a.device)
                        torch.cuda.synchronize(a.device)
                        rec.update({"bls_s": time.time() - t2, "P_seed": float(grid[int(np.nanargmax(pw))]), "n_periods": int(grid.size)})
                    except Exception as e:
                        rec["error"] = repr(e)[:200]
                fh.write(json.dumps(rec) + "\n")
                if k % 500 == 0:
                    print(f"{k}/{len(mine)} {time.time() - t_start:.0f}s", flush=True)
        else:
            seeds = {}
            for p in glob.glob("/kaggle/input/**/b4_bls_*.jsonl.gz", recursive=True):
                for r in read_records(p):
                    if "P_seed" in r:
                        seeds[r["tic"]] = r["P_seed"]
            jobs = [(t, files[t], seeds.get(t, np.nan)) for t in mine]
            with ProcessPoolExecutor(a.workers) as ex:
                for k, rec in enumerate(ex.map(resolver_one, jobs, chunksize=4)):
                    fh.write(json.dumps(rec, default=float) + "\n")
                    if k % 500 == 0:
                        print(f"{k}/{len(jobs)} {time.time() - t_start:.0f}s", flush=True)
    (out / f"done_{tag}.json").write_text(json.dumps({"git_commit": a.commit, "stage": a.stage, "n": len(mine), "wall_s": time.time() - t_start,
                                                       "hardware": hardware(a.device if a.stage == "bls" else None),
                                                       "workers": 1 if a.stage == "bls" else a.workers, "caches": [str(c) for c in caches()]}))


if __name__ == "__main__":
    main()
