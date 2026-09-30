"""Gate 2b checks on a Kaggle GPU (T4) session (decisions A1, A2(iv), G1).

1. Session limits probe: disk (working / scratch), CPU cores, RAM, GPUs.
2. A1: BLS on the 20 smoke targets, 4 targets at a time, on inputs rebuilt exactly as in the smoke
   run (same pinned files, same code path); peaks compared with the smoke run's serial peaks.
3. A2(iv): PyTorch BLS (src/exogargantua/gpu_bls.py) on the same inputs and grids; accepted only if
   every peak is within one period-grid step of astropy's. Hard cap: 2 GPU hours for this part.
Writes <out>/gate2b_results.json. Computes no accuracy against truth.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from exogargantua import gpu_bls, search as SE  # noqa: E402
from exogargantua import sample as S  # noqa: E402
import smoke_run as SR  # noqa: E402

GPU_CAP_S = 2 * 3600


def say(m):
    print(f"[{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}] {m}", flush=True)


def probe():
    info = {"cpu_count": os.cpu_count()}
    for p in ("/kaggle/working", "/kaggle/tmp", "/tmp", "/"):
        if os.path.exists(p):
            u = shutil.disk_usage(p)
            info[f"disk_{p}"] = {"total_gb": round(u.total / 1e9, 1), "free_gb": round(u.free / 1e9, 1)}
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    try:
        info["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                            capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    except Exception as exc:
        info["nvidia_smi"] = repr(exc)
    info["df_h"] = subprocess.run(["df", "-h"], capture_output=True, text=True).stdout
    return info


def rebuild_inputs(smoke, workdir):
    """Same pinned files and the same processing as scripts/smoke_run.py."""
    snaps = json.loads((ROOT / "data" / "raw" / "snapshots.json").read_text())
    toi_table = S.normalise_toi(pd.read_csv(ROOT / snaps["S-TOI"]["file"], comment="#"))
    pins = pd.read_csv(ROOT / "data" / "pinned_products.csv")
    out = []
    for r in smoke["targets"]:
        if "search" not in r:
            continue
        tic, toi = int(r["tic"]), float(r["toi"])
        early = set(int(s) for s in str(r["sectors_early"]).split(";"))
        good = {f["sector"] for f in r["files"] if "excluded" not in f}
        files = pins[(pins["tic"] == tic)].sort_values("sector")
        md5 = {f["sector"]: f.get("md5") for f in r["files"]}

        def fetch(fr):
            dest = workdir / fr.filename
            return fr, dest, SR.download(fr.url, dest)

        with ThreadPoolExecutor(8) as ex:
            got = list(ex.map(fetch, files.itertuples()))
        te, fe = [], []
        for fr, dest, ok in got:
            if not ok or int(fr.sector) not in good:
                continue
            import hashlib
            assert hashlib.md5(dest.read_bytes()).hexdigest() == md5[int(fr.sector)], f"md5 changed {fr.filename}"
            t_, f_, _, _ = SR.load_sector(dest)
            dest.unlink()
            if int(fr.sector) in early:
                te.append(t_); fe.append(f_)
        t = np.concatenate(te)
        fl = np.concatenate(fe)
        m = SR.other_toi_mask(t, tic, toi, toi_table)
        tb, fb = SE.prepare(t[m], fl[m])
        out.append({"toi": r["toi"], "tb": tb, "fb": fb, "serial": r["search"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path("/tmp/gate2b_raw")
    work.mkdir(exist_ok=True)
    res = {"git_commit": args.commit, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}

    res["probe"] = probe()
    say(f"probe: {json.dumps({k: v for k, v in res['probe'].items() if k != 'df_h'})}")

    smoke = json.loads(Path(args.smoke).read_text())
    t0 = time.time()
    inputs = rebuild_inputs(smoke, work)
    res["rebuild_s"] = time.time() - t0
    say(f"rebuilt {len(inputs)} targets in {res['rebuild_s']:.0f}s")
    for x in inputs:  # the rebuilt inputs must be the smoke run's inputs
        assert len(x["tb"]) == x["serial"]["n_binned"], x["toi"]

    # ---- A1: parallel astropy BLS, 4 targets at a time, largest first
    order = sorted(range(len(inputs)), key=lambda i: -len(inputs[i]["tb"]) * inputs[i]["serial"]["bls_n_periods"])
    t0 = time.time()
    par = SE.bls_peaks_parallel([(inputs[i]["tb"], inputs[i]["fb"]) for i in order], workers=4)
    res["bls_parallel"] = {"wall_s": time.time() - t0, "workers": 4,
                           "serial_wall_s_smoke": float(sum(x["serial"]["bls_s"] for x in inputs)), "targets": []}
    for i, p in zip(order, par):
        x = inputs[i]
        res["bls_parallel"]["targets"].append({
            "toi": x["toi"], "serial_bls_period": x["serial"]["bls_period"], "parallel_bls_period": p["bls_period"],
            "identical": p["bls_period"] == x["serial"]["bls_period"], "n_periods": p["bls_n_periods"],
            "parallel_worker_s": p["bls_s"], "serial_s": x["serial"]["bls_s"]})
    n_same = sum(t["identical"] for t in res["bls_parallel"]["targets"])
    say(f"A1 parallel BLS: {n_same}/{len(inputs)} identical, wall {res['bls_parallel']['wall_s']:.0f}s")

    # ---- A2(iv): GPU BLS vs astropy peaks, hard cap (torch/CUDA initialised only after the
    # forked BLS workers above have finished)
    import torch
    res["probe"]["torch"] = torch.__version__
    res["probe"]["cuda_available"] = torch.cuda.is_available()
    res["gpu_bls"] = {"cap_s": GPU_CAP_S, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                      "targets": []}
    t_gpu = time.time()
    for i in sorted(range(len(inputs)), key=lambda i: len(inputs[i]["tb"]) * inputs[i]["serial"]["bls_n_periods"]):
        if time.time() - t_gpu > GPU_CAP_S:
            res["gpu_bls"]["stopped_by_cap"] = True
            break
        x = inputs[i]
        pmin, pmax, _ = SE.period_limits(x["tb"])
        grid = SE.bls_grid(x["tb"], pmin, pmax)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        pw = gpu_bls.bls_power(x["tb"], x["fb"], grid, SE.BLS_DURATIONS)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        g_idx = int(np.argmax(pw))
        a_idx = int(np.argmin(np.abs(grid - x["serial"]["bls_period"])))
        res["gpu_bls"]["targets"].append({
            "toi": x["toi"], "n_periods": int(len(grid)), "gpu_s": time.time() - t0, "astropy_s": x["serial"]["bls_s"],
            "astropy_period": x["serial"]["bls_period"], "gpu_period": float(grid[g_idx]),
            "index_diff": abs(g_idx - a_idx), "within_one_step": abs(g_idx - a_idx) <= 1})
        say(f"GPU BLS {x['toi']}: {res['gpu_bls']['targets'][-1]['gpu_s']:.1f}s vs astropy {x['serial']['bls_s']:.0f}s, "
            f"index diff {abs(g_idx - a_idx)}")
    g = res["gpu_bls"]["targets"]
    res["gpu_bls"]["wall_s"] = time.time() - t_gpu
    res["gpu_bls"]["accepted"] = bool(len(g) == len(inputs) and all(t["within_one_step"] for t in g))
    say(f"A2(iv) GPU BLS accepted: {res['gpu_bls']['accepted']} ({sum(t['within_one_step'] for t in g)}/{len(inputs)})")

    res["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    (out / "gate2b_results.json").write_text(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
