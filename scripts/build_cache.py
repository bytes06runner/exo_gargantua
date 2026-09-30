"""Build the compact light-curve cache in 4 chained chunks (decision D1; Kaggle CPU sessions, G2).

Chunk k (1..4) takes a contiguous block of TICs from data/pinned_products.csv (split at TIC
boundaries, so a star never spans two chunks), copies chunk k-1's cache through (attached via
kernel_sources), then for batches of 500 files: download -> verify/record MD5 and DATA_REL ->
compact (lightkurve `default` quality mask, PDCSAP, median-normalised; time and flux float64,
flux_err float32) -> delete raw -> append to the manifest (checkpoint after every batch).
A rerun skips files already in the manifest.

Output (/kaggle/working/cache):
  lc/<tic>.npz           arrays s<NNNN>_time, s<NNNN>_flux, s<NNNN>_flux_err per pinned sector kept
  manifest.csv           one row per pinned file (pins, header metadata, exclusions X10/X11)
  chunk_<k>_summary.json timings and counts
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
N_CHUNKS, BATCH = 4, 500
UA = {"User-Agent": "exogargantua-cache (mailto:cybrobasics@gmail.com)"}
MANIFEST_COLS = ["tic", "sector", "filename", "role", "md5", "data_release", "crowdsap", "flfrcsap", "cdpp1_0",
                 "n_cadences", "n_good", "good_frac", "raw_bytes", "excluded", "download_utc", "chunk"]


def say(m):
    print(f"[{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}] {m}", flush=True)


def chunk_files(pins: pd.DataFrame, k: int) -> pd.DataFrame:
    """Contiguous TIC blocks with ~equal file counts; deterministic."""
    counts = pins.groupby("tic").size().sort_index()
    target = counts.sum() / N_CHUNKS
    block = np.minimum((counts.cumsum() - 1) // target, N_CHUNKS - 1).astype(int) + 1
    tics = set(block[block == k].index)
    return pins[pins["tic"].isin(tics)].sort_values(["tic", "sector"])


def download(url, dest, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, open(dest, "wb") as fh:
                shutil.copyfileobj(r, fh)
            return True
        except Exception:
            time.sleep(5 * (i + 1))
    return False


def compact(path: str):
    """Read one pinned file -> (arrays, metadata). Runs in a worker process."""
    import warnings
    warnings.simplefilter("ignore")
    import lightkurve as lk
    from astropy.io import fits
    p = Path(path)
    h0, h1 = fits.getheader(p, 0), fits.getheader(p, 1)
    lc = lk.read(p, quality_bitmask="default", flux_column="pdcsap_flux")
    t = np.asarray(lc.time.value, np.float64)
    f = np.asarray(lc.flux.value, np.float64)
    e = np.asarray(lc.flux_err.value, np.float64)
    ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(e)
    med = np.nanmedian(f[ok]) if ok.any() else np.nan
    meta = {"md5": hashlib.md5(p.read_bytes()).hexdigest(), "data_release": h0.get("DATA_REL"),
            "crowdsap": h1.get("CROWDSAP"), "flfrcsap": h1.get("FLFRCSAP"), "cdpp1_0": h1.get("CDPP1_0"),
            "n_cadences": int(len(t)), "n_good": int(ok.sum()), "good_frac": float(ok.sum() / max(1, len(t))),
            "raw_bytes": p.stat().st_size}
    arrays = (t[ok], f[ok] / med, (e[ok] / med).astype(np.float32))
    return arrays, meta


def find_prev_cache():
    for d in sorted(Path("/kaggle/input").glob("*")) if Path("/kaggle/input").exists() else []:
        for m in d.rglob("manifest.csv"):
            return m.parent
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--limit-tics", type=int, default=0, help="debug only")
    args = ap.parse_args()
    t_start = time.time()
    out = Path(args.out)
    (out / "lc").mkdir(parents=True, exist_ok=True)
    raw = Path("/tmp/exog_raw") if Path("/tmp").exists() else out / "_raw"
    raw.mkdir(parents=True, exist_ok=True)

    prev = find_prev_cache()
    if prev is not None and prev.resolve() != out.resolve():
        t0 = time.time()
        shutil.copytree(prev / "lc", out / "lc", dirs_exist_ok=True)
        for f in prev.glob("*.json"):
            shutil.copy2(f, out / f.name)
        shutil.copy2(prev / "manifest.csv", out / "manifest.csv")
        say(f"copied previous cache from {prev} in {time.time() - t0:.0f}s")
    man_path = out / "manifest.csv"
    done = set(pd.read_csv(man_path)["filename"]) if man_path.exists() else set()

    pins = pd.read_csv(ROOT / "data" / "pinned_products.csv")
    todo = chunk_files(pins, args.chunk)
    if args.limit_tics:
        todo = todo[todo["tic"].isin(sorted(todo["tic"].unique())[:args.limit_tics])]
    todo = todo[~todo["filename"].isin(done)]
    say(f"chunk {args.chunk}: {todo['tic'].nunique()} TICs, {len(todo)} files to do ({len(done)} already in manifest)")

    stats = {"downloaded": 0, "x10": 0, "x11": 0, "download_s": 0.0, "compact_s": 0.0}
    with ProcessPoolExecutor(4) as pool:
        for b in range(0, len(todo), BATCH):
            batch = todo.iloc[b:b + BATCH]
            t0 = time.time()
            with ThreadPoolExecutor(8) as ex:
                ok = list(ex.map(lambda r: download(r.url, raw / r.filename), batch.itertuples()))
            stats["download_s"] += time.time() - t0
            t0 = time.time()
            got = [r for r, o in zip(batch.itertuples(), ok) if o]
            results = dict(zip([r.filename for r in got], pool.map(compact, [str(raw / r.filename) for r in got])))
            stats["compact_s"] += time.time() - t0
            rows, per_tic = [], {}
            now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
            for r, o in zip(batch.itertuples(), ok):
                row = {"tic": r.tic, "sector": r.sector, "filename": r.filename, "role": r.role, "excluded": "",
                       "download_utc": now, "chunk": args.chunk}
                if not o:
                    row["excluded"] = "X10: download failed after 3 retries"
                    stats["x10"] += 1
                else:
                    arrays, meta = results[r.filename]
                    row.update(meta)
                    stats["downloaded"] += 1
                    if meta["good_frac"] < 0.5:
                        row["excluded"] = "X11: < 50% finite PDCSAP cadences after default mask"
                        stats["x11"] += 1
                    else:
                        per_tic.setdefault(int(r.tic), {}).update({
                            f"s{int(r.sector):04d}_time": arrays[0], f"s{int(r.sector):04d}_flux": arrays[1],
                            f"s{int(r.sector):04d}_flux_err": arrays[2]})
                    (raw / r.filename).unlink(missing_ok=True)
                rows.append(row)
            for tic, arrs in per_tic.items():  # merge with sectors written by an earlier batch of this TIC
                npz = out / "lc" / f"{tic}.npz"
                if npz.exists():
                    with np.load(npz) as old:
                        arrs = {**{k: old[k] for k in old.files}, **arrs}
                np.savez_compressed(npz, **arrs)
            pd.DataFrame(rows, columns=MANIFEST_COLS).to_csv(man_path, mode="a", header=not man_path.exists(), index=False)
            say(f"  batch {b // BATCH + 1}/{-(-len(todo) // BATCH)}: {len(batch)} files, "
                f"dl {stats['download_s']:.0f}s, compact {stats['compact_s']:.0f}s total")

    summary = {"chunk": args.chunk, "git_commit": args.commit, "files_done_this_run": int(len(todo)), **stats,
               "wall_s": time.time() - t_start, "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "cache_bytes": int(sum(p.stat().st_size for p in (out / "lc").glob("*.npz"))),
               "n_tic_files": int(len(list((out / "lc").glob("*.npz")))), "cpu_count": os.cpu_count()}
    (out / f"chunk_{args.chunk}_summary.json").write_text(json.dumps(summary, indent=1))
    say(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
