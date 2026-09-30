"""Phase 2 smoke run (Kaggle, CPU): download + cache 20 frozen targets, run BLS and TLS on them.

Measures wall-clock per target, CPU/GPU usage and storage per target. It records the periods the
searches return, so the code path is exercised, but deliberately computes NO accuracy: benchmark
results are only looked at in Phase 4.

Reads the frozen files committed in data/ (sample_log.csv, pinned_products.csv,
stellar_params.csv). Writes to --out:
  cache/<tic>/s<NNNN>.npz   slim per-sector arrays (time, flux, flux_err) after the quality mask
  smoke_results.json        per-target timings, storage, file pins (md5, DATA_REL), search periods
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exogargantua import sample as S  # noqa: E402

SEED = 20260930
N_TARGETS = 20
UA = {"User-Agent": "exogargantua-smoke (mailto:cybrobasics@gmail.com)"}
BLS_DURATIONS = np.array([0.04, 0.06, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30])


class CpuSampler:
    """Machine-wide busy cores (psutil), sampled every 0.5 s while a block runs.

    On a dedicated Kaggle VM this measures what the search uses, including TLS's worker processes.
    """

    def __init__(self):
        import psutil
        self.psutil, self.samples, self._stop = psutil, [], threading.Event()

    def __enter__(self):
        self.psutil.cpu_percent(None)
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        return self

    def _run(self):
        n = self.psutil.cpu_count() or 1
        while not self._stop.wait(0.5):
            self.samples.append(self.psutil.cpu_percent(None) / 100.0 * n)

    def __exit__(self, *a):
        self._stop.set()
        self.t.join()

    @property
    def mean_cores(self):
        return float(np.mean(self.samples)) if self.samples else float("nan")


def download(url, dest, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, open(dest, "wb") as fh:
                shutil.copyfileobj(r, fh)
            return True
        except Exception:
            time.sleep(5 * (i + 1))
    return False


def load_sector(path):
    import lightkurve as lk
    from astropy.io import fits
    hdr0, hdr1 = fits.getheader(path, 0), fits.getheader(path, 1)
    lc = lk.read(path, quality_bitmask="default", flux_column="pdcsap_flux")
    t = np.asarray(lc.time.value, float)
    f = np.asarray(lc.flux.value, float)
    e = np.asarray(lc.flux_err.value, float)
    n_total = len(t)
    ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(e)
    med = np.nanmedian(f[ok])
    meta = {"sector": int(hdr0.get("SECTOR", -1)), "data_release": hdr0.get("DATA_REL"),
            "crowdsap": hdr1.get("CROWDSAP"), "flfrcsap": hdr1.get("FLFRCSAP"),
            "cdpp1_0": hdr1.get("CDPP1_0"), "n_cadences": n_total, "n_good": int(ok.sum()),
            "good_frac": float(ok.sum() / max(1, n_total))}
    return t[ok], (f[ok] / med), (e[ok] / med), meta


def search(time_, flux, n_threads, r_star, m_star, pmax_cap=30.0):
    from astropy.timeseries import BoxLeastSquares
    from transitleastsquares import transitleastsquares
    from wotan import flatten

    out = {}
    t0 = time.time()
    flat = flatten(time_, flux, method="biweight", window_length=0.75, break_tolerance=0.5)
    ok = np.isfinite(flat)
    tb, fb = bin10(time_[ok], flat[ok])
    out["detrend_bin_s"] = time.time() - t0
    span = tb.max() - tb.min()
    pmin, pmax = 0.5, min(pmax_cap, 0.95 * span)
    out["n_binned"], out["span_d"], out["pmax"] = int(tb.size), float(span), float(pmax)

    t0 = time.time()
    with CpuSampler() as cs:
        bls = BoxLeastSquares(tb, fb)
        periods = bls.autoperiod(BLS_DURATIONS, minimum_period=pmin, maximum_period=pmax,
                                 minimum_n_transit=2, frequency_factor=1.0)
        res = bls.power(periods, BLS_DURATIONS, objective="likelihood")
    out["bls_s"], out["bls_cpu_cores_mean"] = time.time() - t0, cs.mean_cores
    out["bls_n_periods"] = int(len(periods))
    out["bls_period"] = float(res.period[np.argmax(res.power)])

    t0 = time.time()
    with CpuSampler() as cs:
        kw = dict(period_min=pmin, period_max=pmax, use_threads=n_threads, show_progress_bar=False)
        if np.isfinite(r_star) and np.isfinite(m_star):
            kw.update(R_star=r_star, M_star=m_star, R_star_min=0.1, R_star_max=max(3.0, 1.5 * r_star),
                      M_star_min=0.1, M_star_max=max(2.5, 1.5 * m_star))
        tls = transitleastsquares(tb, fb).power(**kw)
    out["tls_s"], out["tls_cpu_cores_mean"] = time.time() - t0, cs.mean_cores
    out["tls_period"], out["tls_sde"] = float(tls.period), float(tls.SDE)
    out["tls_n_periods"] = int(len(tls.periods))
    return out


def bin10(t, f, width=10.0 / 1440.0):
    idx = np.floor((t - t.min()) / width).astype(np.int64)
    counts = np.bincount(idx)
    keep = counts > 0
    tb = (np.bincount(idx, weights=t) / np.maximum(counts, 1))[keep]
    fb = (np.bincount(idx, weights=f) / np.maximum(counts, 1))[keep]
    return tb, fb


def other_toi_mask(t, tic, this_toi, toi_table):
    m = np.ones(t.size, bool)
    for _, o in toi_table[(toi_table["tic"] == tic) & (toi_table["toi"] != this_toi)].iterrows():
        if np.isfinite(o["period"]) and o["period"] > 0 and np.isfinite(o["epoch_btjd"]):
            half = max(o["duration_h"] / 24.0, 0.05) if np.isfinite(o["duration_h"]) else 0.1
            ph = (t - o["epoch_btjd"] + 0.5 * o["period"]) % o["period"] - 0.5 * o["period"]
            m &= np.abs(ph) > half  # +-1 duration
    return m


def gpu_info():
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20).stdout.strip() or "none"
    except Exception:
        return "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", required=True)
    args = ap.parse_args()
    out, cache = Path(args.out), Path(args.out) / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    rawdir = Path("/tmp/exog_raw")
    rawdir.mkdir(exist_ok=True)

    log = pd.read_csv(ROOT / "data" / "sample_log.csv", dtype={"toi": str})
    pins = pd.read_csv(ROOT / "data" / "pinned_products.csv")
    stars = pd.read_csv(ROOT / "data" / "stellar_params.csv").drop_duplicates("tic").set_index("tic")
    snaps = json.loads((ROOT / "data" / "raw" / "snapshots.json").read_text())
    toi_table = S.normalise_toi(pd.read_csv(ROOT / snaps["S-TOI"]["file"], comment="#"))

    elig = log[(log["decision"] == "include") & (log["b2_eligible"] == "pending_H3")].sort_values("toi")
    pick = elig.sample(N_TARGETS, random_state=SEED)
    n_threads = os.cpu_count() or 1
    rows = []
    t_all = time.time()
    for _, r in pick.iterrows():
        tic, toi = int(r["tic"]), float(r["toi"])
        rec = {"toi": r["toi"], "tic": tic, "truth_tier": r["truth_tier"], "sectors_all": r["sectors_all"],
               "sectors_early": r["sectors_early"], "files": []}
        t_target = time.time()
        files = pins[pins["tic"] == tic].sort_values("sector")
        t0 = time.time()

        def fetch(fr):
            dest = rawdir / fr.filename
            ok = download(fr.url, dest)
            return fr, dest, ok

        with ThreadPoolExecutor(8) as ex:
            got = list(ex.map(fetch, files.itertuples()))
        rec["download_s"] = time.time() - t0
        early = set(int(s) for s in str(r["sectors_early"]).split(";"))
        te, fe, ee = [], [], []
        t0 = time.time()
        for fr, dest, ok in got:
            f = {"sector": int(fr.sector), "filename": fr.filename}
            if not ok:
                f["excluded"] = "X10: download failed after 3 retries"
                rec["files"].append(f)
                continue
            raw_bytes = dest.stat().st_size
            f["md5"] = hashlib.md5(dest.read_bytes()).hexdigest()
            t_, f_, e_, meta = load_sector(dest)
            f.update(meta, raw_bytes=raw_bytes)
            if meta["good_frac"] < 0.5:
                f["excluded"] = "X11: < 50% finite PDCSAP cadences after default mask"
            else:
                npz = cache / str(tic) / f"s{int(fr.sector):04d}.npz"
                npz.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(npz, time=t_, flux=f_.astype(np.float32), flux_err=e_.astype(np.float32))
                f["slim_bytes"] = npz.stat().st_size
                if int(fr.sector) in early:
                    te.append(t_); fe.append(f_); ee.append(e_)
            dest.unlink()
            rec["files"].append(f)
        rec["read_cache_s"] = time.time() - t0
        rec["raw_bytes"] = int(sum(f.get("raw_bytes", 0) for f in rec["files"]))
        rec["slim_bytes"] = int(sum(f.get("slim_bytes", 0) for f in rec["files"]))
        kept = [f for f in rec["files"] if "excluded" not in f]
        if len(kept) < 2:
            rec["excluded"] = "X12: fewer than 2 sectors after X10/X11"
            rows.append(rec)
            continue
        t = np.concatenate(te) if te else np.array([])
        fl = np.concatenate(fe) if fe else np.array([])
        # H3 (truth ephemeris coverage in early data); logged only, no accuracy computed
        trow = toi_table[toi_table["toi"] == toi].iloc[0]
        rec["h3_transits_in_early"] = S.n_covered_transits(t, float(r["truth_period"]), float(trow["epoch_btjd"]),
                                                          float(trow["duration_h"]) / 24.0)
        rec["b2_eligible"] = "yes" if rec["h3_transits_in_early"] >= 2 else "no (H3)"
        m = other_toi_mask(t, tic, toi, toi_table)
        rs = float(stars.loc[tic, "rad"]) if tic in stars.index else np.nan
        ms = float(stars.loc[tic, "mass"]) if tic in stars.index else np.nan
        rec["search"] = search(t[m], fl[m], n_threads, rs, ms)
        rec["target_wall_s"] = time.time() - t_target
        rows.append(rec)
        print(f"TOI {r['toi']}: {rec['target_wall_s']:.0f}s  dl={rec['download_s']:.0f}s "
              f"bls={rec['search']['bls_s']:.0f}s tls={rec['search']['tls_s']:.0f}s "
              f"raw={rec['raw_bytes']/1e6:.1f}MB slim={rec['slim_bytes']/1e6:.1f}MB", flush=True)

    import importlib.metadata as md
    env = {"cpu_count": os.cpu_count(), "gpu": gpu_info(), "platform": platform.platform(),
           "python": sys.version.split()[0],
           "versions": {p: md.version(p) for p in ("numpy", "scipy", "astropy", "lightkurve", "wotan",
                                                    "transitleastsquares", "pandas")},
           "kaggle_kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "")}
    try:
        import psutil
        env["memory_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    res = {"git_commit": args.commit, "seed": SEED, "n_targets": N_TARGETS,
           "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "total_wall_s": time.time() - t_all, "env": env,
           "config": {"detrend": "wotan biweight 0.75 d, break_tolerance 0.5 d", "bin_minutes": 10,
                      "bls_durations_d": BLS_DURATIONS.tolist(), "tls": "defaults, TIC R*/M*"},
           "targets": rows}
    (out / "smoke_results.json").write_text(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
