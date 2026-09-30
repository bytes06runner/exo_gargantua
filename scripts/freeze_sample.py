"""Freeze the Phase 2 sample (docs/sample_definition.md). Runs on Kaggle (CPU, internet on).

Inputs: the committed catalog snapshots in data/raw/ (snapshots.json).
Network: MAST 2-min LC bulk scripts (sectors 1-106), MAST SPOC tcestats files, MAST TIC,
Gaia DR3 archive.
Outputs (to --out):
  sample_log.csv            one row per TOI in the snapshot, with decision/reason codes
  pinned_products.csv       one row per pinned LC file (TOI hosts + injection pool)
  injection_pool_log.csv    every injection-pool star examined, with decision/reason codes
  stellar_params.csv        TIC v8.2 (+ Gaia DR3 RUWE/variability) for all pinned stars
  spoc_tces_toi_hosts.csv   SPOC TCEs (all runs) on TOI hosts, for the B2 SPOC baseline
  freeze_manifest.json      counts, listing dates, runtimes, commit, config
  raw/                      the raw MAST scripts and tcestats files (gzip), for the Kaggle dataset
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import io
import json
import os
import platform
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exogargantua import sample as S  # noqa: E402

LC_PAGE = "https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_ffi-tp-lc-dv.html"
TCE_PAGE = "https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_tce.html"
MAST = "https://archive.stsci.edu"
SEED = 20260930
POOL_PER_STRATUM = 750
STRATA = [(2, 3), (4, 6), (7, 12), (13, 10_000)]
UA = {"User-Agent": "exogargantua-freeze (mailto:cybrobasics@gmail.com)"}
TCE_COLS = ["ticid", "tce_plnt_num", "sectors", "tce_period", "tce_period_err", "tce_time0bt", "tce_time0bt_err",
            "tce_duration", "tce_depth", "tce_model_snr", "tce_max_mult_ev", "tce_num_transits", "tce_impact"]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def get(url, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
                return r.read(), dict(r.headers)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


# ------------------------------------------------------------------ stage 1: MAST LC products
def mast_products(out: Path, max_sector: int) -> tuple[pd.DataFrame, dict]:
    page = get(LC_PAGE)[0].decode()
    links = sorted(set(re.findall(r'/missions/tess/download_scripts/sector/tesscurl_sector_(\d+)_lc\.sh', page)), key=int)
    sectors = [int(s) for s in links if S.SECTOR_RANGE[0] <= int(s) <= min(S.SECTOR_RANGE[1], max_sector)]
    raw = out / "raw" / "mast_lc_scripts"
    raw.mkdir(parents=True, exist_ok=True)

    def one(sec):
        url = f"{MAST}/missions/tess/download_scripts/sector/tesscurl_sector_{sec}_lc.sh"
        body, hdr = get(url)
        (raw / f"tesscurl_sector_{sec}_lc.sh.gz").write_bytes(gzip.compress(body, mtime=0))
        df = S.parse_bulk_script(body.decode())
        df["bulk_script"] = url
        df["bulk_script_last_modified"] = hdr.get("Last-Modified", "")
        return sec, df, hdr.get("Last-Modified", "")

    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, sectors))
    prods = pd.concat([r[1] for r in res], ignore_index=True)
    info = {"sectors_listed": sectors, "listing_utc": now(),
            "script_last_modified": {r[0]: r[2] for r in res}, "n_products": int(len(prods))}
    return prods, info


# ------------------------------------------------------------------ stage 2: SPOC TCEs
def spoc_tces(out: Path) -> tuple[pd.DataFrame, dict]:
    page = get(TCE_PAGE)[0].decode()
    files = sorted(set(re.findall(r'/missions/tess/catalogs/tce/(tess\d+-s(\d{4})-s(\d{4})_dvr-tcestats\.csv)', page)))
    raw = out / "raw" / "spoc_tcestats"
    raw.mkdir(parents=True, exist_ok=True)

    def one(f):
        name, s0, s1 = f
        body, hdr = get(f"{MAST}/missions/tess/catalogs/tce/{name}")
        (raw / f"{name}.gz").write_bytes(gzip.compress(body, mtime=0))
        df = pd.read_csv(io.BytesIO(body), comment="#", low_memory=False)
        df = df[[c for c in TCE_COLS if c in df.columns]].copy()
        df["run"], df["run_s0"], df["run_s1"] = name, int(s0), int(s1)
        return df

    with ThreadPoolExecutor(6) as ex:
        tces = pd.concat(list(ex.map(one, files)), ignore_index=True)
    return tces, {"tce_files": [f[0] for f in files], "n_tce_rows": int(len(tces)), "listing_utc": now()}


# ------------------------------------------------------------------ stage 3: stellar parameters
def tic_params(tics: list[int]) -> pd.DataFrame:
    from astroquery.mast import Catalogs
    cols = ["ID", "Teff", "e_Teff", "logg", "e_logg", "Tmag", "rad", "e_rad", "mass", "e_mass", "rho", "e_rho",
            "MH", "GAIA", "objType", "lumclass"]
    out = []
    for i in range(0, len(tics), 500):
        chunk = [str(t) for t in tics[i:i + 500]]
        for k in range(5):
            try:
                t = Catalogs.query_criteria(catalog="Tic", ID=chunk).to_pandas()
                break
            except Exception:
                if k == 4:
                    raise
                time.sleep(10 * (k + 1))
        out.append(t[[c for c in cols if c in t.columns]])
    df = pd.concat(out, ignore_index=True).rename(columns={"ID": "tic"})
    df["tic"] = df["tic"].astype("int64")
    return df


def gaia_dr3(dr2_ids: list[int]) -> pd.DataFrame:
    """DR2 -> DR3 via gaiadr3.dr2_neighbourhood (closest match), then RUWE and variability flag."""
    from astroquery.gaia import Gaia
    rows = []
    ids = [int(x) for x in dr2_ids if pd.notna(x)]
    for i in range(0, len(ids), 1000):
        chunk = ",".join(str(x) for x in ids[i:i + 1000])
        q = f"""
        SELECT n.dr2_source_id, n.dr3_source_id, n.angular_distance, g.ruwe, g.phot_variable_flag
        FROM gaiadr3.dr2_neighbourhood AS n JOIN gaiadr3.gaia_source AS g ON g.source_id = n.dr3_source_id
        WHERE n.dr2_source_id IN ({chunk})"""
        for k in range(5):
            try:
                rows.append(Gaia.launch_job_async(q).get_results().to_pandas())
                break
            except Exception:
                if k == 4:
                    raise
                time.sleep(10 * (k + 1))
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["dr2_source_id", "dr3_source_id", "angular_distance", "ruwe", "phot_variable_flag"])
    df = df.sort_values("angular_distance").drop_duplicates("dr2_source_id")
    return df


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--max-sector", type=int, default=S.SECTOR_RANGE[1], help="debug only: list fewer sectors")
    ap.add_argument("--skip-tce", action="store_true", help="debug only")
    ap.add_argument("--skip-pool", action="store_true", help="debug only")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    timings = {}
    snaps = json.loads((ROOT / "data" / "raw" / "snapshots.json").read_text())
    snap_date = snaps["S-TOI"]["downloaded_utc"][:10]

    t = time.time(); prods, lc_info = mast_products(out, args.max_sector); timings["mast_lc_listing_s"] = time.time() - t
    prods.to_csv(out / f"mast_lc_products_{lc_info['listing_utc'][:10].replace('-', '')}.csv.gz", index=False)
    t = time.time()
    if args.skip_tce:
        tces, tce_info = pd.DataFrame(columns=TCE_COLS + ["run", "run_s0", "run_s1"]), {"skipped": True}
    else:
        tces, tce_info = spoc_tces(out)
    timings["spoc_tce_listing_s"] = time.time() - t
    orbit = pd.read_csv(ROOT / snaps["S-ORB"]["file"])
    starts = S.sector_start_btjd(orbit)

    sectors_by_tic = prods.groupby("tic")["sector"].apply(lambda s: sorted(set(s))).to_dict()
    files_by_tic = prods.sort_values("sector").groupby("tic")["filename"].apply(list).to_dict()

    # ---------------- TOI sample
    toi = S.normalise_toi(pd.read_csv(ROOT / snaps["S-TOI"]["file"], comment="#"))
    ps = pd.read_csv(ROOT / snaps["S-PS"]["file"], low_memory=False)
    ps["tic"] = pd.to_numeric(ps["tic_id"].astype(str).str.replace("TIC", "").str.strip(), errors="coerce")
    ps_by_tic = {int(k): g for k, g in ps.dropna(subset=["tic"]).groupby("tic")}
    eb = pd.read_csv(ROOT / snaps["S-EB"]["file"])
    eb_by_tic = {int(k): g for k, g in eb.groupby("tess_id")}
    dups = S.duplicate_tois(toi)

    log = []
    for _, r in toi.iterrows():
        row = dict(toi=f"{r['toi']:.2f}", tic=int(r["tic"]), decision="include", reason="", stage="freeze",
                   timestamp_utc=now(), git_commit=args.commit, snapshot_date=snap_date,
                   sectors_all="", sectors_early="", sectors_later="", product_files="",
                   lc_listing_date=lc_info["listing_utc"], data_release="", download_date="",
                   truth_tier="", truth_period="", truth_source="", b2_eligible="", b2_reason="")
        secs = sectors_by_tic.get(int(r["tic"]), [])
        if not (np.isfinite(r["period"]) and S.P_MIN <= r["period"] <= S.P_MAX):
            row.update(decision="exclude", reason="X02: period missing/zero or outside 0.5-30 d")
        elif r["toi"] in dups:
            row.update(decision="exclude", reason="X03: duplicate of a lower-numbered TOI")
        elif len(secs) < 2:
            row.update(decision="exclude", reason=f"X01: {len(secs)} SPOC 120-s sector(s) in S-LC")
        if row["decision"] == "include":
            early, later = S.early_later_split(secs, starts)
            files = files_by_tic[int(r["tic"])]
            tr = S.assign_truth(r, ps_by_tic.get(int(r["tic"]), pd.DataFrame()), eb_by_tic.get(int(r["tic"]), pd.DataFrame()))
            h1 = tr.tier in ("A", "B")
            h2 = len(later) > 0
            row.update(sectors_all=";".join(map(str, secs)), sectors_early=";".join(map(str, early)),
                       sectors_later=";".join(map(str, later)), product_files=";".join(files),
                       truth_tier=tr.tier, truth_period=("" if not np.isfinite(tr.period) else repr(tr.period)),
                       truth_source=tr.source or tr.note,
                       b2_eligible="pending_H3" if (h1 and h2) else "no",
                       b2_reason=("H3 evaluated at download" if (h1 and h2) else
                                  ";".join(c for c, ok in (("H1: no Tier A/B truth", h1), ("H2: no later sectors", h2)) if not ok)))
        log.append(row)
    sample_log = pd.DataFrame(log)
    sample_log.to_csv(out / "sample_log.csv", index=False)
    inc = sample_log[sample_log["decision"] == "include"]
    toi_host_tics = sorted(set(inc["tic"]))

    # ---------------- SPOC TCEs on TOI hosts (B2 baseline input)
    tces[tces["ticid"].isin(toi_host_tics)].to_csv(out / "spoc_tces_toi_hosts.csv.gz", index=False)

    # ---------------- injection pool
    t = time.time()
    ctoi = pd.read_csv(ROOT / snaps["S-CTOI"]["file"], comment="#")
    ctoi_tics = set(pd.to_numeric(ctoi["TIC ID"], errors="coerce").dropna().astype("int64"))
    eb2 = set()
    for key in ("S-EB2-new", "S-EB2-known"):
        for line in (ROOT / snaps[key]["file"]).read_text().splitlines():
            if line[:10].strip().isdigit():
                eb2.add(int(line[:10]))
    excluded_sets = {
        "P02": set(toi["tic"]) | ctoi_tics,
        "P03": set(tces["ticid"].astype("int64")),
        "P04": set(eb["tess_id"].astype("int64")) | eb2,
    }
    n_sec = {tic: len(s) for tic, s in sectors_by_tic.items()}
    cand = [tic for tic, n in n_sec.items() if n >= 2]  # P01
    counts = {"P01_candidates": len(cand)}
    for code, bad in excluded_sets.items():
        before = len(cand)
        cand = [c for c in cand if c not in bad]
        counts[f"{code}_removed"] = before - len(cand)
    rng = np.random.default_rng(SEED)
    pool_log = []
    for lo, hi in ([] if args.skip_pool else STRATA):
        stratum = np.array(sorted(c for c in cand if lo <= n_sec[c] <= hi), dtype="int64")
        rng.shuffle(stratum)
        accepted, i = 0, 0
        while accepted < POOL_PER_STRATUM and i < len(stratum):
            batch = stratum[i:i + 500].tolist()
            i += len(batch)
            tp = tic_params(batch).set_index("tic")
            tp["GAIA"] = pd.to_numeric(tp["GAIA"], errors="coerce")
            tp = tp[~tp.index.duplicated(keep=False)]
            gaia = gaia_dr3(tp["GAIA"].dropna().astype("int64").tolist()).set_index("dr2_source_id")
            for tic in batch:  # keep the random order
                rec = dict(tic=tic, stratum=f"{lo}-{hi if hi < 10_000 else 'inf'}", n_sectors=n_sec[tic],
                           decision="exclude", reason="", stage="freeze", timestamp_utc=now(), git_commit=args.commit)
                if accepted >= POOL_PER_STRATUM:
                    break
                p = tp.loc[tic] if tic in tp.index else None
                if p is None:
                    rec["reason"] = "P06: TIC record missing/duplicated"
                elif not (3200 <= p["Teff"] <= 7000 and p["logg"] >= 4.0 and 7 <= p["Tmag"] <= 13
                          and np.isfinite(p["rad"]) and np.isfinite(p["mass"])):
                    rec["reason"] = "P06: Teff/logg/Tmag/radius/mass outside rule"
                else:
                    g = gaia.loc[int(p["GAIA"])] if pd.notna(p["GAIA"]) and int(p["GAIA"]) in gaia.index else None
                    if g is None:
                        rec["reason"] = "P05: no Gaia DR3 match"
                    elif str(g["phot_variable_flag"]).upper() == "VARIABLE":
                        rec["reason"] = "P05: Gaia DR3 VARIABLE"
                    elif not (g["ruwe"] < 1.4):
                        rec["reason"] = "P05: RUWE >= 1.4 or missing"
                    else:
                        rec.update(decision="include", reason="P01-P06 pass (P07/P08 at download)")
                        accepted += 1
                pool_log.append(rec)
        counts[f"stratum_{lo}_{hi}_eligible_candidates"] = int(len(stratum))
        counts[f"stratum_{lo}_{hi}_accepted"] = accepted
    pool_log = pd.DataFrame(pool_log, columns=["tic", "stratum", "n_sectors", "decision", "reason", "stage",
                                               "timestamp_utc", "git_commit"])
    pool_log.to_csv(out / "injection_pool_log.csv", index=False)
    timings["injection_pool_s"] = time.time() - t
    pool_tics = pool_log.loc[pool_log["decision"] == "include", "tic"].astype("int64").tolist()

    # ---------------- stellar parameters (TOI hosts + pool) and pinned products
    t = time.time()
    stars = tic_params(toi_host_tics + [x for x in pool_tics if x not in set(toi_host_tics)])
    stars["GAIA"] = pd.to_numeric(stars["GAIA"], errors="coerce")
    g = gaia_dr3(stars["GAIA"].dropna().astype("int64").tolist())
    stars = stars.merge(g, how="left", left_on="GAIA", right_on="dr2_source_id")
    stars.to_csv(out / "stellar_params.csv", index=False)
    timings["stellar_params_s"] = time.time() - t

    role = {**{t_: "toi_host" for t_ in toi_host_tics}, **{t_: "pool" for t_ in pool_tics}}
    pinned = prods[prods["tic"].isin(role)].copy()
    pinned["role"] = pinned["tic"].map(role)
    pinned["url"] = pinned["filename"].map(S.product_url)
    for c in ("data_release", "md5", "download_date"):
        pinned[c] = ""
    pinned.sort_values(["tic", "sector"]).to_csv(out / "pinned_products.csv", index=False)

    manifest = {
        "git_commit": args.commit, "seed": SEED, "snapshot_date": snap_date, "finished_utc": now(),
        "runtime_s": round(time.time() - t_start, 1), "timings_s": {k: round(v, 1) for k, v in timings.items()},
        "host": {"cpu_count": os.cpu_count(), "platform": platform.platform(), "python": sys.version.split()[0],
                 "kaggle_kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "")},
        "mast_lc": {k: v for k, v in lc_info.items() if k != "script_last_modified"},
        "mast_lc_script_last_modified": lc_info["script_last_modified"],
        "spoc_tce": tce_info,
        "toi_counts": {
            "snapshot_rows": int(len(sample_log)),
            "included": int((sample_log["decision"] == "include").sum()),
            "by_reason": sample_log.loc[sample_log["decision"] == "exclude", "reason"].str[:3].value_counts().to_dict(),
            "truth_tier": inc["truth_tier"].replace("", "none").value_counts().to_dict(),
            "b2_pending_H3": int((inc["b2_eligible"] == "pending_H3").sum()),
        },
        "pool_counts": counts,
        "pinned_products": {"files": int(len(pinned)), "toi_host_files": int((pinned["role"] == "toi_host").sum()),
                            "pool_files": int((pinned["role"] == "pool").sum())},
    }
    (out / "freeze_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(json.dumps(manifest["toi_counts"], indent=1), json.dumps(counts, indent=1), manifest["runtime_s"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
