"""B5 (EB alias benchmark) sample selection from the frozen 2026-09-30 snapshots only (docs/b5_definition.md).
No light curve is downloaded. Writes data/b5_sample_log.csv (every catalog EB with decision and reason) and
data/b5_pinned_products.csv (pinned SPOC 120-s files of the included EBs), and prints the counts.

Rules (codes logged):
  Y01  the TIC has SPOC 120-s light-curve products ("-s_lc.fits") in >= 2 sectors in S-LC
  Y02  catalog period finite and 0.5 <= P <= 30 d, and bjd0 finite
  Y03  exactly one catalog signal on the TIC (several signals: excluded, truth ambiguous)
  Y04  the TIC is not a B2 host (sample_log b2_eligible == pending_H3) and not an injection-pool star
       (data/injection_pool_log.csv decision == include)
  Y05  early window E (sectors starting < first + 365.25 d) and later sectors L (rest) both non-empty (= B2 H2)
Truth tier: B, or B-amb by the pre-registered proxy of docs/sample_definition.md §3.2 (sample.eb_period_ambiguous);
B-amb is reported separately and never in the primary metric.
Then (if needed for cost, docs/b5_definition.md) a pre-seeded random subset numpy.random.default_rng(20261021).
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exogargantua import sample as S  # noqa: E402

N_MAX = 1500          # cap fixed in docs/b5_definition.md before any data (cost)
SEED = 20261021


def main():
    snaps = json.loads((ROOT / "data" / "raw" / "snapshots.json").read_text())
    eb = pd.read_csv(ROOT / snaps["S-EB"]["file"])
    lc = pd.read_csv(ROOT / "data" / "raw" / "mast_lc_products_20260930.csv.gz")
    lc = lc[lc["filename"].str.endswith("-s_lc.fits")]
    starts = S.sector_start_btjd(pd.read_csv(ROOT / snaps["S-ORB"]["file"]))
    log = pd.read_csv(ROOT / "data" / "sample_log.csv", dtype={"toi": str})
    b2_tics = set(log.loc[log["b2_eligible"] == "pending_H3", "tic"].astype(int))
    pool = pd.read_csv(ROOT / "data" / "injection_pool_log.csv")
    pool_tics = set(pool.loc[pool["decision"] == "include", "tic"].astype(int))
    secs = lc.groupby("tic")["sector"].apply(lambda s: sorted(set(int(x) for x in s)))
    n_sig = eb.groupby("tess_id").size()
    rows = []
    for _, r in eb.iterrows():
        tic = int(r["tess_id"])
        rec = {"tic": tic, "signal_id": int(r["signal_id"]), "period": r["period"], "bjd0": r["bjd0"]}
        s = secs.get(tic, [])
        P = r["period"]
        if len(s) < 2:
            reason = f"Y01: {len(s)} SPOC 120-s sector(s)"
        elif not (np.isfinite(P) and 0.5 <= P <= 30 and np.isfinite(r["bjd0"])):
            reason = "Y02: period outside 0.5-30 d or missing ephemeris"
        elif n_sig[tic] > 1:
            reason = f"Y03: {n_sig[tic]} catalog signals on the TIC"
        elif tic in b2_tics or tic in pool_tics:
            reason = "Y04: B2 host" if tic in b2_tics else "Y04: injection-pool star"
        else:
            early, later = S.early_later_split(s, starts)
            reason = "Y05: no later sectors" if not later else ""
            rec.update({"sectors_all": ";".join(map(str, s)), "sectors_early": ";".join(map(str, early)),
                        "sectors_later": ";".join(map(str, later))})
        rec["tier"] = ("B-amb" if S.eb_period_ambiguous(r) else "B") if not reason else ""
        rec["decision"], rec["reason"] = ("candidate", "") if not reason else ("exclude", reason)
        rows.append(rec)
    d = pd.DataFrame(rows)
    cand = d[d["decision"] == "candidate"].sort_values("tic")
    sel = cand
    if len(cand) > N_MAX:
        pick = np.sort(np.random.default_rng(SEED).choice(cand["tic"].to_numpy(), N_MAX, replace=False))
        d.loc[d["decision"].eq("candidate") & ~d["tic"].isin(pick), ["decision", "reason"]] = ["exclude", "S01: not in the pre-seeded subset"]
        sel = cand[cand["tic"].isin(pick)]
    d.loc[d["decision"] == "candidate", "decision"] = "include"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    d["git_commit"] = commit
    d.to_csv(ROOT / "data" / "b5_sample_log.csv", index=False)
    pins = lc[lc["tic"].isin(sel["tic"])][["tic", "sector", "filename"]].sort_values(["tic", "sector"])
    pins.to_csv(ROOT / "data" / "b5_pinned_products.csv", index=False)
    rep = {"git_commit": commit, "catalog_rows": int(len(d)), "reasons": d["reason"].replace("", "included").str.split(":").str[0].value_counts().to_dict(),
           "candidates_before_cap": int(len(cand)), "included": int(len(sel)), "tier_included": sel["tier"].value_counts().to_dict(),
           "pinned_files": int(len(pins)), "early_sectors_median": float(sel["sectors_early"].str.count(";").add(1).median()) if len(sel) else None}
    print(json.dumps(rep, indent=1))
    (ROOT / "results" / "b5_selection.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
