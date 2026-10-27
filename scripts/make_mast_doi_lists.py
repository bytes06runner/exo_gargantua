"""Observation lists for the MAST DOI (data editor item II). Lists every SPOC 120-s light curve actually loaded by a
benchmark (not every pinned file): B1 injection sectors, B2 early windows, B5 early windows and audit later sectors,
B4 Sector 49. MAST obs_id = product filename without "_lc.fits".
Outputs (paper/data/mast/):
  all_light_curves_used_obs_ids.csv        every light curve analysed (for a bulk DOI request to MAST)
  collection_DOI_obs_ids_released_tables.csv  light curves behind the two released per-target tables (B2, B5 audit)
  collection_DOI_TIC_list.csv              the TIC IDs of that subset (for the Portal's target-list upload)
  summary.json                              counts and sector range (read by make_paper_assets.py)
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "data" / "mast"


def rows(df, role):
    d = df[["tic", "sector", "filename"]].copy()
    d["obs_id"] = d["filename"].str.replace("_lc.fits", "", regex=False)
    d["used_in"] = role
    return d


def pairs(df, col):
    return {(int(r.tic), int(s)) for r in df.itertuples() for s in str(getattr(r, col)).split(";")}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pin = pd.read_csv(ROOT / "data" / "pinned_products.csv")
    e5 = pd.read_csv(ROOT / "data" / "b5_pinned_products.csv")
    a5 = pd.read_csv(ROOT / "data" / "b5_pins_audit.csv")
    b4 = pd.read_csv(ROOT / "data" / "b4_pins.csv")
    b2 = pd.read_csv(ROOT / "paper" / "data" / "b2_per_target.csv")
    inj = pd.read_csv(ROOT / "data" / "b1_injections.csv")
    key = list(zip(pin["tic"], pin["sector"]))

    early = pairs(b2, "sectors_early")
    used_b1 = pairs(inj, "sectors")
    B2 = rows(pin[[k in early for k in key]], "B2 early window")
    B1 = rows(pin[[k in used_b1 for k in key]], "B1 injection hosts")
    B5 = pd.concat([rows(e5[e5["role"] == "b5_early"], "B5 early window"), rows(a5, "B5 later sectors (audit)")])
    t5 = set(pd.read_csv(ROOT / "paper" / "data" / "b5_candidate_period_revisions.csv")["tic"])
    sub = pd.concat([B2, B5[B5["tic"].isin(t5)]]).drop_duplicates("obs_id")
    allu = pd.concat([B2, B5, B1, rows(b4, "B4 Sector 49 runtime")]).drop_duplicates("obs_id")
    assert len(B2) == len(early), "every B2 early-window file must be pinned"

    allu.to_csv(OUT / "all_light_curves_used_obs_ids.csv", index=False)
    sub.to_csv(OUT / "collection_DOI_obs_ids_released_tables.csv", index=False)
    sub.drop_duplicates("tic")[["tic"]].to_csv(OUT / "collection_DOI_TIC_list.csv", index=False)
    summ = {"all_files": int(len(allu)), "all_stars": int(allu["tic"].nunique()),
            "sector_min": int(allu["sector"].min()), "sector_max": int(allu["sector"].max()),
            "by_benchmark": {k: int(v) for k, v in allu["used_in"].value_counts().items()},
            "subset_files": int(len(sub)), "subset_stars": int(sub["tic"].nunique())}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
