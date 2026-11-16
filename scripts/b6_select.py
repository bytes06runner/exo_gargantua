"""B6 selection (docs/b6_definition.md): the replication sample is every EB that passed the B5 rules Y01-Y05 and was left out
of B5 only by the random cap S01 (data/b5_sample_log.csv, written at the B5 pre-registration, commit 4435520). No new rule and
no cap. Writes data/b6_sample_log.csv (same columns as the B5 log), data/b6_pinned_products.csv (early and later files from the
frozen MAST listing), data/b6_pins_early.csv and results/b6_selection.json. Reads no light curve and no run result."""

import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = "https://mast.stsci.edu/api/v0.1/Download/file/?uri=mast:TESS/product/"


def main():
    b5 = pd.read_csv(ROOT / "data/b5_sample_log.csv")
    d = b5[b5["reason"].fillna("").str.startswith("S01")].copy()
    assert len(d) == 1782 and d["tier"].isin(["B", "B-amb"]).all() and d["sectors_early"].notna().all() and d["sectors_later"].notna().all()
    assert not set(d["tic"]) & set(b5.loc[b5["decision"] == "include", "tic"]), "B6 must share no star with B5"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    d["decision"], d["reason"], d["git_commit"] = "include", "B6: passed Y01-Y05, excluded from B5 by the random cap S01", commit
    d = d.sort_values("tic")
    d.to_csv(ROOT / "data/b6_sample_log.csv", index=False)
    lc = pd.read_csv(ROOT / "data/raw/mast_lc_products_20260930.csv.gz")
    lc = lc[lc["filename"].str.endswith("-s_lc.fits")]
    role = {}
    for r in d.itertuples():
        for s in str(r.sectors_early).split(";"):
            role[(int(r.tic), int(s))] = "b6_early"
        for s in str(r.sectors_later).split(";"):
            role[(int(r.tic), int(s))] = "b6_later"
    pins = lc[lc["tic"].isin(d["tic"])][["tic", "sector", "filename"]].copy()
    pins["role"] = [role.get((int(t), int(s)), "") for t, s in zip(pins["tic"], pins["sector"])]
    pins = pins[pins["role"] != ""].sort_values(["tic", "sector", "filename"]).drop_duplicates(["tic", "sector"], keep="last")
    pins["url"] = URL + pins["filename"]
    assert len(pins) == len(role), (len(pins), len(role))
    pins.to_csv(ROOT / "data/b6_pinned_products.csv", index=False)
    pins[pins["role"] == "b6_early"].to_csv(ROOT / "data/b6_pins_early.csv", index=False)
    rep = {"git_commit": commit, "source_log_commit": sorted(b5["git_commit"].unique().tolist()), "included": int(len(d)),
           "tier_included": {k: int(v) for k, v in d["tier"].value_counts().items()},
           "pinned_files_early": int((pins["role"] == "b6_early").sum()), "pinned_files_later": int((pins["role"] == "b6_later").sum()),
           "early_sectors_median": float(d["sectors_early"].str.count(";").add(1).median())}
    (ROOT / "results/b6_selection.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
