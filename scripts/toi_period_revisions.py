"""POST HOC, descriptive (decision Rev-1): how often has a TOI period been revised by an alias factor?
Compares the period of each TOI in the published Prime Mission catalog (Guerrero et al. 2021, table 2, release
2020-10-18; frozen snapshot) with the period of the same TOI in the frozen ExoFOP table of 2026-09-30.
ratio = P_now / P_then. "unchanged": |ratio - 1| < 1 %. "alias": within 1 % of a factor in the resolver's alias set
other than 1. "other": anything else. Current TFOPWG dispositions and membership of the star in the TESS EB catalogs
(Prsa et al. 2022; Kostov et al. 2025) are tabulated for the alias-revised TOIs.
Output: results/toi_period_revisions.json and results/toi_period_revisions.csv (the alias-revised TOIs)."""

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from b2_catalog_seeds import guerrero_table  # noqa: E402
from b2_score import wilson  # noqa: E402
from independent_period_check import kostov  # noqa: E402

ALIASES = [Fraction(1, 5), Fraction(1, 4), Fraction(1, 3), Fraction(1, 2), Fraction(2, 3), Fraction(3, 2), Fraction(2), Fraction(3), Fraction(4), Fraction(5)]


def label(r):
    if not np.isfinite(r) or r <= 0:
        return "no period"
    if abs(r - 1) < 0.01:
        return "unchanged"
    for a in ALIASES:
        if abs(r / float(a) - 1) < 0.01:
            return str(a)
    return "other"


def main():
    g = guerrero_table().reset_index()
    toi = pd.read_csv(ROOT / "data/raw/exofop_toi_20260930.csv", comment="#", low_memory=False)
    toi["toi"] = toi["TOI"].map(lambda x: f"{float(x):.2f}")
    m = g.merge(toi[["toi", "TIC ID", "TFOPWG Disposition", "Period (days)"]], on="toi", how="left")
    m["P_then"], m["P_now"] = m["per"], pd.to_numeric(m["Period (days)"], errors="coerce")
    both = m[(m["P_then"] > 0) & (m["P_now"] > 0)].copy()
    both["ratio"] = both["P_now"] / both["P_then"]
    both["change"] = both["ratio"].map(label)
    eb = set(pd.read_csv(ROOT / "data/raw/tess_ebs_prsa2022_v1.0_20260930.csv")["tess_id"]) | set(kostov().index)
    both["in_eb_catalog"] = both["TIC ID"].isin(eb)
    both["disp"] = both["TFOPWG Disposition"].fillna("none")
    al = both[~both["change"].isin(["unchanged", "other"])]
    n = len(both)

    def disp(df):
        return {"n": int(len(df)), "FP_or_FA": int(df["disp"].isin(["FP", "FA"]).sum()), "CP_or_KP": int(df["disp"].isin(["CP", "KP"]).sum()),
                "PC_APC_or_none": int(df["disp"].isin(["PC", "APC", "none"]).sum()), "in_eb_catalog": int(df["in_eb_catalog"].sum()),
                "FP_FA_or_in_eb_catalog": int((df["disp"].isin(["FP", "FA"]) | df["in_eb_catalog"]).sum())}

    rep = {"label": "POST HOC, descriptive (decision Rev-1)", "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "historical_catalog": "Guerrero et al. (2021) table 2, release 2020-10-18", "current_catalog": "ExoFOP TOI table, 2026-09-30",
           "tois_in_historical_catalog": int(len(g)), "with_period_in_both": int(n),
           "unchanged": int((both["change"] == "unchanged").sum()), "other_change": int((both["change"] == "other").sum()),
           "alias_revised": {"count": int(len(al)), "fraction": len(al) / n, "ci95": wilson(int(len(al)), int(n))},
           "alias_revised_by_factor": {k: int(v) for k, v in al["change"].value_counts().items()},
           "period_doubled_or_more": int((al["ratio"] > 1.2).sum()), "period_halved_or_less": int((al["ratio"] < 0.8).sum()),
           "alias_revised_dispositions": disp(al), "unchanged_dispositions": disp(both[both["change"] == "unchanged"]),
           "alias_revised_by_pipeline": {k: int(v) for k, v in al["pipeline"].value_counts().items()}}
    al[["toi", "TIC ID", "pipeline", "P_then", "P_now", "ratio", "change", "disp", "in_eb_catalog"]].to_csv(ROOT / "results/toi_period_revisions.csv", index=False, float_format="%.8g")
    (ROOT / "results/toi_period_revisions.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
