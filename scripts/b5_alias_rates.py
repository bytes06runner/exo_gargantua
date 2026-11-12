"""POST HOC, descriptive (decision Alias-1): where the raw search peaks fall relative to the catalog period on the B5
primary sample. Output: results/b5_alias_rates.json."""

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from b2_score import wilson  # noqa: E402

ALIASES = ["1/5", "1/4", "1/3", "1/2", "2/3", "3/2", "2", "3", "4", "5"]


def label(P, Pc):
    if not np.isfinite(P):
        return "no period"
    r = P / Pc
    if abs(r - 1) < 1e-3:
        return "correct"
    if abs(r - 1) < 0.01:
        return "within 1% (not an alias)"
    for a in ALIASES:
        if abs(r / float(Fraction(a)) - 1) < 0.01:
            return a
    return "unrelated"


def main():
    d = pd.read_csv(ROOT / "results/b5_per_target.csv")
    d = d[d["primary_sample"]]
    rep = {"label": "POST HOC, descriptive (decision Alias-1)", "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(), "n": int(len(d))}
    for name, col in (("raw_tls", "P_tls"), ("raw_bls", "P_bls"), ("resolver", "P_resolver")):
        lab = pd.Series([label(P, Pc) for P, Pc in zip(d[col], d["P_catalog"])])
        al = lab.isin(ALIASES)
        rep[name] = {"counts": {k: int(v) for k, v in lab.value_counts().items()},
                     "at_an_alias": {"count": int(al.sum()), "fraction": float(al.mean()), "ci95": wilson(int(al.sum()), len(d))},
                     "at_half": {"count": int((lab == "1/2").sum()), "fraction": float((lab == "1/2").mean())},
                     "at_double": {"count": int((lab == "2").sum()), "fraction": float((lab == "2").mean())},
                     "at_a_multiple": {"count": int(lab.isin(["2", "3", "4", "5"]).sum()), "fraction": float(lab.isin(["2", "3", "4", "5"]).mean())},
                     "share_of_errors_at_an_alias": float(al.sum() / max((lab != "correct").sum(), 1))}
    (ROOT / "results/b5_alias_rates.json").write_text(json.dumps(rep, indent=1, default=float))
    for k in ("raw_tls", "raw_bls", "resolver"):
        print(k, rep[k]["counts"], round(rep[k]["at_an_alias"]["fraction"], 3), round(rep[k]["share_of_errors_at_an_alias"], 3))


if __name__ == "__main__":
    main()
