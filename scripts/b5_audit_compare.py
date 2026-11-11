"""Decision B5-r1.2: compare the catalog-truth audit of the first B5 run (results/kaggle/b5_audit, run at commit
7689b8c) with the audit of the rerun (results/kaggle/b5r_audit, commit 94b3abe). The audit code is unchanged apart
from the run it reads T_seed from; the target lists differ because the rerun's resolver answers differ.
Output: results/b5_audit_compare.json."""

import json
from math import pi
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V = ["later data support v2", "later data support the catalog", "inconclusive", "insufficient"]


def counts(d):
    v = d["verdict"].where(d["verdict"].isin(V), "insufficient")
    return {k: int((v == k).sum()) for k in V} | {"n": int(len(d))}


def main():
    o = pd.read_csv(ROOT / "results/kaggle/b5_audit/b5audit/b5_audit.csv")
    n = pd.read_csv(ROOT / "results/kaggle/b5r_audit/b5audit/b5_audit.csv")
    done_o = json.loads((ROOT / "results/kaggle/b5_audit/b5audit/b5_audit_done.json").read_text())
    done_n = json.loads((ROOT / "results/kaggle/b5r_audit/b5audit/b5_audit_done.json").read_text())
    m = o.merge(n, on="tic", suffixes=("_o", "_n"))
    same = np.abs(m["P_v2_o"] / m["P_v2_n"] - 1) < 1e-3
    s = m[same]
    st = pd.read_csv(ROOT / "data/b5_stellar_params.csv").set_index("tic")
    new_only = n[~n["tic"].isin(o["tic"])].copy()
    rad, mass = new_only["tic"].map(st["rad"]), new_only["tic"].map(st["mass"])
    floor = 2 * pi * np.sqrt((3 * rad * 6.957e8) ** 3 / (6.674e-11 * mass * 1.989e30)) / 86400
    cat_new = new_only["verdict"] == "later data support the catalog"
    rep = {"first_run": {"audit_commit": done_o["git_commit"], **counts(o)}, "rerun": {"audit_commit": done_n["git_commit"], **counts(n)},
           "code_change": "scripts/b5_audit.py reads T_seed from results/kaggle/b5r instead of results/kaggle/b5; fit model, thresholds and verdict rule unchanged; frozen resolver unchanged",
           "in_both": int(len(m)), "in_both_same_resolver_period": int(same.sum()),
           "same_period_same_verdict": int((s["verdict_o"] == s["verdict_n"]).sum()),
           "in_both_different_resolver_period": int((~same).sum()),
           "only_first_run": counts(o[~o["tic"].isin(n["tic"])]), "only_rerun": counts(new_only),
           "only_rerun_catalog_favored_below_tls_floor": int((cat_new & (new_only["P_catalog"] < floor)).sum())}
    (ROOT / "results/b5_audit_compare.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
