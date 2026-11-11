"""Catalog-truth audit summary (docs/b5_definition.md §6; SECONDARY, never changes the primary B5 result).
Reads results/kaggle/b5_audit/b5audit/b5_audit.csv. Output: results/b5_audit_report.json."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from b2_score import wilson  # noqa: E402

VERDICTS = ("later data support v2", "later data support the catalog", "inconclusive", "insufficient")


def ratio_label(r):
    for name, v in (("1/2", 0.5), ("2", 2.0), ("1/3", 1 / 3), ("3", 3.0), ("2/3", 2 / 3), ("3/2", 1.5), ("1/4", 0.25), ("4", 4.0),
                    ("1/5", 0.2), ("5", 5.0)):
        if abs(r / v - 1) < 0.01:
            return name
    return "non-alias"


def tally(df):
    n = len(df)
    v = df["verdict"].where(df["verdict"].isin(VERDICTS), "insufficient")
    return {"n": int(n), **{k: {"count": int((v == k).sum()), "fraction": float((v == k).mean()) if n else None,
                                "ci95": wilson(int((v == k).sum()), n)} for k in VERDICTS}}


def main():
    d = pd.read_csv(ROOT / "results/kaggle/b5r_audit/b5audit/b5_audit.csv")
    d["ratio_label"] = [ratio_label(r) for r in d["ratio"]]
    done = json.loads((ROOT / "results/kaggle/b5r_audit/b5audit/b5_audit_done.json").read_text())
    rep = {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "run_commit": done["git_commit"], "label": "SECONDARY: catalog-truth audit (does not change the primary B5 result)",
           "errors": int(d["verdict"].str.startswith("error").sum()),
           "all": tally(d), "by_tier": {t: tally(g) for t, g in d.groupby("tier")},
           "by_ratio": {r: tally(g) for r, g in d.groupby("ratio_label")},
           "delta_chi2_median": float(np.nanmedian(d.get("delta_chi2", pd.Series(dtype=float))))}
    (ROOT / "results" / "b5_audit_report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: rep[k] for k in ("errors", "all", "by_tier")}, indent=1, default=float))
    print({r: {k: v["count"] for k, v in t.items() if k != "n"} | {"n": t["n"]} for r, t in rep["by_ratio"].items()})


if __name__ == "__main__":
    main()
