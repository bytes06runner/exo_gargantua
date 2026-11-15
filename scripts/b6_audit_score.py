"""B6 catalog-truth audit (docs/b6_definition.md P3). Same verdict rule as scripts/b5_audit_report.py.
Output: results/b6_audit_report.json."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from b5_audit_report import tally  # noqa: E402


def main():
    d = pd.read_csv(ROOT / "results/kaggle/b6_audit/b6audit/b5_audit.csv")
    done = json.loads((ROOT / "results/kaggle/b6_audit/b6audit/b5_audit_done.json").read_text())
    a = tally(d)
    v, c = a["later data support v2"]["count"], a["later data support the catalog"]["count"]
    rep = {"protocol": "docs/b6_definition.md P3", "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "run_commit": done["git_commit"], "errors": int(d["verdict"].astype(str).str.startswith("error").sum()), "all": a,
           "by_tier": {t: tally(g) for t, g in d.groupby("tier")},
           "P3_prediction": "favor-resolver and favor-catalog counts within a factor 1.5", "P3_holds": bool(max(v, c) <= 1.5 * min(v, c))}
    (ROOT / "results/b6_audit_report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: rep[k] for k in ("errors", "P3_holds")}, indent=1), v, c)


if __name__ == "__main__":
    main()
