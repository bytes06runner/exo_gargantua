"""POST HOC check (decision Sec-1): planets with a detected secondary event in B2. Tier A only. A TOI has a detected secondary
if, in the TLS-seeded run, the alias row matching the truth within 0.1 % (else the chosen row) has sec_snr >= 5.
Output: results/b2_secondary_check.json."""

import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from b1_run import read_records  # noqa: E402
from b2_score import wilson, mcnemar  # noqa: E402
import resolver_models as RM  # noqa: E402


def main():
    v2 = RM.load("v2")[0]
    d = pd.read_csv(ROOT / "results/b2_per_target.csv", dtype={"toi": str})
    d = d[d["tier"] == "A"].set_index("toi")
    sec, matched = {}, {}
    for p in glob.glob(str(ROOT / "results/kaggle/b2res/*/b2res/b2res_*.jsonl.gz")):
        for r in read_records(p):
            if r["toi"] not in d.index or r["seeds"].get("tls") is None:
                continue
            run = r["runs"].get(f"{r['seeds']['tls']:.9f}") or next((v for k, v in r["runs"].items() if abs(float(k) / r["seeds"]["tls"] - 1) < 1e-8), None)
            if run is None or "rows" not in run:
                continue
            Pt = float(d.loc[r["toi"], "P_true"])
            m = [x for x in run["rows"] if abs(x["P"] / Pt - 1) < 1e-3]
            matched[r["toi"]] = bool(m)
            if not m:
                i, _ = RM.decide(v2, [(x["r"], x["P"], x["x"]) for x in run["rows"]])
                m = [run["rows"][i]]
            s = m[0]["x"].get("sec_snr")
            sec[r["toi"]] = float(s) if s is not None else np.nan
    d["sec_snr"] = pd.Series(sec)
    have = d[d["sec_snr"].notna()]
    det = have["sec_snr"] >= 5

    def block(x):
        return {"n": int(len(x)), "resolver": {"accuracy": float(x["ok_ours"].mean()), "ci95": wilson(int(x["ok_ours"].sum()), len(x))},
                "raw_tls": {"accuracy": float(x["ok_tls"].mean()), "ci95": wilson(int(x["ok_tls"].sum()), len(x))},
                "mcnemar_vs_tls": mcnemar(x["ok_tls"], x["ok_ours"])}

    d["truth_row"] = pd.Series(matched)
    have = d[d["sec_snr"].notna()]
    rep = {"label": "POST HOC (decision Sec-1)", "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "tier_A": int(len(d)), "with_resolver_record": int(len(have)), "detected_secondary": block(have[det]), "no_detected_secondary": block(have[~det]),
           # split added after the first result (descriptive): is the secondary seen at the true period, or only at a wrong one?
           "detected_secondary_at_true_period": block(have[det & have["truth_row"].astype(bool)]),
           "detected_secondary_true_period_not_a_candidate": int((det & ~have["truth_row"].astype(bool)).sum())}
    (ROOT / "results/b2_secondary_check.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
