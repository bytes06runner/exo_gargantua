"""B5 sample accounting, catalog -> primary sample, one count per pre-registered rule of docs/b5_definition.md.
Selection rules Y01-Y05 and the S01 cap come from data/b5_sample_log.csv (written by scripts/b5_select.py at selection time).
Tier B-amb (§2) comes from the tier column of the same log. H3 (§1, after download) is computed by the run itself and stored
in every run record (h3_eclipses); this script writes the per-star H3 log to data/b5_h3_log.csv.
Output: results/b5_sample_accounting.json."""

import glob
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from b1_run import read_records  # noqa: E402


def h3(run):
    out = {}
    for p in glob.glob(str(ROOT / f"results/kaggle/{run}/*/b5/b5_*.jsonl.gz")):
        for r in read_records(p):
            out[(r["tic"], r["engine"])] = r.get("h3_eclipses")
    return out


def main():
    log = pd.read_csv(ROOT / "data/b5_sample_log.csv")
    rule = log["reason"].fillna("").str.extract(r"^(Y0\d|S01)")[0].fillna("included")
    inc = log[log["decision"] == "include"].copy()
    new, old = h3("b5r"), h3("b5")
    inc["h3_eclipses"] = [new[(int(t), "bls_gpu")] for t in inc["tic"]]
    assert all(new[(int(t), "tls")] == new[(int(t), "bls_gpu")] for t in inc["tic"]), "H3 must not depend on the engine"
    assert all(old[(int(t), "bls_gpu")] == new[(int(t), "bls_gpu")] for t in inc["tic"]), "H3 must be identical in both runs"
    inc["h3_pass"] = inc["h3_eclipses"] >= 2
    inc[["tic", "tier", "h3_eclipses", "h3_pass"]].to_csv(ROOT / "data/b5_h3_log.csv", index=False)
    amb = inc["tier"] == "B-amb"
    rep = {"catalog_rows": int(len(log)),
           "excluded_by_rule": {k: int(v) for k, v in rule[log["decision"] == "exclude"].value_counts().sort_index().items()},
           "passed_Y01_Y05": int((log["decision"] == "include").sum() + (rule == "S01").sum()),
           "included_after_S01_cap": int(len(inc)),
           "tier_B_amb_reported_separately": int(amb.sum()), "tier_B": int((~amb).sum()),
           "tier_B_failing_H3": int((~amb & ~inc["h3_pass"]).sum()), "tier_B_amb_failing_H3": int((amb & ~inc["h3_pass"]).sum()),
           "primary_sample": int((~amb & inc["h3_pass"]).sum()), "h3_identical_in_both_runs": True}
    assert rep["primary_sample"] == json.loads((ROOT / "results/b5_report.json").read_text())["primary_sample"]
    (ROOT / "results/b5_sample_accounting.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
