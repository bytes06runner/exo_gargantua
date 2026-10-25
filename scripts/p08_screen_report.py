"""Full P08 screen result (amendments A5, A6): the injection pool after P07 and P08.

Reads results/kaggle/p08screen/*/p08screen/p08_gpu_*.csv (7 GPU sessions, 14 device-shards) and
data/injection_pool_log.csv. Writes
  data/injection_pool_screened.csv  one row per frozen pool star: stratum, P07, P08 under the raw rule (V1) and
                                     under A6, raw and A6 SDE, and `in_pool` = P07 pass and A6 pass (the rule in
                                     effect; no replacement of excluded stars)
  results/p08_screen_report.json    pool sizes by sector stratum under both rules (sensitivity, A6), checks.
"""

from __future__ import annotations

import glob
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exogargantua import p08io  # noqa: E402


def main():
    files = sorted(glob.glob(str(ROOT / "results/kaggle/p08screen/*/p08screen/p08_gpu_*.csv")))
    done = sorted(glob.glob(str(ROOT / "results/kaggle/p08screen/*/p08screen/done_*.json")))
    scr = pd.concat(p08io.read_rows(f) for f in files)
    pool = pd.read_csv(ROOT / "data" / "injection_pool_log.csv")
    pool = pool[pool["decision"] == "include"][["tic", "stratum"]]
    assert scr["tic"].is_unique, "a star was screened twice"
    m = pool.merge(scr, on="tic", how="left", validate="one_to_one")
    missing = int(m["p07"].isna().sum())
    m["in_pool"] = (m["p07"] == "pass") & (m["p08_a6"] == "pass")
    m["in_pool_raw_rule"] = (m["p07"] == "pass") & (m["p08"] == "pass")
    cols = ["tic", "stratum", "p07", "p07_reason", "window_sectors", "sde", "p08", "sde_a6", "p08_a6", "in_pool", "in_pool_raw_rule"]
    m[cols].sort_values("tic").to_csv(ROOT / "data" / "injection_pool_screened.csv", index=False)

    def sizes(col):
        by = {st: {"n": int(len(g)), "in_pool": int(g[col].sum())} for st, g in m.groupby("stratum")}
        return {"total": int(m[col].sum()), "by_stratum": by, "injections_at_10_per_star": int(m[col].sum()) * 10}

    s = m[m["p07"] == "pass"]
    rep = {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "run_commits": sorted({json.loads(Path(f).read_text())["git_commit"] for f in done}),
           "device_shards_done": len(done), "pool_stars": int(len(pool)), "screened": int(len(scr)), "missing": missing,
           "p07_pass": int((m["p07"] == "pass").sum()), "p07_exclude": int((m["p07"] == "exclude").sum()),
           "A6_rule_in_effect": sizes("in_pool"), "raw_rule_V1_sensitivity": sizes("in_pool_raw_rule"),
           "decision_changes_among_p07_pass": {"raw_exclude_a6_pass": int(((s["p08"] == "exclude") & (s["p08_a6"] == "pass")).sum()),
                                               "raw_pass_a6_exclude": int(((s["p08"] == "pass") & (s["p08_a6"] == "exclude")).sum())},
           "gpu_search_hours": float(scr["search_s"].sum() / 3600)}
    (ROOT / "results" / "p08_screen_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
