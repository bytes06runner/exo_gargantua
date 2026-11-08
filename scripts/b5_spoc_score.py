"""POST HOC SPOC baseline for B5 (decision B5-spoc). SPOC-1: TCE from the single-sector run of the EB's first early sector.
SPOC-E: longest run lying inside the early window (ties to the later processing date). A TCE is the EB's if it is on the
same TIC and |P_TCE/(r P_cat) - 1| < 0.01 for r in the alias set; no epoch test, because a TCE may sit on the secondary
eclipse; highest tce_model_snr if several. No match: no period, counted wrong; accuracy where a period exists is also
reported. Correct within 0.1 % of the catalog period. Paired exact McNemar against the v2 TLS-seeded resolver.
Inputs: data/spoc_tces_b5_hosts.csv.gz, results/b5_per_target.csv. Outputs: results/b5_spoc_report.json,
results/b5_spoc_seeds.csv."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from exogargantua import sample as S  # noqa: E402
from b2_score import mcnemar, wilson  # noqa: E402

TOL = 1e-3


def best(cands, Pc):
    keep = [c for c in cands if S.alias_ratio_match(c["tce_period"], Pc) is not None]
    if not keep:
        return np.nan
    return float(max(keep, key=lambda c: c["tce_model_snr"] if np.isfinite(c["tce_model_snr"]) else -np.inf)["tce_period"])


def seeds(row, t):
    early = [int(x) for x in str(row["sectors_early"]).split(";")]
    Pc = float(row["P_catalog"])
    one = t[(t["run_s0"] == early[0]) & (t["run_s1"] == early[0])]
    p1 = best(one.to_dict("records"), Pc)
    pE = np.nan
    inside = t[(t["run_s0"] >= min(early)) & (t["run_s1"] <= max(early))].copy()
    if not inside.empty:
        inside["span"] = inside["run_s1"] - inside["run_s0"]
        for span in sorted(inside["span"].unique(), reverse=True):
            for run in inside[inside["span"] == span].sort_values("run", ascending=False)["run"].unique():
                pE = best(inside[inside["run"] == run].to_dict("records"), Pc)
                if np.isfinite(pE):
                    break
            if np.isfinite(pE):
                break
    return p1, pE


def acc(x):
    x = np.asarray(x, bool)
    return {"n": int(x.size), "accuracy": float(x.mean()) if x.size else None, "ci95": wilson(int(x.sum()), int(x.size))}


def block(d):
    out = {"n": int(len(d))}
    for s in ("spoc1", "spocE"):
        has, ok = d[f"has_{s}"], d[f"ok_{s}"]
        out[s] = {**acc(ok), "coverage": float(has.mean()), "accuracy_where_period_given": acc(ok[has]),
                  "resolver_vs_spoc": mcnemar(ok, d["ok_resolver"]), "resolver_vs_spoc_where_period_given": mcnemar(ok[has], d.loc[has, "ok_resolver"])}
    return out


def main():
    d = pd.read_csv(ROOT / "results/b5_per_target.csv")
    tces = pd.read_csv(ROOT / "data/spoc_tces_b5_hosts.csv.gz")
    by = {int(k): g for k, g in tces.groupby("ticid")}
    empty = tces.iloc[0:0]
    rows = [seeds(r, by.get(int(r["tic"]), empty)) for r in d.to_dict("records")]
    d["P_spoc1"], d["P_spocE"] = [x[0] for x in rows], [x[1] for x in rows]
    for s in ("spoc1", "spocE"):
        d[f"has_{s}"] = d[f"P_{s}"].notna()
        d[f"ok_{s}"] = d[f"has_{s}"] & (np.abs(d[f"P_{s}"] / d["P_catalog"] - 1) < TOL)
    d[["tic", "P_spoc1", "P_spocE"]].to_csv(ROOT / "results/b5_spoc_seeds.csv", index=False, float_format="%.8g")
    rep = {"label": "POST HOC (decision B5-spoc); the B5 success criterion is unchanged",
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "stars_with_any_tce": int(tces["ticid"].nunique()), "primary": block(d[d["primary_sample"]]), "tier_b_amb": block(d[(d["tier"] == "B-amb") & d["scored"]])}
    (ROOT / "results/b5_spoc_report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
