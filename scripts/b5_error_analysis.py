"""POST HOC, descriptive (decision Err-1): every B5 primary binary where the TLS-seeded v2 period is wrong, assigned to the
first matching category (order fixed in docs/decisions.md before counting). Also, for context, the error rate of all primary
binaries carrying each flag (flags are not exclusive there).
Output: results/b5_error_analysis.json and results/b5_error_analysis.csv."""

import glob
import json
import subprocess
import sys
from math import pi
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from b1_run import read_records  # noqa: E402
from b2_score import wilson  # noqa: E402

TOL = 1e-3
CATS = ["TLS search failed", "truth not among candidates, below TLS period floor", "truth not among candidates, other",
        "period within 1% but outside 0.1%", "nearly equal eclipses (depth ratio > 0.8)", "eccentric (secondary > 0.05 from phase 0.5)",
        "at most three eclipses in the early window", "seed S/N below 20", "other"]


def first(a, b):
    return a.where(a.notna(), b)


def main():
    d = pd.read_csv(ROOT / "results/b5_per_target.csv")
    d = d[d["primary_sample"]].copy()
    eb = pd.read_csv(ROOT / "data/raw/tess_ebs_prsa2022_v1.0_20260930.csv").drop_duplicates("tess_id").set_index("tess_id")
    e = eb.reindex(d["tic"])
    prim, sec = first(e["prim_depth_pf"], e["prim_depth_2g"]).to_numpy(), first(e["sec_depth_pf"], e["sec_depth_2g"]).to_numpy()
    ppos, spos = first(e["prim_pos_pf"], e["prim_pos_2g"]).to_numpy(), first(e["sec_pos_pf"], e["sec_pos_2g"]).to_numpy()
    d["depth_ratio"] = np.where((prim > 0) & np.isfinite(sec), sec / prim, np.nan)
    d["sec_phase_off"] = np.abs(((spos - ppos) % 1.0) - 0.5)
    st = pd.read_csv(ROOT / "data/b5_stellar_params.csv").set_index("tic")
    rad, mass = d["tic"].map(st["rad"]), d["tic"].map(st["mass"])
    d["below_floor"] = d["P_catalog"] < 2 * pi * np.sqrt((3 * rad * 6.957e8) ** 3 / (6.674e-11 * mass * 1.989e30)) / 86400
    ceil = {}
    for p in glob.glob(str(ROOT / "results/kaggle/b5r/*/b5/b5_tls_*.jsonl.gz")):
        for r in read_records(p):
            ceil[r["tic"]] = None if "error" in r else r["rows"]
    Pc = dict(zip(d["tic"], d["P_catalog"]))
    d["tls_failed"] = [ceil[t] is None for t in d["tic"]]
    d["truth_in_candidates"] = [ceil[t] is not None and any(abs(x["P"] / Pc[t] - 1) < TOL for x in ceil[t]) for t in d["tic"]]
    d["near_miss"] = (np.abs(d["P_resolver"] / d["P_catalog"] - 1) < 0.01) & ~d["ok_resolver"]
    flags = {"nearly equal eclipses (depth ratio > 0.8)": d["depth_ratio"] > 0.8, "eccentric (secondary > 0.05 from phase 0.5)": d["sec_phase_off"] > 0.05,
             "at most three eclipses in the early window": d["eclipses_in_early_window"] <= 3, "seed S/N below 20": d["snr_seed"] < 20,
             "below TLS period floor": d["below_floor"]}
    conds = [d["tls_failed"], ~d["truth_in_candidates"] & d["below_floor"], ~d["truth_in_candidates"], d["near_miss"],
             flags[CATS[4]], flags[CATS[5]], flags[CATS[6]], flags[CATS[7]], pd.Series(True, index=d.index)]
    cat = pd.Series("", index=d.index)
    for name, c in zip(CATS, conds):
        cat = cat.where((cat != "") | ~c.fillna(False), name)
    d["category"] = cat.where(~d["ok_resolver"], "")
    wrong = d[~d["ok_resolver"]]
    n = len(wrong)

    def rate(m):
        m = m.fillna(False)
        k, N = int((~d.loc[m, "ok_resolver"]).sum()), int(m.sum())
        return {"n_with_flag": N, "wrong": k, "error_rate": k / N if N else None, "ci95": wilson(k, N)}

    rep = {"label": "POST HOC, descriptive (decision Err-1)", "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "primary_n": int(len(d)), "wrong": int(n), "overall_error_rate": n / len(d),
           "categories": {c: {"count": int((wrong["category"] == c).sum()), "fraction_of_errors": float((wrong["category"] == c).mean())} for c in CATS},
           "error_rate_by_flag": {k: rate(v) for k, v in flags.items()},
           "catalog_depth_ratio_available": int(d["depth_ratio"].notna().sum()), "catalog_secondary_phase_available": int(d["sec_phase_off"].notna().sum())}
    assert sum(v["count"] for v in rep["categories"].values()) == n
    d.loc[~d["ok_resolver"], ["tic", "P_catalog", "P_tls", "P_resolver", "confidence_v2", "snr_seed", "eclipses_in_early_window", "depth_ratio", "sec_phase_off", "below_floor", "category"]].to_csv(
        ROOT / "results/b5_error_analysis.csv", index=False, float_format="%.6g")
    (ROOT / "results/b5_error_analysis.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
