"""POST HOC group permutation importance of the frozen learned combiner (decision FI-1). No retraining.
For each group of columns, every target's values are replaced by those of another randomly chosen target at the same alias
factor (seed 20261112, 5 repeats); importance = drop in top-1 accuracy (0.1 % tolerance). Samples: B1(i) test split with
wrong-alias seeds, and the B5 primary sample with TLS seeds (rerun).
d_logpost is the principled posterior, i.e. the sum of all six terms, so single-term importances are lower limits.
Output: results/feature_importance.json."""

import glob
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from b1_run import read_records  # noqa: E402

GROUPS = {"F1 coverage": ["d_F1_coverage", "n_cov", "cov_frac", "n_missing", "snr_all"],
          "F2 odd/even": ["d_F2_oddeven", "oe_sigma"],
          "F3 secondary": ["d_F3_secondary", "sec_snr", "sec_depth_ratio", "sec_phase_off05", "sec_dur_ratio", "twin"],
          "F4 density": ["d_F4_density", "log_rho_ratio"],
          "F5 coherence": ["d_F5_coherence", "n_timed", "oc_chi2_red"],
          "F6 shape": ["d_F6_shape", "dur_ratio_oe", "shape_chi2"],
          "fitted depth and ingress": ["depth", "tau_over_T"],
          "period": ["log10_P"],
          "principled posterior": ["d_logpost"]}
SEED, REPEATS, TOL = 20261112, 5, 1e-3


def table(records):
    """Long table: one row per (target, alias) with the model columns, the alias key and correctness."""
    rows = []
    for g, (rows_, Pt) in enumerate(records):
        for x in rows_:
            rows.append({"g": g, "r": x["r"], "ok": abs(x["P"] / Pt - 1) < TOL, **x["x"]})
    return pd.DataFrame(rows)


def accuracy(lc, d, X):
    p = np.clip(lc.clf.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    z = np.log(p / (1 - p))
    best = pd.Series(z).groupby(d["g"].to_numpy()).idxmax().to_numpy()  # softmax and temperature are monotonic per target
    return float(d["ok"].to_numpy()[best].mean())


def importance(lc, records):
    d = table(records)
    assert set(c for v in GROUPS.values() for c in v) == set(lc.columns), "every model column must be in exactly one group"
    X0 = d.reindex(columns=lc.columns).to_numpy(float)
    base = accuracy(lc, d, X0)
    rng = np.random.default_rng(SEED)
    idx_by_r = {r: np.flatnonzero((d["r"] == r).to_numpy()) for r in d["r"].unique()}
    out = {}
    for name, cols in GROUPS.items():
        ci = [lc.columns.index(c) for c in cols]
        drops = []
        for _ in range(REPEATS):
            X = X0.copy()
            for r, idx in idx_by_r.items():
                X[np.ix_(idx, ci)] = X0[np.ix_(rng.permutation(idx), ci)]
            drops.append(base - accuracy(lc, d, X))
        out[name] = {"drop_mean": float(np.mean(drops)), "drop_min": float(np.min(drops)), "drop_max": float(np.max(drops))}
    return {"n": int(d["g"].nunique()), "accuracy": base, "importance": out}


TERMS = ("d_F1_coverage", "d_F2_oddeven", "d_F3_secondary", "d_F4_density", "d_F5_coherence", "d_F6_shape")


def ablation(records):
    """Decision FI-1b: principled score on the alias rows with one term left out. No fitted model involved."""
    from fractions import Fraction
    full_ok, agree, drop_ok = [], [], {k: [] for k in TERMS}
    for rows, Pt in records:
        ok = np.array([abs(x["P"] / Pt - 1) < TOL for x in rows])
        prior = np.array([-np.log(Fraction(x["r"]).numerator) for x in rows])
        F = np.array([[x["x"].get(k, 0.0) or 0.0 for k in TERMS] for x in rows], float)
        F = np.nan_to_num(F, nan=0.0)
        i = int(np.argmax(prior + F.sum(1)))
        full_ok.append(ok[i])
        agree.append(i == int(np.argmax([x["x"]["d_logpost"] for x in rows])))
        for j, k in enumerate(TERMS):
            drop_ok[k].append(ok[int(np.argmax(prior + F.sum(1) - F[:, j]))])
    base = float(np.mean(full_ok))
    return {"n": len(records), "accuracy_all_terms": base, "agreement_with_stored_principled_choice": float(np.mean(agree)),
            "without": {k: {"accuracy": float(np.mean(v)), "change": float(np.mean(v)) - base} for k, v in drop_ok.items()}}


def main():
    with open(ROOT / "models/resolver/learned.pkl", "rb") as fh:
        lc = pickle.load(fh)
    inj = pd.read_csv(ROOT / "data/b1_injections.csv").set_index("inj_id")
    b1, seen = [], set()
    for p in sorted(glob.glob(str(ROOT / "results/kaggle/b1/*/b1/b1_wrong_*.jsonl.gz"))):
        for r in read_records(p):
            if r["inj_id"] in seen or inj.loc[r["inj_id"], "split"] != "test" or "error" in r:
                continue
            seen.add(r["inj_id"])
            b1.append((r["rows"], float(inj.loc[r["inj_id"], "P_true"])))
    prim = pd.read_csv(ROOT / "results/b5_per_target.csv")
    prim = dict(zip(prim.loc[prim["primary_sample"], "tic"], prim.loc[prim["primary_sample"], "P_catalog"]))
    b5 = []
    for p in sorted(glob.glob(str(ROOT / "results/kaggle/b5r/*/b5/b5_tls_*.jsonl.gz"))):
        for r in read_records(p):
            if r["tic"] in prim and "error" not in r:
                b5.append((r["rows"], float(prim[r["tic"]])))
    rep = {"label": "POST HOC (decision FI-1)", "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "seed": SEED, "repeats": REPEATS, "groups": GROUPS,
           "note": "records with a run error are excluded, so the base accuracy can differ slightly from the reported one",
           "B1_wrong_alias_seed": importance(lc, b1), "B5_primary_tls_seed": importance(lc, b5),
           "principled_leave_one_out": {"B1_wrong_alias_seed": ablation(b1), "B5_primary_tls_seed": ablation(b5)}}
    (ROOT / "results/feature_importance.json").write_text(json.dumps(rep, indent=1))
    for k in ("B1_wrong_alias_seed", "B5_primary_tls_seed"):
        print(k, rep[k]["n"], round(rep[k]["accuracy"], 4))
        for n_, v in sorted(rep[k]["importance"].items(), key=lambda kv: -kv[1]["drop_mean"]):
            print(f"   {n_:28s} {v['drop_mean']:+.4f}  [{v['drop_min']:+.4f}, {v['drop_max']:+.4f}]")
        a_ = rep["principled_leave_one_out"][k]
        print("   principled all terms", round(a_["accuracy_all_terms"], 4), "agreement", round(a_["agreement_with_stored_principled_choice"], 4))
        for n_, v in a_["without"].items():
            print(f"     without {n_:18s} {v['accuracy']:.4f} ({v['change']:+.4f})")


if __name__ == "__main__":
    main()
