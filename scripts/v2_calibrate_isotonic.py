"""Resolver v2 (decision V2-f): fit the isotonic map on ALL v2 calibration runs (BLS + TLS seeds pooled; failed
searches have no candidates and are excluded), re-choose both abstain thresholds with the 95 % rule (b1_train.threshold).
Writes models/resolver_v2/isotonic.pkl, models/resolver_v2/calibration.json, results/v2_calibration_final.json."""

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
from exogargantua.calibration_v2 import IsotonicTopCalibrated  # noqa: E402
from exogargantua.resolver import combine as C  # noqa: E402
from b1_run import read_records  # noqa: E402
from b1_train import TOL, threshold  # noqa: E402


def main():
    from sklearn.isotonic import IsotonicRegression
    inj = pd.read_csv(ROOT / "data" / "v2_calib_injections.csv").set_index("inj_id")
    with open(ROOT / "models" / "resolver" / "learned.pkl", "rb") as fh:
        lc = pickle.load(fh)
    runs = []
    for p in sorted(glob.glob(str(ROOT / "results/kaggle/v2cal/*/v2cal/v2cal_search_*.jsonl.gz"))):
        eng = "tls" if "_tls_" in Path(p).name else "bls"
        runs += [(eng, r) for r in read_records(p)]
    ok = [(e, r) for e, r in runs if "error" not in r]
    G = [[(x["r"], x["P"], x["x"]) for x in r["rows"]] for _, r in ok]
    L = [np.array([abs(x["P"] / inj.loc[r["inj_id"], "P_true"] - 1) < TOL for x in r["rows"]]) for _, r in ok]
    P1 = [lc.predict(g) for g in G]
    top = np.array([p.max() for p in P1])
    corr = np.array([l[int(np.argmax(p))] for p, l in zip(P1, L)])
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(top, corr.astype(float))
    v2 = IsotonicTopCalibrated(lc, iso)
    P2 = [v2.predict(g) for g in G]
    c2 = np.array([p.max() for p in P2])
    k2 = np.array([l[int(np.argmax(p))] for p, l in zip(P2, L)])
    assert np.array_equal(k2, corr), "calibration must not change the chosen alias"
    cpr = np.array([max(r["summary"]["p_alias"].values()) for _, r in ok])
    kpr = np.array([l[int(np.argmax([r["summary"]["p_alias"][x["r"]] for x in r["rows"]]))] for (_, r), l in zip(ok, L)])
    thr = {"learned": threshold(c2, k2), "principled": threshold(cpr, kpr)}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    out = ROOT / "models" / "resolver_v2"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.iterdir():
        f.unlink()
    with open(out / "isotonic.pkl", "wb") as fh:
        pickle.dump(iso, fh)
    meta = {"version": "v2", "decision": "V2-f", "map": "isotonic on max probability (v1 classifier + v1 temperature)",
            "base_classifier": "models/resolver/learned.pkl (v1, unchanged)", "base_temperature": lc.temperature,
            "abstain_threshold": thr, "threshold_rule": "smallest t in 0.50..0.99 with accuracy >= 0.95 (0.99 if none)",
            "fitted_on": "all v2 calibration runs, BLS + TLS seeds pooled", "n_runs": len(ok), "fitted_at_commit": commit}
    (out / "calibration.json").write_text(json.dumps(meta, indent=1))
    eng = np.array([e for e, _ in ok])
    rep = {"git_commit": commit, "n_runs": len(ok), "errors_excluded": len(runs) - len(ok), **{k: meta[k] for k in ("abstain_threshold",)},
           "in_sample": {s: {"n": int(m.sum()), "accuracy": float(corr[m].mean()), "ece_v1": C.ece(top[m], corr[m]),
                             "ece_v2": C.ece(c2[m], k2[m]), "coverage_at_v2_threshold": float((c2[m] >= thr["learned"]).mean()),
                             "accuracy_at_v2_threshold": float(k2[m][c2[m] >= thr["learned"]].mean()) if (c2[m] >= thr["learned"]).any() else None}
                         for s, m in (("all", np.ones(len(ok), bool)), ("bls", eng == "bls"), ("tls", eng == "tls"))},
           "note": "In-sample (fit data). Out-of-sample estimate from host-split CV: results/v2_calibration_diagnostic.json."}
    (ROOT / "results" / "v2_calibration_final.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
