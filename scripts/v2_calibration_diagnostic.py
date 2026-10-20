"""Diagnostic on the v2 calibration data (training hosts only; no test or holdout data): where is the learned
combiner miscalibrated, and how would a calibration map fitted on ALL runs behave? Compares, with 2-fold
cross-validation split by host: (i) the recorded V2-a method (temperature fitted on truth-in-candidates runs),
(ii) temperature fitted on all runs (truth absent = no candidate correct), (iii) isotonic regression of the
maximum probability on all runs. Output: results/v2_calibration_diagnostic.json."""

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
from exogargantua.resolver import combine as C  # noqa: E402
from b1_run import read_records  # noqa: E402

TOL = 1e-3


def main():
    inj = pd.read_csv(ROOT / "data" / "v2_calib_injections.csv").set_index("inj_id")
    with open(ROOT / "models" / "resolver" / "learned.pkl", "rb") as fh:
        lc = pickle.load(fh)
    recs = [r for p in sorted(glob.glob(str(ROOT / "results/kaggle/v2cal/*/v2cal/v2cal_search_*.jsonl.gz"))) for r in read_records(p)]
    recs = [r for r in recs if "error" not in r]
    Z, L, host = [], [], []
    for r in recs:
        g = [(x["r"], x["P"], x["x"]) for x in r["rows"]]
        Z.append(lc.logits(g))
        L.append(np.array([abs(x["P"] / inj.loc[r["inj_id"], "P_true"] - 1) < TOL for x in r["rows"]]))
        host.append(int(inj.loc[r["inj_id"], "tic"]))
    host = np.array(host)
    has = np.array([l.any() for l in L])

    def probs(T):
        out = []
        for z in Z:
            zz = z / T
            out.append(np.exp(zz - np.logaddexp.reduce(zz)))
        return out

    def conf_corr(P, idx=None):
        idx = range(len(P)) if idx is None else idx
        return np.array([p.max() for p in P]), np.array([L[i][int(np.argmax(p))] for p, i in zip(P, idx)])

    def fit_T(idx, all_runs):
        grid = np.exp(np.linspace(np.log(0.05), np.log(20), 200))
        best = (np.inf, 1.0)
        for T in grid:
            nll = 0.0
            for i in idx:
                zz = Z[i] / T
                lse = np.logaddexp.reduce(zz)
                if has[i]:
                    nll -= zz[int(np.argmax(L[i]))] - lse
                elif all_runs:  # truth absent: penalise the mass put on the (wrong) top candidate
                    nll -= np.log(max(1 - np.exp(zz.max() - lse), 1e-12))
            best = min(best, (nll, T))
        return best[1]

    c1, k1 = conf_corr(probs(lc.temperature))
    out = {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "runs": len(recs), "truth_in_candidates": int(has.sum()),
           "v1_on_calibration_data": {"ece_all": C.ece(c1, k1), "ece_truth_in_candidates": C.ece(c1[has], k1[has]),
                                      "mean_conf_truth_absent": float(c1[~has].mean()), "accuracy_truth_absent": float(k1[~has].mean())},
           "cross_validated_ece": {}}
    uh = np.unique(host)
    fold = np.isin(host, np.random.default_rng(20261006).permutation(uh)[: uh.size // 2])
    from sklearn.isotonic import IsotonicRegression
    res = {"V2-a_temperature_truth_in_candidates": [], "temperature_all_runs": [], "isotonic_pmax_all_runs": []}
    for tr in (fold, ~fold):
        te = ~tr
        tri, tei = np.where(tr)[0], np.where(te)[0]
        for key, allr in (("V2-a_temperature_truth_in_candidates", False), ("temperature_all_runs", True)):
            T = fit_T(tri, allr)
            PT = probs(T)
            c, k = conf_corr([PT[i] for i in tei], tei)
            res[key].append(C.ece(c, k))
        P1 = probs(lc.temperature)
        ctr, ktr = conf_corr([P1[i] for i in tri], tri)
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(ctr, ktr.astype(float))
        cte, kte = conf_corr([P1[i] for i in tei], tei)
        res["isotonic_pmax_all_runs"].append(C.ece(iso.predict(cte), kte))
    out["cross_validated_ece"] = {k: {"folds": v, "mean": float(np.mean(v))} for k, v in res.items()}
    (ROOT / "results" / "v2_calibration_diagnostic.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
