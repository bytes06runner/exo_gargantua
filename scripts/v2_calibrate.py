"""Resolver v2 = recalibration only (decisions V2-a, V2-b). Refits the learned combiner's temperature and both
abstain thresholds on the v2 calibration runs (training-host injections, GPU BLS- and TLS-seeded, pooled). The
classifier itself is v1's (models/resolver/learned.pkl, unchanged). Same temperature grid/objective and threshold
rule as v1 (scripts/b1_train.py). Writes models/resolver_v2/calibration.json and results/v2_calibration_report.json
(v1 vs v2 on the calibration data, per seed source)."""

from __future__ import annotations

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
from b1_train import threshold, TOL  # noqa: E402


def main():
    inj = pd.read_csv(ROOT / "data" / "v2_calib_injections.csv").set_index("inj_id")
    assert set(inj["split"]) == {"v2cal"}
    with open(ROOT / "models" / "resolver" / "learned.pkl", "rb") as fh:
        lc = pickle.load(fh)
    v1 = json.loads((ROOT / "models" / "resolver" / "thresholds.json").read_text())
    runs = []
    for p in sorted(glob.glob(str(ROOT / "results/kaggle/v2cal/*/v2cal/v2cal_search_*.jsonl.gz"))):
        engine = "tls" if "_tls_" in Path(p).name else "bls"
        runs += [(engine, r) for r in read_records(p)]
    ok = [(e, r) for e, r in runs if "error" not in r]
    groups = [[(x["r"], x["P"], x["x"]) for x in r["rows"]] for _, r in ok]
    labels = [[int(abs(x["P"] / inj.loc[r["inj_id"], "P_true"] - 1) < TOL) for x in r["rows"]] for _, r in ok]
    has = [i for i, l in enumerate(labels) if any(l)]
    lc2 = pickle.loads(pickle.dumps(lc))
    lc2.calibrate([groups[i] for i in has], [labels[i] for i in has])

    def evaluate(model, name):
        conf, corr, eng = [], [], []
        for (e, r), g, l in zip(ok, groups, labels):
            if name == "principled":
                pa = r["summary"]["p_alias"]
                p = np.array([pa[x["r"]] for x in r["rows"]])
            else:
                p = model.predict(g)
            conf.append(float(p.max())); corr.append(bool(l[int(np.argmax(p))])); eng.append(e)
        return np.array(conf), np.array(corr), np.array(eng)

    thr2 = {}
    rep = {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "runs": len(runs), "errors_skipped": len(runs) - len(ok), "runs_with_truth_in_candidates": len(has),
           "temperature": {"v1": v1["temperature"], "v2": lc2.temperature}, "per_combiner": {}}
    for name, m1, m2 in (("learned", lc, lc2), ("principled", None, None)):
        c1, k1, e1 = evaluate(m1, name)
        c2, k2, _ = evaluate(m2, name)
        thr2[name] = threshold(c2, k2)
        block = {"threshold": {"v1": v1["abstain_threshold"][name], "v2": thr2[name]}}
        for src in ("all", "bls", "tls"):
            m = np.ones_like(e1, bool) if src == "all" else e1 == src
            block[src] = {"n": int(m.sum()), "accuracy": float(k1[m].mean()),
                          "ece_v1": C.ece(c1[m], k1[m]), "ece_v2": C.ece(c2[m], k2[m])}
        rep["per_combiner"][name] = block
    out = ROOT / "models" / "resolver_v2"
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps({
        "version": "v2", "base_classifier": "models/resolver/learned.pkl (v1, unchanged)", "temperature": lc2.temperature,
        "abstain_threshold": thr2, "threshold_rule": v1["threshold_rule"], "fitted_on": "v2 calibration runs (BLS + TLS seeds pooled)",
        "fitted_at_commit": rep["git_commit"], "n_runs": len(ok)}, indent=1))
    (ROOT / "results" / "v2_calibration_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
