"""Train the learned combiner and choose both abstain thresholds on the B1 TRAINING split only
(docs/injection_design.md §6, decision F1). Test-split records are never read.

  fit split   -> HistGradientBoosting combiner (B1(i) wrong-seed runs)
  calib split -> its temperature; each combiner's abstain threshold: the smallest t in {0.50, ..., 0.99} with
                 accuracy >= 0.95 among non-abstained calibration injections (0.99 if none)
Writes models/resolver/learned.pkl, models/resolver/thresholds.json (frozen with the resolver, F1) and
results/b1_training_report.json (training-split metrics only).
"""

from __future__ import annotations

import glob
import gzip
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exogargantua.resolver import combine as C  # noqa: E402

TOL = 1e-3
TARGET_ACC = 0.95
GRID = np.round(np.arange(0.50, 1.00, 0.01), 2)


def load_training():
    inj = pd.read_csv(ROOT / "data" / "b1_injections.csv").set_index("inj_id")
    keep = set(inj.index[inj["split"].isin(["fit", "calib"])])
    recs = []
    for p in sorted(glob.glob(str(ROOT / "results/kaggle/b1/*/b1/b1_wrong_*.jsonl.gz"))):
        with gzip.open(p, "rt") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec["inj_id"] in keep:  # test-split records are discarded unread beyond their id
                    recs.append(rec)
    return inj, recs


def threshold(conf, correct):
    conf, correct = np.asarray(conf), np.asarray(correct, bool)
    for t in GRID:
        m = conf >= t
        if m.any() and correct[m].mean() >= TARGET_ACC:
            return float(t)
    return 0.99


def main():
    inj, recs = load_training()
    ok = [r for r in recs if "error" not in r]
    split = {s: [r for r in ok if inj.loc[r["inj_id"], "split"] == s] for s in ("fit", "calib")}

    def groups(rs):
        g = [[(x["r"], x["P"], x["x"]) for x in r["rows"]] for r in rs]
        lab = [[int(abs(x["P"] / inj.loc[r["inj_id"], "P_true"] - 1) < TOL) for x in r["rows"]] for r in rs]
        return g, lab

    g_fit, l_fit = groups(split["fit"])
    g_cal, l_cal = groups(split["calib"])
    lc = C.LearnedCombiner().fit(g_fit, l_fit)
    has_truth = [i for i, l in enumerate(l_cal) if any(l)]
    lc.calibrate([g_cal[i] for i in has_truth], [l_cal[i] for i in has_truth])

    conf = {"principled": [], "learned": []}
    corr = {"principled": [], "learned": []}
    for r, g, l in zip(split["calib"], g_cal, l_cal):
        pa = r["summary"]["p_alias"]
        p_pr = np.array([pa[x["r"]] for x in r["rows"]])
        p_le = lc.predict(g)
        for name, p in (("principled", p_pr), ("learned", p_le)):
            conf[name].append(float(p.max()))
            corr[name].append(bool(l[int(np.argmax(p))]))
    thr = {name: threshold(conf[name], corr[name]) for name in conf}

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    md = ROOT / "models" / "resolver"
    md.mkdir(parents=True, exist_ok=True)
    with open(md / "learned.pkl", "wb") as fh:
        pickle.dump(lc, fh)
    meta = {"trained_at_commit": commit, "sklearn_params": lc.params, "temperature": lc.temperature, "columns": lc.columns,
            "abstain_threshold": thr, "threshold_rule": f"smallest t in 0.50..0.99 with calib accuracy >= {TARGET_ACC}",
            "n_fit": len(split["fit"]), "n_calib": len(split["calib"]),
            "errors_skipped": len(recs) - len(ok)}
    (md / "thresholds.json").write_text(json.dumps(meta, indent=1))
    rep = {"git_commit": commit, **{k: v for k, v in meta.items() if k != "columns"}, "calibration_split": {}}
    for name in conf:
        c, k = np.array(conf[name]), np.array(corr[name])
        keep = c >= thr[name]
        rep["calibration_split"][name] = {"n": int(c.size), "accuracy": float(k.mean()), "ece": C.ece(c, k),
                                          "coverage_at_threshold": float(keep.mean()),
                                          "accuracy_at_threshold": float(k[keep].mean()) if keep.any() else None}
    (ROOT / "results" / "b1_training_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
