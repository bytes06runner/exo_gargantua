"""Phase 3 synthetic check, analysis part (local; small): both combiners on the synthetic run.

Split by target index: train (i mod 10 in 0-5), calibration (6-7), test (8-9). The learned combiner is trained on
the train split and temperature-calibrated on the calibration split -- of SYNTHETIC data, as a code check only;
the paper's learned combiner is trained on B1 injections (brief §5(b)). Everything is evaluated on the test split.
Reported per combiner: top-1 alias accuracy, ECE (10 bins) of the maximum probability, accuracy at 90 % coverage,
abstain rate and accuracy of the non-abstained at the default threshold (0.9), by class / SNR / seed factor,
confident-resolution rate on signal-free light curves, a reliability table, the accuracy-coverage curve, and a
paired McNemar test between the combiners. Wilson 95 % intervals on proportions.
Output: results/resolver_synth_check.json.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exogargantua.resolver import combine as C  # noqa: E402

DETECTABLE_SNR = 7.1


def wilson(k, n, z=1.96):
    if n == 0:
        return [math.nan, math.nan]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def mcnemar(a, b):
    """Exact two-sided McNemar on paired correctness arrays."""
    from scipy.stats import binomtest
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"only_first_correct": n01, "only_second_correct": n10, "p_value": float(p)}


def load(paths):
    recs = []
    for p in paths:
        with gzip.open(p, "rt") as fh:
            recs += [json.loads(l) for l in fh]
    return sorted(recs, key=lambda r: r["i"])


def evaluate(conf, correct, abstain=C.DEFAULT_ABSTAIN):
    conf, correct = np.asarray(conf, float), np.asarray(correct, bool)
    n = conf.size
    keep = conf >= abstain
    acc90, th90 = C.accuracy_at_coverage(conf, correct, 0.9) if n else (math.nan, math.nan)
    rel = []
    edges = np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        rel.append({"bin": [float(lo), float(hi)], "n": int(m.sum()), "mean_conf": float(conf[m].mean()) if m.any() else None,
                    "accuracy": float(correct[m].mean()) if m.any() else None})
    return {"n": int(n), "accuracy": float(correct.mean()) if n else math.nan, "accuracy_ci95": wilson(int(correct.sum()), n),
            "ece": C.ece(conf, correct) if n else math.nan, "accuracy_at_90pct_coverage": acc90, "threshold_at_90pct_coverage": th90,
            "abstain_rate": float(1 - keep.mean()) if n else math.nan,
            "accuracy_non_abstained": float(correct[keep].mean()) if keep.any() else math.nan,
            "accuracy_non_abstained_ci95": wilson(int(correct[keep].sum()), int(keep.sum())),
            "reliability": rel,
            "accuracy_coverage": [(a, b, c) for a, b, c in C.accuracy_coverage(conf, correct, np.linspace(0, 1, 21))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=str(ROOT / "results/kaggle/resolver_synth/*/synth/synth_*.jsonl.gz"))
    ap.add_argument("--out", default=str(ROOT / "results" / "resolver_synth_check.json"))
    a = ap.parse_args()
    paths = sorted(glob.glob(a.glob))
    recs = load(paths)
    errors = [r for r in recs if "error" in r]
    recs = [r for r in recs if "error" not in r]
    split = {"train": [r for r in recs if r["i"] % 10 <= 5], "calib": [r for r in recs if r["i"] % 10 in (6, 7)],
             "test": [r for r in recs if r["i"] % 10 >= 8]}

    def groups(rs):
        return [[(x["r"], x["P"], x["x"]) for x in r["rows"]] for r in rs], [[int(x["correct"]) for x in r["rows"]] for r in rs]

    g_tr, l_tr = groups(split["train"])
    g_ca, l_ca = groups([r for r in split["calib"] if any(x["correct"] for x in r["rows"])])
    lc = C.LearnedCombiner().fit(g_tr, l_tr).calibrate(g_ca, l_ca)

    test = split["test"]
    rows = []
    for r in test:
        g = [(x["r"], x["P"], x["x"]) for x in r["rows"]]
        corr = np.array([x["correct"] for x in r["rows"]])
        pa = r["summary"]["p_alias"]
        p_pr = np.array([pa[x["r"]] for x in r["rows"]])
        p_le = lc.predict(g)
        t = r["truth"]
        rows.append({"cls": t["cls"], "snr": t["snr_true"], "seed_r": t["seed_r"], "n_sectors": t["n_sectors"],
                     "pr_conf": float(p_pr.max()), "pr_correct": bool(corr[int(np.argmax(p_pr))]),
                     "le_conf": float(p_le.max()), "le_correct": bool(corr[int(np.argmax(p_le))]),
                     "has_truth": bool(corr.any()), "runtime_s": r["summary"]["runtime_s"]})
    sig = [x for x in rows if x["cls"] != "none" and x["snr"] >= DETECTABLE_SNR]
    none = [x for x in rows if x["cls"] == "none"]
    rep = {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "inputs": [str(Path(p).relative_to(ROOT)) if Path(p).is_relative_to(ROOT) else str(p) for p in paths],
           "run_commits": sorted({json.loads(Path(p).with_name(Path(p).name.replace("synth_", "done_").replace(".jsonl.gz", ".json")).read_text())["git_commit"]
                                  for p in paths}),
           "note": "Synthetic light curves; learned combiner trained/calibrated on synthetic train/calibration splits as a code check only.",
           "n_targets": len(recs), "n_errors": len(errors), "errors": [e["error"][:200] for e in errors[:10]],
           "split_sizes": {k: len(v) for k, v in split.items()}, "learned_temperature": lc.temperature,
           "detectable_snr_threshold": DETECTABLE_SNR, "test_detectable_signals": len(sig),
           "test_truth_in_candidates": int(sum(x["has_truth"] for x in sig)), "runtime_s_median": float(np.median([x["runtime_s"] for x in rows])),
           "runtime_s_p95": float(np.percentile([x["runtime_s"] for x in rows], 95))}
    for name in ("pr", "le"):
        res = {"all_detectable": evaluate([x[f"{name}_conf"] for x in sig], [x[f"{name}_correct"] for x in sig])}
        for key, vals in (("by_class", sorted({x["cls"] for x in sig})), ("by_seed_r", sorted({x["seed_r"] for x in sig}))):
            field = "cls" if key == "by_class" else "seed_r"
            res[key] = {v: {k: e[k] for k in ("n", "accuracy", "accuracy_ci95", "ece", "abstain_rate", "accuracy_non_abstained")}
                        for v in vals for e in [evaluate([x[f"{name}_conf"] for x in sig if x[field] == v],
                                                         [x[f"{name}_correct"] for x in sig if x[field] == v])]}
        res["by_snr"] = {}
        for lo, hi in ((7.1, 15), (15, 50), (50, 1e9)):
            ss = [x for x in sig if lo <= x["snr"] < hi]
            e = evaluate([x[f"{name}_conf"] for x in ss], [x[f"{name}_correct"] for x in ss])
            res["by_snr"][f"{lo}-{hi:g}"] = {k: e[k] for k in ("n", "accuracy", "accuracy_ci95", "ece", "abstain_rate", "accuracy_non_abstained")}
        conf_none = np.array([x[f"{name}_conf"] for x in none])
        k = int(np.sum(conf_none >= C.DEFAULT_ABSTAIN))
        res["signal_free_confident_rate"] = {"n": len(none), "confident": k, "rate": k / len(none) if none else math.nan,
                                             "ci95": wilson(k, len(none))}
        rep["principled" if name == "pr" else "learned"] = res
    rep["mcnemar_principled_vs_learned"] = mcnemar(np.array([x["pr_correct"] for x in sig]), np.array([x["le_correct"] for x in sig]))
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: rep[k] for k in ("n_targets", "n_errors", "test_detectable_signals", "learned_temperature")}, indent=1))
    for n in ("principled", "learned"):
        e = rep[n]["all_detectable"]
        print(n, {k: e[k] for k in ("n", "accuracy", "ece", "accuracy_at_90pct_coverage", "abstain_rate", "accuracy_non_abstained")},
              "signal-free confident:", rep[n]["signal_free_confident_rate"]["rate"])


if __name__ == "__main__":
    main()
