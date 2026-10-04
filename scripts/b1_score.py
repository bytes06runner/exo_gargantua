"""Score the B1 TEST split (docs/injection_design.md §5-6). Guarded: refuses unless `resolver-frozen-v1` exists
and the resolver (code, trained combiner, thresholds) is unchanged since the tag (decision F1).

B1(i): wrong-seed runs on all test injections. B1(ii): the 2,000-injection subsample seeded by TLS and by BLS.
Per combiner (principled, learned with the frozen temperature) and abstain thresholds from
models/resolver/thresholds.json: alias accuracy (refined period within 0.1 % of the truth), ECE, accuracy at
90 % coverage, abstain rate, accuracy of the non-abstained -- overall, by sector stratum, by alias (seed)
factor and by class, with Wilson 95 % intervals; paired McNemar between the combiners.
Additional (not pre-registered, reported as such): the same split by the number of true transit epochs inside
the observed sectors (0-1 / 2-3 / >= 4).
Output: results/b1_test_report.json.
"""

from __future__ import annotations

import glob
import gzip
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exogargantua import sample as S, scoring  # noqa: E402
from exogargantua.resolver import combine as C  # noqa: E402

SECTOR_DAYS = 27.4


def wilson(k, n, z=1.96):
    if n == 0:
        return [math.nan, math.nan]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def metrics(conf, correct, thr):
    conf, correct = np.asarray(conf, float), np.asarray(correct, bool)
    n = int(conf.size)
    if n == 0:
        return {"n": 0}
    keep = conf >= thr
    a90, _ = C.accuracy_at_coverage(conf, correct, 0.9)
    return {"n": n, "accuracy": float(correct.mean()), "accuracy_ci95": wilson(int(correct.sum()), n), "ece": C.ece(conf, correct),
            "accuracy_at_90pct_coverage": a90, "abstain_rate": float(1 - keep.mean()),
            "abstain_rate_ci95": wilson(int((~keep).sum()), n),
            "accuracy_non_abstained": float(correct[keep].mean()) if keep.any() else None,
            "accuracy_non_abstained_ci95": wilson(int(correct[keep].sum()), int(keep.sum()))}


def mcnemar(a, b):
    from scipy.stats import binomtest
    n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    return {"only_principled_correct": n01, "only_learned_correct": n10,
            "p_value": float(binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0}


def n_epochs(row, starts):
    n = 0
    for s in str(row["sectors"]).split(";"):
        a = starts[int(s)]
        k0, k1 = math.ceil((a - row["t0_true"]) / row["P_true"]), math.floor((a + SECTOR_DAYS - row["t0_true"]) / row["P_true"])
        n += max(0, k1 - k0 + 1)
    return n


def score(pattern, inj, lc, thr, starts):
    rows = []
    for p in sorted(glob.glob(str(ROOT / pattern))):
        with gzip.open(p, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                if inj.loc[r["inj_id"], "split"] != "test":
                    continue
                t = inj.loc[r["inj_id"]]
                base = {"inj_id": r["inj_id"], "stratum": t["stratum"], "seed_r": t["seed_r"], "cls": t["cls"],
                        "epochs": n_epochs(t, starts), "error": "error" in r}
                if "error" in r:
                    rows.append({**base, "pr_conf": 0.0, "pr_ok": False, "le_conf": 0.0, "le_ok": False})
                    continue
                ok = np.array([scoring.period_correct(x["P"], t["P_true"]) for x in r["rows"]])
                pa = r["summary"]["p_alias"]
                p_pr = np.array([pa[x["r"]] for x in r["rows"]])
                p_le = lc.predict([(x["r"], x["P"], x["x"]) for x in r["rows"]])
                rows.append({**base, "pr_conf": float(p_pr.max()), "pr_ok": bool(ok[int(np.argmax(p_pr))]),
                             "le_conf": float(p_le.max()), "le_ok": bool(ok[int(np.argmax(p_le))])})
    d = pd.DataFrame(rows)
    if d.empty:
        return None
    d["epoch_bin"] = pd.cut(d["epochs"], [-1, 1, 3, 10 ** 6], labels=["0-1", "2-3", ">=4"]).astype(str)
    out = {"n": int(len(d)), "errors_counted_wrong": int(d["error"].sum())}
    for name, key in (("principled", "pr"), ("learned", "le")):
        res = {"all": metrics(d[f"{key}_conf"], d[f"{key}_ok"], thr[name])}
        for by in ("stratum", "seed_r", "cls", "epoch_bin"):
            res[f"by_{by}"] = {str(v): metrics(g[f"{key}_conf"], g[f"{key}_ok"], thr[name]) for v, g in d.groupby(by)}
        out[name] = res
    out["mcnemar"] = mcnemar(d["pr_ok"].to_numpy(), d["le_ok"].to_numpy())
    return out


def main():
    tag_commit = scoring.require_frozen_resolver()
    inj = pd.read_csv(ROOT / "data" / "b1_injections.csv").set_index("inj_id")
    with open(ROOT / "models" / "resolver" / "learned.pkl", "rb") as fh:
        lc = pickle.load(fh)
    thr = json.loads((ROOT / "models" / "resolver" / "thresholds.json").read_text())["abstain_threshold"]
    starts = S.sector_start_btjd(pd.read_csv(sorted((ROOT / "data" / "raw").glob("tess_orbit_times_*.csv"))[0]))
    rep = {"resolver_frozen_v1": tag_commit, "abstain_threshold": thr,
           "B1_i_wrong_seed": score("results/kaggle/b1/*/b1/b1_wrong_*.jsonl.gz", inj, lc, thr, starts),
           "B1_ii_tls_seed": score("results/kaggle/b1/*/b1/b1_search_tls_*.jsonl.gz", inj, lc, thr, starts),
           "B1_ii_bls_seed": score("results/kaggle/b1/*/b1/b1_search_bls_gpu_*.jsonl.gz", inj, lc, thr, starts)}
    (ROOT / "results" / "b1_test_report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: (v if not isinstance(v, dict) or "principled" not in v else
                          {c: v[c]["all"] for c in ("principled", "learned")}) for k, v in rep.items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
