"""B2 temporal-holdout scoring (UNSEALS the holdout). Guarded: needs `resolver-frozen-v1` AND `resolver-frozen-v2`
with the frozen paths unchanged (decisions F1, V2-a, V2-c).

Sample (docs/sample_definition.md §3-4): b2_eligible pending_H3 TOIs that pass H3 (h3 files) with Tier A or Tier B
truth (B-amb excluded from the primary metric). Correct iff |P / P_true - 1| < 0.001; a method with no period is
wrong (coverage also reported).
Methods:
  baselines   raw seed periods: tls, bls, tschudi (Tschudi 2026a on the TLS result), spoc1 (single-sector SPOC TCE),
              spocE, qlp_hist, spoc_hist
  resolver    v1 and v2 learned combiner (v2 PRIMARY, V2-c), and the principled combiner, each seeded by every source
  PRIMARY "ours" (pre-registered before unsealing, decision V2-e): v2 learned combiner seeded with the TLS peak.
Reported: accuracy (Wilson 95 %), ECE of v1 and v2 per seed source, abstention/coverage at each version's threshold,
the "truth among the 11 candidates" ceiling per seed source; paired McNemar tests (exact) of ours-primary against
every baseline, and of the resolver seeded by each source against that source's raw seed; everything overall, by
truth tier (A confirmed planets / B eclipsing binaries) and by early-sector stratum.
Output: results/b2_report.json.
"""

from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from exogargantua import scoring  # noqa: E402
from exogargantua.resolver import combine as C  # noqa: E402
from b1_run import read_records  # noqa: E402
import resolver_models as RM  # noqa: E402

SOURCES = ("tls", "bls", "tschudi", "spoc1", "spocE", "qlp_hist", "spoc_hist")
STRATA = (("1", 1, 1), ("2-3", 2, 3), ("4-6", 4, 6), ("7-12", 7, 12), ("13+", 13, 10 ** 6))


def wilson(k, n, z=1.96):
    if n == 0:
        return [math.nan, math.nan]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def mcnemar(a, b):
    from scipy.stats import binomtest
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    return {"only_first_correct": n01, "only_second_correct": n10, "n": int(a.size),
            "p_value": float(binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0}


def acc(x):
    x = np.asarray(x, bool)
    return {"n": int(x.size), "accuracy": float(x.mean()) if x.size else None, "ci95": wilson(int(x.sum()), int(x.size))}


def main():
    t1 = scoring.require_frozen_resolver()
    t2 = scoring.require_frozen_resolver(tag_name=scoring.FROZEN_TAG_V2, paths=scoring.FROZEN_PATHS_V2)
    lc = {v: RM.load(v) for v in ("v1", "v2")}
    log = pd.read_csv(ROOT / "data" / "sample_log.csv", dtype={"toi": str}).set_index("toi")
    h3 = pd.concat(pd.read_csv(f, dtype={"toi": str}) for f in glob.glob(str(ROOT / "results/kaggle/b2/*/b2/h3_*.csv"))).set_index("toi")
    recs = [r for p in sorted(glob.glob(str(ROOT / "results/kaggle/b2res/*/b2res/b2res_*.jsonl.gz"))) for r in read_records(p)]
    rows = []
    for r in recs:
        toi = r["toi"]
        tier = log.loc[toi, "truth_tier"]
        elig = toi in h3.index and h3.loc[toi, "b2_eligible"] == "yes"
        if not elig or tier not in ("A", "B"):
            continue
        Pt = float(log.loc[toi, "truth_period"])
        n_early = len(str(log.loc[toi, "sectors_early"]).split(";"))
        row = {"toi": toi, "tier": tier, "stratum": next(s for s, lo, hi in STRATA if lo <= n_early <= hi)}
        for src in SOURCES:
            P = (r.get("seeds") or {}).get(src)
            row[f"seed_has_{src}"] = P is not None
            row[f"seed_ok_{src}"] = bool(P is not None and scoring.period_correct(P, Pt))
            run = r.get("runs", {}).get(f"{P:.10g}") if P is not None else None
            if run is None or "error" in run or "error" in r:
                for v in ("v1", "v2", "pr"):
                    row[f"{v}_ok_{src}"], row[f"{v}_conf_{src}"], row[f"{v}_abst_{src}"] = False, 0.0, True
                row[f"ceil_{src}"] = False
                continue
            ok = np.array([scoring.period_correct(x["P"], Pt) for x in run["rows"]])
            row[f"ceil_{src}"] = bool(ok.any())
            g = [(x["r"], x["P"], x["x"]) for x in run["rows"]]
            pa = run["summary"]["p_alias"]
            for v, p, thr in (("v1", lc["v1"][0].predict(g), lc["v1"][1]["learned"]),
                              ("v2", lc["v2"][0].predict(g), lc["v2"][1]["learned"]),
                              ("pr", np.array([pa[x["r"]] for x in run["rows"]]), lc["v2"][1]["principled"])):
                row[f"{v}_ok_{src}"], row[f"{v}_conf_{src}"] = bool(ok[int(np.argmax(p))]), float(p.max())
                row[f"{v}_abst_{src}"] = bool(p.max() < thr)
        rows.append(row)
    d = pd.DataFrame(rows)

    def block(df):
        out = {"n": int(len(df)), "baselines": {}, "resolver": {}, "paired": {}}
        for src in SOURCES:
            out["baselines"][src] = {**acc(df[f"seed_ok_{src}"]), "coverage": float(df[f"seed_has_{src}"].mean()),
                                     "accuracy_where_period_given": acc(df.loc[df[f"seed_has_{src}"], f"seed_ok_{src}"])}
            res = {"truth_in_candidates_ceiling": acc(df[f"ceil_{src}"])}
            for v in ("v1", "v2", "pr"):
                keep = ~df[f"{v}_abst_{src}"]
                res[v] = {**acc(df[f"{v}_ok_{src}"]), "ece": C.ece(df[f"{v}_conf_{src}"], df[f"{v}_ok_{src}"]) if len(df) else None,
                          "abstain_rate": float(1 - keep.mean()) if len(df) else None,
                          "accuracy_non_abstained": acc(df.loc[keep, f"{v}_ok_{src}"])}
            out["resolver"][src] = res
            out["paired"][f"resolver_v2_seeded_{src}_vs_raw_{src}"] = mcnemar(df[f"seed_ok_{src}"], df[f"v2_ok_{src}"])
            out["paired"][f"resolver_v1_seeded_{src}_vs_raw_{src}"] = mcnemar(df[f"seed_ok_{src}"], df[f"v1_ok_{src}"])
            out["paired"][f"PRIMARY_v2_tls_vs_baseline_{src}"] = mcnemar(df[f"seed_ok_{src}"], df["v2_ok_tls"])
            has = df[f"seed_has_{src}"]
            out["paired"][f"PRIMARY_v2_tls_vs_baseline_{src}_where_period_given"] = mcnemar(df.loc[has, f"seed_ok_{src}"], df.loc[has, "v2_ok_tls"])
        out["paired"]["v2_vs_v1_tls_seeded"] = mcnemar(df["v1_ok_tls"], df["v2_ok_tls"])
        return out

    rep = {"resolver_frozen_v1": t1, "resolver_frozen_v2": t2,
           "run_commits": sorted({json.loads(Path(f).read_text())["git_commit"] for f in glob.glob(str(ROOT / "results/kaggle/b2res/*/b2res/done_*.json"))}),
           "note": ("v2's calibration (temperature, thresholds) was fitted on BLS + TLS seeds only (decision V2-b); its ECE for "
                    "SPOC, QLP and Tschudi seeds is out-of-distribution. PRIMARY = v2 learned combiner seeded with the TLS peak "
                    "(V2-e). McNemar 'first' = baseline / raw seed / v1, 'second' = resolver / v2."),
           "records": len(recs), "load_errors": sum("error" in r for r in recs), "primary_sample": int(len(d)),
           "all": block(d), "by_tier": {t: block(g) for t, g in d.groupby("tier")},
           "by_stratum": {s: block(g) for s, g in d.groupby("stratum")}}
    (ROOT / "results" / "b2_report.json").write_text(json.dumps(rep, indent=1, default=float))
    a = rep["all"]
    print(json.dumps({"n": a["n"], "PRIMARY_v2_tls": a["resolver"]["tls"]["v2"],
                      "baselines": {s: a["baselines"][s]["accuracy"] for s in SOURCES}}, indent=1, default=float))


if __name__ == "__main__":
    main()
