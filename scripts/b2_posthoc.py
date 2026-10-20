"""POST HOC analyses after Gate 4 (owner instruction 2026-10-20; none of these were pre-registered).

  A. Selective accuracy: on the exact B2 TOIs where v2 (TLS-seeded, primary) answers (confidence >= threshold),
     the accuracy of raw TLS, BLS, SPOC-1, SPOC-E and ours on the same TOIs (paired McNemar).
  B. Risk-coverage curves on the full B2 sample: ours (v2 confidence), TLS ranked by its own SDE, BLS ranked by its
     own maximum BLS power (no BLS SDE was stored; power is a per-light-curve quantity, noted as a limitation).
  C. Why B1 and B2 differ: B1(ii) TLS-seeded test injections vs B2 TOIs -- seed SNR and depth measured identically
     (the resolver's own fit at the seed alias, r = 1), number of sectors, true period, signal class, raw TLS
     accuracy; two-sample KS tests.
  D. B3 go/no-go: among B2 TOIs where v2 confidently disagrees (> 0.1 %) with a SPOC period (SPOC-1, SPOC-E,
     SPOC-hist) or with the current ExoFOP TOI period, how often is v2 right (n, precision, Wilson 95 %)?
Guarded by both freeze tags. Output: results/b2_posthoc_report.json.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from exogargantua import sample as S, scoring  # noqa: E402
from b1_run import read_records  # noqa: E402
from b2_score import wilson, mcnemar  # noqa: E402
import resolver_models as RM  # noqa: E402

TOL = 1e-3


def ok(P, Pt):
    return bool(P is not None and np.isfinite(P) and P > 0 and abs(P / Pt - 1) < TOL)


def acc(x):
    x = np.asarray(x, bool)
    return {"n": int(x.size), "accuracy": float(x.mean()) if x.size else None, "ci95": wilson(int(x.sum()), int(x.size))}


def risk_coverage(conf, correct, grid=np.round(np.arange(0.05, 1.0001, 0.05), 2)):
    conf, correct = np.asarray(conf, float), np.asarray(correct, bool)
    o = np.argsort(-conf, kind="stable")
    out = []
    for c in grid:
        k = max(1, int(round(c * conf.size)))
        out.append({"coverage": float(c), "accuracy": float(correct[o[:k]].mean())})
    return out


def main():
    t1 = scoring.require_frozen_resolver()
    t2 = scoring.require_frozen_resolver(tag_name=scoring.FROZEN_TAG_V2, paths=scoring.FROZEN_PATHS_V2)
    v2, thr = RM.load("v2")
    thr = thr["learned"]
    log = pd.read_csv(ROOT / "data" / "sample_log.csv", dtype={"toi": str}).set_index("toi")
    h3 = pd.concat(pd.read_csv(f, dtype={"toi": str}) for f in glob.glob(str(ROOT / "results/kaggle/b2/*/b2/h3_*.csv"))).set_index("toi")
    tls = pd.concat(pd.read_csv(f, dtype={"toi": str}) for f in glob.glob(str(ROOT / "results/kaggle/b2/*/b2/seeds_tls_*.csv"))).set_index("toi")
    bls = pd.concat(pd.read_csv(f, dtype={"toi": str}) for f in glob.glob(str(ROOT / "results/kaggle/b2/*/b2/seeds_bls_gpu_*.csv"))).set_index("toi")
    snaps = json.loads((ROOT / "data" / "raw" / "snapshots.json").read_text())
    cur = S.normalise_toi(pd.read_csv(ROOT / snaps["S-TOI"]["file"], comment="#"))
    cur_period = dict(zip(cur["toi"].round(2), cur["period"]))
    rows = []
    for r in (r for p in sorted(glob.glob(str(ROOT / "results/kaggle/b2res/*/b2res/b2res_*.jsonl.gz"))) for r in read_records(p)):
        toi = r["toi"]
        tier = log.loc[toi, "truth_tier"]
        if not (toi in h3.index and h3.loc[toi, "b2_eligible"] == "yes" and tier in ("A", "B")):
            continue
        Pt = float(log.loc[toi, "truth_period"])
        seeds = r["seeds"]
        run = r["runs"].get(f"{seeds['tls']:.10g}") if seeds.get("tls") else None
        row = {"toi": toi, "tier": tier, "n_sectors": len(str(log.loc[toi, "sectors_early"]).split(";")), "P_true": Pt,
               "tls_sde": float(tls.loc[toi, "tls_sde"]) if toi in tls.index else np.nan,
               "bls_power": float(bls.loc[toi, "bls_power_max"]) if toi in bls.index else np.nan,
               "P_current": cur_period.get(round(float(toi), 2), np.nan)}
        for s in ("tls", "bls", "spoc1", "spocE", "spoc_hist"):
            row[f"P_{s}"] = seeds.get(s)
            row[f"ok_{s}"] = ok(seeds.get(s), Pt)
        if run is None or "error" in run:
            row.update({"conf": 0.0, "P_ours": np.nan, "ok_ours": False, "snr_seed": np.nan, "depth_seed": np.nan})
        else:
            g = [(x["r"], x["P"], x["x"]) for x in run["rows"]]
            i, c = RM.decide(v2, g)
            seed_row = next(x for x in run["rows"] if x["r"] == "1")
            row.update({"conf": c, "P_ours": run["rows"][i]["P"], "ok_ours": ok(run["rows"][i]["P"], Pt),
                        "snr_seed": seed_row["x"].get("snr_all"), "depth_seed": seed_row["x"].get("depth")})
        rows.append(row)
    d = pd.DataFrame(rows)
    rep = {"label": "POST HOC (not pre-registered)", "resolver_frozen_v1": t1, "resolver_frozen_v2": t2, "n": int(len(d)),
           "v2_threshold": thr}

    # A. selective accuracy
    ans = d["conf"] >= thr
    sub = d[ans]
    rep["A_selective"] = {"answered": int(ans.sum()), "coverage": float(ans.mean()), "ours": acc(sub["ok_ours"]),
                          "on_answered_subset": {s: {**acc(sub[f"ok_{s}"]),
                                                     "mcnemar_vs_ours": mcnemar(sub[f"ok_{s}"], sub["ok_ours"]),
                                                     "coverage": float(sub[f"P_{s}"].notna().mean())}
                                                 for s in ("tls", "bls", "spoc1", "spocE")},
                          "on_abstained_subset": {s: acc(d.loc[~ans, f"ok_{s}"]) for s in ("ours", "tls", "bls", "spoc1")}}
    # B. risk-coverage
    cov_ours = float(ans.mean())
    curves = {"ours_v2_conf": risk_coverage(d["conf"], d["ok_ours"]),
              "tls_by_sde": risk_coverage(d["tls_sde"].fillna(-np.inf), d["ok_tls"]),
              "bls_by_power": risk_coverage(d["bls_power"].fillna(-np.inf), d["ok_bls"])}
    k = int(round(cov_ours * len(d)))
    at = {}
    for name, conf, corr in (("ours_v2_conf", d["conf"], d["ok_ours"]), ("tls_by_sde", d["tls_sde"].fillna(-np.inf), d["ok_tls"]),
                             ("bls_by_power", d["bls_power"].fillna(-np.inf), d["ok_bls"])):
        o = np.argsort(-np.asarray(conf), kind="stable")[:k]
        at[name] = acc(np.asarray(corr)[o])
        at[name]["area_under_accuracy_coverage"] = float(np.mean([p["accuracy"] for p in curves[name]]))
    rep["B_risk_coverage"] = {"curves": curves, "at_ours_coverage": {"coverage": cov_ours, **at},
                              "note": "BLS confidence = max BLS likelihood power (no BLS SDE stored); not comparable across light curves."}

    # C. B1(ii) vs B2
    from scipy.stats import ks_2samp
    inj = pd.read_csv(ROOT / "data" / "b1_injections.csv").set_index("inj_id")
    b1 = []
    for r in (r for p in sorted(glob.glob(str(ROOT / "results/kaggle/b1/*/b1/b1_search_tls_*.jsonl.gz"))) for r in read_records(p)):
        if "error" in r or inj.loc[r["inj_id"], "split"] != "test":
            continue
        t = inj.loc[r["inj_id"]]
        sr = next(x for x in r["rows"] if x["r"] == "1")
        b1.append({"snr_seed": sr["x"].get("snr_all"), "depth_seed": sr["x"].get("depth"), "n_sectors": int(t["n_sectors"]),
                   "P_true": float(t["P_true"]), "eb": t["cls"] != "planet", "ok_tls": ok(r["seed"]["P_seed"], float(t["P_true"]))})
    b1 = pd.DataFrame(b1)
    comp = {}
    for col in ("snr_seed", "depth_seed", "n_sectors", "P_true"):
        x, y = b1[col].dropna().to_numpy(float), d[col].dropna().to_numpy(float)
        q = lambda a: {f"p{p}": float(np.percentile(a, p)) for p in (10, 25, 50, 75, 90)}  # noqa: E731
        comp[col] = {"B1ii": q(x), "B2": q(y), "ks_stat": float(ks_2samp(x, y).statistic), "ks_p": float(ks_2samp(x, y).pvalue)}
    comp["eb_fraction"] = {"B1ii": float(b1["eb"].mean()), "B2": float((d["tier"] == "B").mean())}
    comp["single_sector_fraction"] = {"B1ii": float((b1["n_sectors"] == 1).mean()), "B2": float((d["n_sectors"] == 1).mean())}
    comp["raw_tls_accuracy"] = {"B1ii": acc(b1["ok_tls"]), "B2": acc(d["ok_tls"])}
    comp["n"] = {"B1ii": int(len(b1)), "B2": int(len(d))}
    rep["C_B1_vs_B2"] = comp

    # D. B3 go/no-go
    D = {}
    for ref in ("spoc1", "spocE", "spoc_hist", "current"):
        Pref = d["P_current"] if ref == "current" else d[f"P_{ref}"]
        has = Pref.notna() & d["P_ours"].notna()
        dis = has & ans & (np.abs(d["P_ours"] / Pref.astype(float) - 1) >= TOL)
        sub = d[dis]
        ref_ok = [ok(p, t) for p, t in zip(Pref[dis], sub["P_true"])]
        D[ref] = {"n_reference_available": int(has.sum()), "n_confident_disagreements": int(dis.sum()),
                  "v2_right": acc(sub["ok_ours"]), "reference_right": acc(ref_ok),
                  "neither_right": int(((~sub["ok_ours"]) & ~np.array(ref_ok, bool)).sum()) if len(sub) else 0}
    rep["D_B3_go_no_go"] = D
    (ROOT / "results" / "b2_posthoc_report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({"A": {k: rep["A_selective"][k] for k in ("answered", "coverage", "ours")},
                      "A_subset": {s: (v["accuracy"], v["mcnemar_vs_ours"]["p_value"]) for s, v in rep["A_selective"]["on_answered_subset"].items()},
                      "B_at_cov": {k: (v["accuracy"] if isinstance(v, dict) else v) for k, v in rep["B_risk_coverage"]["at_ours_coverage"].items()},
                      "D": {k: (v["n_confident_disagreements"], v["v2_right"]["accuracy"], v["v2_right"]["ci95"], v["reference_right"]["accuracy"])
                            for k, v in D.items()}}, indent=1, default=float))


if __name__ == "__main__":
    main()
