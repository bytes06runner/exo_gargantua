"""B5 scoring (docs/b5_definition.md). Guarded by resolver-frozen-v2 (and v1, for the secondary v1 numbers).

Primary sample: Tier B, H3 (>= 2 primary eclipses in E with >= 50 % coverage), runs without load errors.
PRIMARY criterion: v2 (TLS-seeded, top-1) beats raw TLS AND raw BLS, each exact paired McNemar p < 0.01.
Secondary: accuracies (Wilson 95 %), ECE and abstention, truth-among-candidates ceiling per seed source, v2 BLS-seeded
vs raw BLS, v1, Tier B-amb separately, breakdowns by early-sector count and period range.
Also writes data/b5_audit_targets.csv: scored targets where v2's answer differs from the catalog by > 0.1 % (§6).
Output: results/b5_report.json.
"""

import glob
import json
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
from b2_score import wilson, mcnemar  # noqa: E402
import resolver_models as RM  # noqa: E402

TOL = 1e-3
PRANGE = (("<1", 0, 1), ("1-3", 1, 3), ("3-10", 3, 10), (">10", 10, 1e9))


def acc(x):
    x = np.asarray(x, bool)
    return {"n": int(x.size), "accuracy": float(x.mean()) if x.size else None, "ci95": wilson(int(x.sum()), int(x.size))}


def main():
    t1 = scoring.require_frozen_resolver()
    t2 = scoring.require_frozen_resolver(tag_name=scoring.FROZEN_TAG_V2, paths=scoring.FROZEN_PATHS_V2)
    models = {v: RM.load(v) for v in ("v1", "v2")}
    log = pd.read_csv(ROOT / "data" / "b5_sample_log.csv").set_index("tic")
    runs = {}
    for p in sorted(glob.glob(str(ROOT / "results/kaggle/b5/*/b5/b5_*.jsonl.gz"))):
        for r in read_records(p):
            runs[(r["tic"], r["engine"])] = r
    rows = []
    for tic in log.index[log["decision"] == "include"]:
        L = log.loc[tic]
        Pt = float(L["period"])
        rec = {"tic": int(tic), "tier": L["tier"], "P_true": Pt, "n_early": len(str(L["sectors_early"]).split(";")),
               "sectors_later": L["sectors_later"]}
        h3 = None
        for eng, key in (("tls", "tls"), ("bls_gpu", "bls")):
            r = runs.get((int(tic), eng))
            if r is None or "error" in r:
                rec.update({f"seed_ok_{key}": False, f"ceil_{key}": False})
                for v in ("v1", "v2"):
                    rec.update({f"{v}_ok_{key}": False, f"{v}_conf_{key}": 0.0, f"{v}_abst_{key}": True})
                rec[f"P_v2_{key}"] = np.nan
                rec[f"err_{key}"] = (r or {}).get("error", "missing")
                h3 = h3 if r is None else r.get("h3_eclipses", h3)
                continue
            h3 = r.get("h3_eclipses", h3)
            ok = np.array([abs(x["P"] / Pt - 1) < TOL for x in r["rows"]])
            g = [(x["r"], x["P"], x["x"]) for x in r["rows"]]
            rec[f"seed_ok_{key}"] = bool(abs(r["seed"]["P_seed"] / Pt - 1) < TOL)
            rec[f"ceil_{key}"] = bool(ok.any())
            for v in ("v1", "v2"):
                i, c = RM.decide(models[v][0], g)
                rec.update({f"{v}_ok_{key}": bool(ok[i]), f"{v}_conf_{key}": c, f"{v}_abst_{key}": bool(c < models[v][1]["learned"])})
                if v == "v2":
                    rec[f"P_v2_{key}"] = float(r["rows"][i]["P"])
                    rec[f"T_v2_{key}"] = float(r["rows"][i]["x"].get("tau_over_T", np.nan))
            rec[f"err_{key}"] = ""
        rec["h3"] = h3
        rows.append(rec)
    d = pd.DataFrame(rows)
    d["h3_pass"] = d["h3"].fillna(0) >= 2
    d["prange"] = [next(n for n, lo, hi in PRANGE if lo <= p < hi) for p in d["P_true"]]

    def block(df):
        out = {"n": int(len(df)), "raw": {k: acc(df[f"seed_ok_{k}"]) for k in ("tls", "bls")}}
        for key in ("tls", "bls"):
            out[f"resolver_seeded_{key}"] = {"ceiling": acc(df[f"ceil_{key}"])}
            for v in ("v1", "v2"):
                keep = ~df[f"{v}_abst_{key}"]
                out[f"resolver_seeded_{key}"][v] = {**acc(df[f"{v}_ok_{key}"]),
                                                    "ece": C.ece(df[f"{v}_conf_{key}"], df[f"{v}_ok_{key}"]) if len(df) else None,
                                                    "abstain_rate": float(1 - keep.mean()) if len(df) else None,
                                                    "accuracy_non_abstained": acc(df.loc[keep, f"{v}_ok_{key}"])}
        out["PRIMARY_v2_tls_vs_raw_tls"] = mcnemar(df["seed_ok_tls"], df["v2_ok_tls"])
        out["PRIMARY_v2_tls_vs_raw_bls"] = mcnemar(df["seed_ok_bls"], df["v2_ok_tls"])
        out["v2_bls_vs_raw_bls"] = mcnemar(df["seed_ok_bls"], df["v2_ok_bls"])
        return out

    prim = d[(d["tier"] == "B") & d["h3_pass"]]
    a = block(prim)
    met = all(a[k]["p_value"] < 0.01 and a[k]["only_second_correct"] > a[k]["only_first_correct"]
              for k in ("PRIMARY_v2_tls_vs_raw_tls", "PRIMARY_v2_tls_vs_raw_bls"))
    rep = {"resolver_frozen_v1": t1, "resolver_frozen_v2": t2,
           "run_commits": sorted({json.loads(Path(f).read_text())["git_commit"] for f in glob.glob(str(ROOT / "results/kaggle/b5/*/b5/done_*.json"))}),
           "included": int(len(d)), "h3_pass": int(d["h3_pass"].sum()), "run_errors": {k: int((d[f"err_{k}"] != "").sum()) for k in ("tls", "bls")},
           "primary_sample": int(len(prim)), "SUCCESS_CRITERION_MET": bool(met),
           "criterion": "v2 (TLS-seeded) beats raw TLS AND raw BLS, exact McNemar p < 0.01 each (docs/b5_definition.md §4)",
           "primary": a, "tier_b_amb": block(d[(d["tier"] == "B-amb") & d["h3_pass"]]),
           "by_n_early": {str(k): block(g) for k, g in prim.assign(ne=np.minimum(prim["n_early"], 4)).groupby("ne")},
           "by_period": {k: block(g) for k, g in prim.groupby("prange")}}
    (ROOT / "results" / "b5_report.json").write_text(json.dumps(rep, indent=1, default=float))
    scored = d[d["h3_pass"]]
    aud = scored[np.isfinite(scored["P_v2_tls"]) & (np.abs(scored["P_v2_tls"] / scored["P_true"] - 1) >= TOL)]
    aud[["tic", "tier", "P_true", "P_v2_tls", "sectors_later"]].to_csv(ROOT / "data" / "b5_audit_targets.csv", index=False)
    print(json.dumps({"primary_sample": rep["primary_sample"], "criterion_met": met,
                      "raw_tls": a["raw"]["tls"]["accuracy"], "raw_bls": a["raw"]["bls"]["accuracy"],
                      "v2_tls": a["resolver_seeded_tls"]["v2"]["accuracy"], "mcnemar_tls": a["PRIMARY_v2_tls_vs_raw_tls"],
                      "mcnemar_bls": a["PRIMARY_v2_tls_vs_raw_bls"], "audit_targets": int(len(aud))}, indent=1, default=float))


if __name__ == "__main__":
    main()
