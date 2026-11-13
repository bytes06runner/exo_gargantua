"""B6 scoring (docs/b6_definition.md). Written before any B6 search result exists; run once on the complete set of runs.
1. Standard report through the B5 scorer (same code path): results/b6_report.json, data/b6_audit_targets.csv.
2. Confirmatory hypotheses H1-H3 with Bonferroni correction, recomputed here independently from the run records.
3. Secondary predictions P1 (calibration) and P2 (sum of terms, each term left out); P3 is added by the audit report.
4. Pooled B5 + B6 estimates (secondary).
Refuses to run unless every included star has a TLS and a BLS record and the search and resolver code of the B6 runs is
byte-identical to that of the B5 rerun.
Output: results/b6_confirmatory.json, results/b6_per_target.csv."""

import glob
import json
import subprocess
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
from b5_tls_floor import floor_days  # noqa: E402
from feature_importance import ablation  # noqa: E402
import b5_score  # noqa: E402
import resolver_models as RM  # noqa: E402

TOL, ALPHA, N_HYP = 1e-3, 0.01, 3
B5_RERUN_COMMIT = "ac6df0327016aee07fc3379b7b7a6a00ae09113e"


def acc(x):
    x = np.asarray(x, bool)
    return {"n": int(x.size), "accuracy": float(x.mean()) if x.size else None, "ci95": wilson(int(x.sum()), int(x.size))}


def test(base_ok, res_ok):
    m = mcnemar(np.asarray(base_ok, bool), np.asarray(res_ok, bool))
    m["passes"] = bool(m["p_value"] < ALPHA / N_HYP and m["only_second_correct"] > m["only_first_correct"])
    return m


def per_target(run_dir, log_path, stars_path):
    v2, thr = RM.load("v2")
    log = pd.read_csv(ROOT / log_path)
    log = log[log["decision"] == "include"].set_index("tic")
    st = pd.read_csv(ROOT / stars_path).drop_duplicates("tic").set_index("tic")
    runs = {}
    for p in sorted(glob.glob(str(ROOT / run_dir / "*/b[56]/b5_*.jsonl.gz"))):
        for r in read_records(p):
            runs[(r["tic"], r["engine"])] = r
    missing = [t for t in log.index if (int(t), "tls") not in runs or (int(t), "bls_gpu") not in runs]
    assert not missing, f"{len(missing)} included stars have no TLS or BLS record; B6 is scored once, on complete runs"
    rows, groups = [], {"tls": [], "bls": []}
    for tic, L in log.iterrows():
        tic, Pc = int(tic), float(L["period"])
        rec = {"tic": tic, "tier": L["tier"], "sectors_early": L["sectors_early"], "n_sectors_early": len(str(L["sectors_early"]).split(";")), "P_catalog": Pc}
        for eng, key in (("tls", "tls"), ("bls_gpu", "bls")):
            r = runs[(tic, eng)]
            rec["eclipses_in_early_window"] = r.get("h3_eclipses", rec.get("eclipses_in_early_window", 0))
            if "error" in r:  # no period: counted wrong (protocol §2)
                rec.update({f"P_{key}": np.nan, f"P_resolver_{key}": np.nan, f"conf_{key}": 0.0, f"ok_{key}": False, f"ok_resolver_{key}": False, f"ceiling_{key}": False})
                continue
            g = [(x["r"], x["P"], x["x"]) for x in r["rows"]]
            i, c = RM.decide(v2, g)
            Ps, Pr = r["seed"]["P_seed"], r["rows"][i]["P"]
            rec.update({f"P_{key}": Ps, f"P_resolver_{key}": Pr, f"conf_{key}": c, f"ok_{key}": abs(Ps / Pc - 1) < TOL, f"ok_resolver_{key}": abs(Pr / Pc - 1) < TOL,
                        f"ceiling_{key}": any(abs(x["P"] / Pc - 1) < TOL for x in r["rows"])})
            if key == "tls":
                seed = next(x["x"] for x in r["rows"] if x["r"] == "1")
                rec.update({"tls_sde": r["seed"].get("tls_sde"), "snr_seed": seed.get("snr_all"), "depth_seed": seed.get("depth")})
            groups[key].append((tic, r["rows"], Pc))
        rad, mass = (st.loc[tic, "rad"], st.loc[tic, "mass"]) if tic in st.index else (np.nan, np.nan)
        rec["tls_floor_d"] = float(floor_days(rad, mass)) if np.isfinite(rad) and np.isfinite(mass) else np.nan
        rec["abstain_v2"] = rec["conf_tls"] < thr["learned"]
        rows.append(rec)
    d = pd.DataFrame(rows)
    d["scored"] = d["eclipses_in_early_window"] >= 2
    d["primary_sample"] = d["scored"] & (d["tier"] == "B")
    return d, groups


def hypotheses(p):
    grid = p[p["tls_floor_d"].notna() & (p["P_catalog"] >= p["tls_floor_d"])]
    h1a, h1b = test(p["ok_tls"], p["ok_resolver_tls"]), test(p["ok_bls"], p["ok_resolver_tls"])
    h2, h3 = test(grid["ok_tls"], grid["ok_resolver_tls"]), test(p["ok_bls"], p["ok_resolver_bls"])
    return {"n_primary": int(len(p)), "raw_tls": acc(p["ok_tls"]), "raw_bls": acc(p["ok_bls"]), "resolver_tls_seed": acc(p["ok_resolver_tls"]),
            "resolver_bls_seed": acc(p["ok_resolver_bls"]), "ceiling_tls_seed": acc(p["ceiling_tls"]),
            "H1_tls_seeded_vs_raw_tls": h1a, "H1_tls_seeded_vs_raw_bls": h1b, "H1_passes": bool(h1a["passes"] and h1b["passes"]),
            "H2_inside_tls_grid": {"n": int(len(grid)), "raw_tls": acc(grid["ok_tls"]), "resolver_tls_seed": acc(grid["ok_resolver_tls"]), **h2},
            "H3_bls_seeded_vs_raw_bls": h3,
            "below_tls_floor": {"n": int((p["P_catalog"] < p["tls_floor_d"]).sum()), "raw_tls": acc(p.loc[p["P_catalog"] < p["tls_floor_d"], "ok_tls"]),
                                "resolver_tls_seed": acc(p.loc[p["P_catalog"] < p["tls_floor_d"], "ok_resolver_tls"])}}


def main():
    b5_score.main(runs="results/kaggle/b6", log_path="data/b6_sample_log.csv", out="results/b6_report.json", audit_out="data/b6_audit_targets.csv")
    commits = sorted({json.loads(Path(f).read_text())["git_commit"] for f in glob.glob(str(ROOT / "results/kaggle/b6/*/b6/done_*.json"))})
    same = {c: scoring._git("diff", "--quiet", c, B5_RERUN_COMMIT, "--", "src/exogargantua/search.py", "src/exogargantua/resolver", "models").returncode == 0 for c in commits}
    assert commits and all(same.values()), f"B6 runs must use the search and resolver code of the B5 rerun: {same}"
    d, groups = per_target("results/kaggle/b6", "data/b6_sample_log.csv", "data/b6_stellar_params.csv")
    d.to_csv(ROOT / "results/b6_per_target.csv", index=False, float_format="%.8g")
    p = d[d["primary_sample"]]
    std = json.loads((ROOT / "results/b6_report.json").read_text())
    assert std["primary_sample"] == len(p) and abs(std["primary"]["resolver_seeded_tls"]["v2"]["accuracy"] - p["ok_resolver_tls"].mean()) < 1e-12, "the two scorers must agree"
    prim = set(p["tic"])
    rep = {"protocol": "docs/b6_definition.md", "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "run_commits": commits, "code_identical_to_b5_rerun": same, "alpha_per_hypothesis": ALPHA / N_HYP,
           "included": int(len(d)), "scored": int(d["scored"].sum()), "run_errors_tls": int(d["P_tls"].isna().sum()), "run_errors_bls": int(d["P_bls"].isna().sum()),
           "confirmatory": hypotheses(p)}
    c = rep["confirmatory"]
    rep["all_three_hypotheses_pass"] = bool(c["H1_passes"] and c["H2_inside_tls_grid"]["passes"] and c["H3_bls_seeded_vs_raw_bls"]["passes"])
    ece = C.ece(p["conf_tls"], p["ok_resolver_tls"])
    ab = ablation([(rows, Pc) for tic, rows, Pc in groups["tls"] if tic in prim])
    learned_same = float(p.loc[p["P_tls"].notna(), "ok_resolver_tls"].mean())
    rep["secondary"] = {"P1_ece": {"ece": ece, "prediction": "above 0.05", "prediction_holds": bool(ece > 0.05)},
                        "P2_sum_of_terms": {**ab, "learned_on_same_binaries": learned_same,
                                            "prediction_within_0.02_of_learned": bool(abs(ab["accuracy_all_terms"] - learned_same) <= 0.02),
                                            "prediction_F4_not_needed": bool(ab["without"]["d_F4_density"]["change"] >= 0),
                                            "prediction_F1_F2_each_worth_0.2": bool(ab["without"]["d_F1_coverage"]["change"] < -0.2 and ab["without"]["d_F2_oddeven"]["change"] < -0.2)},
                        "abstain_rate": float(p["abstain_v2"].mean()), "accuracy_non_abstained": acc(p.loc[~p["abstain_v2"], "ok_resolver_tls"]),
                        "tier_b_amb": {"n": int((d["scored"] & (d["tier"] == "B-amb")).sum()),
                                       **{k: acc(d.loc[d["scored"] & (d["tier"] == "B-amb"), k]) for k in ("ok_tls", "ok_bls", "ok_resolver_tls")}}}
    # pooled B5 + B6 (secondary): the same table for the B5 rerun
    d5, _ = per_target("results/kaggle/b5r", "data/b5_sample_log.csv", "data/b5_stellar_params.csv")
    assert not set(d5["tic"]) & set(d["tic"])
    rep["b5_same_analysis"] = hypotheses(d5[d5["primary_sample"]])
    rep["pooled_b5_b6"] = hypotheses(pd.concat([d5[d5["primary_sample"]], p]))
    (ROOT / "results/b6_confirmatory.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: rep[k] for k in ("included", "scored", "run_errors_tls", "all_three_hypotheses_pass", "confirmatory")}, indent=1, default=float))


if __name__ == "__main__":
    main()
