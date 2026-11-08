"""Independent audit of the reported benchmark numbers. Recomputes them from the raw Kaggle records with statistics
written here (Wilson interval, exact binomial McNemar with integer arithmetic, ECE), without importing the scoring
scripts, and compares with results/*.json. Also checks the sample separation the benchmarks rely on (no star shared
between training data and held-out samples, early and later windows disjoint). Only the frozen classifier is shared
with the original scoring, because the numbers being checked are its outputs.
Output: results/audit_recompute.json. Exit status 1 if anything disagrees."""

import glob
import gzip
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import resolver_models as RM  # noqa: E402

TOL = 1e-3
CHECKS = []


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def mcnemar_p(b, c):
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k + 1))
    from fractions import Fraction
    return float(min(Fraction(1), Fraction(2 * tail, 2 ** n)))


def ece(conf, ok, bins=10):
    conf, ok = np.asarray(conf, float), np.asarray(ok, float)
    tot, n = 0.0, conf.size
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        m = (conf > lo) & (conf <= hi) if i else (conf >= 0) & (conf <= hi)
        if m.any():
            tot += m.sum() / n * abs(conf[m].mean() - ok[m].mean())
    return float(tot)


def records(pattern):
    out = []
    for p in sorted(glob.glob(str(ROOT / pattern), recursive=True)):
        try:
            with gzip.open(p, "rt") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            out.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass  # truncated last line of a shard cut by the session limit
        except EOFError:
            pass  # shard cut by the 12 h session limit and completed by its --resume run
    return out


def same(name, mine, reported, tol=1e-9):
    ok = (mine == reported) if isinstance(mine, (int, str, bool, np.integer)) else abs(float(mine) - float(reported)) <= tol
    CHECKS.append({"check": name, "recomputed": mine if not isinstance(mine, np.integer) else int(mine), "reported": reported, "ok": bool(ok)})


def J(n):
    return json.loads((ROOT / "results" / n).read_text())


def choose(models, rows):
    g = [(x["r"], x["P"], x["x"]) for x in rows]
    return {v: RM.decide(models[v][0], g) for v in models}


def audit_b5(models):
    rep = J("b5_report.json")
    log = pd.read_csv(ROOT / "data" / "b5_sample_log.csv").set_index("tic")
    runs = {}
    for r in records("results/kaggle/b5/*/b5/b5_*.jsonl.gz"):
        runs[(r["tic"], r["engine"])] = r
    inc = log[log["decision"] == "include"]
    same("B5 included", int(len(inc)), rep["included"])
    rows = []
    for tic, L in inc.iterrows():
        Pt = float(L["period"])
        t, b = runs[(int(tic), "tls")], runs[(int(tic), "bls_gpu")]
        assert "error" not in t and "error" not in b
        if t["h3_eclipses"] < 2:
            continue
        ch = choose(models, t["rows"])
        i, c = ch["v2"]
        rows.append({"tic": int(tic), "tier": L["tier"], "P": Pt, "tls": abs(t["seed"]["P_seed"] / Pt - 1) < TOL, "bls": abs(b["seed"]["P_seed"] / Pt - 1) < TOL,
                     "v2": abs(t["rows"][i]["P"] / Pt - 1) < TOL, "conf": c, "ceil": any(abs(x["P"] / Pt - 1) < TOL for x in t["rows"])})
    d = pd.DataFrame(rows)
    same("B5 H3 pass", int(len(d)), rep["h3_pass"])
    p = d[d["tier"] == "B"]
    pr = rep["primary"]
    same("B5 primary N", int(len(p)), rep["primary_sample"])
    for k, key in (("tls", ("raw", "tls")), ("bls", ("raw", "bls"))):
        same(f"B5 raw {k} accuracy", float(p[k].mean()), pr[key[0]][key[1]]["accuracy"])
    v = pr["resolver_seeded_tls"]["v2"]
    same("B5 v2 accuracy", float(p["v2"].mean()), v["accuracy"])
    lo, hi = wilson(int(p["v2"].sum()), len(p))
    same("B5 v2 CI low", lo, v["ci95"][0], 1e-3)
    same("B5 v2 CI high", hi, v["ci95"][1], 1e-3)
    same("B5 v2 ECE", ece(p["conf"], p["v2"]), v["ece"], 1e-9)
    same("B5 v2 abstain", float((p["conf"] < models["v2"][1]["learned"]).mean()), v["abstain_rate"])
    same("B5 ceiling", float(p["ceil"].mean()), pr["resolver_seeded_tls"]["ceiling"]["accuracy"])
    for base, key in (("tls", "PRIMARY_v2_tls_vs_raw_tls"), ("bls", "PRIMARY_v2_tls_vs_raw_bls")):
        fix, brk = int((p["v2"] & ~p[base]).sum()), int((~p["v2"] & p[base]).sum())
        same(f"B5 McNemar vs {base}: resolver only", fix, pr[key]["only_second_correct"])
        same(f"B5 McNemar vs {base}: seed only", brk, pr[key]["only_first_correct"])
        same(f"B5 McNemar vs {base}: p (ratio)", mcnemar_p(fix, brk) / pr[key]["p_value"], 1.0, 1e-6)
    return d


def audit_b2(models):
    rep = J("b2_report.json")
    log = pd.read_csv(ROOT / "data" / "sample_log.csv", dtype={"toi": str}).set_index("toi")
    h3 = pd.concat(pd.read_csv(f, dtype={"toi": str}) for f in glob.glob(str(ROOT / "results/kaggle/b2/*/b2/h3_*.csv"))).set_index("toi")
    recs = records("results/kaggle/b2res/*/b2res/b2res_*.jsonl.gz")
    same("B2 records", len(recs), rep["records"])
    rows = []
    for r in recs:
        toi = r["toi"]
        tier = log.loc[toi, "truth_tier"]
        if toi not in h3.index or h3.loc[toi, "b2_eligible"] != "yes" or tier not in ("A", "B"):
            continue
        Pt = float(log.loc[toi, "truth_period"])
        row = {"toi": toi, "tier": tier, "n_early": len(str(log.loc[toi, "sectors_early"]).split(";"))}
        for s in ("tls", "bls", "spoc1"):
            P = r["seeds"].get(s)
            row[f"ok_{s}"] = P is not None and abs(P / Pt - 1) < TOL
            row[f"has_{s}"] = P is not None
        Pts = r["seeds"].get("tls")
        run = r["runs"].get(f"{Pts:.10g}") if Pts is not None else None
        if run is None or "error" in run:
            row.update({"ours": False, "conf": 0.0})
        else:
            i, c = choose(models, run["rows"])["v2"]
            row.update({"ours": abs(run["rows"][i]["P"] / Pt - 1) < TOL, "conf": c})
        rows.append(row)
    d = pd.DataFrame(rows)
    al = rep["all"]
    same("B2 primary N", int(len(d)), rep["primary_sample"])
    same("B2 v2 accuracy", float(d["ours"].mean()), al["resolver"]["tls"]["v2"]["accuracy"])
    same("B2 v2 ECE", ece(d["conf"], d["ours"]), al["resolver"]["tls"]["v2"]["ece"], 1e-9)
    for s in ("tls", "bls", "spoc1"):
        same(f"B2 raw {s}", float(d[f"ok_{s}"].mean()), al["baselines"][s]["accuracy"])
        m = al["paired"][f"PRIMARY_v2_tls_vs_baseline_{s}"]
        fix, brk = int((d["ours"] & ~d[f"ok_{s}"]).sum()), int((~d["ours"] & d[f"ok_{s}"]).sum())
        same(f"B2 McNemar vs {s}: resolver only", fix, m["only_second_correct"])
        same(f"B2 McNemar vs {s}: seed only", brk, m["only_first_correct"])
        same(f"B2 McNemar vs {s}: p (ratio)", mcnemar_p(fix, brk) / m["p_value"], 1.0, 1e-6)
    one = d[d["n_early"] == 1]
    s1 = rep["by_stratum"]["1"]
    same("B2 single-sector N", int(len(one)), s1["n"])
    same("B2 single-sector resolver", float(one["ours"].mean()), s1["resolver"]["tls"]["v2"]["accuracy"])
    same("B2 single-sector TLS", float(one["ok_tls"].mean()), s1["baselines"]["tls"]["accuracy"])
    csv = pd.read_csv(ROOT / "results" / "b2_per_target.csv", dtype={"toi": str}).set_index("toi")
    bad = (d.set_index("toi")["ours"] != csv.loc[d["toi"], "ok_ours"]).sum()
    same("B2 per-target table disagreements", int(bad), 0)
    return d


def audit_b1(models):
    rep = J("b1_test_report.json")
    post = J("b1_v2_posthoc_report.json")
    inj = pd.read_csv(ROOT / "data" / "b1_injections.csv").set_index("inj_id")
    for key, pat in (("B1_i_wrong_seed", "results/kaggle/b1/*/b1/b1_wrong_*.jsonl.gz"), ("B1_ii_tls_seed", "results/kaggle/b1/*/b1/b1_search_tls_*.jsonl.gz"),
                     ("B1_ii_bls_seed", "results/kaggle/b1/*/b1/b1_search_bls_gpu_*.jsonl.gz")):
        seen, ok, c1, c2, n = set(), [], [], [], 0
        for r in records(pat):
            if r["inj_id"] in seen or inj.loc[r["inj_id"], "split"] != "test":
                continue
            seen.add(r["inj_id"])
            Pt = float(inj.loc[r["inj_id"], "P_true"])
            n += 1
            if "error" in r:
                ok.append(False), c1.append(0.0), c2.append(0.0)
                continue
            ch = choose(models, r["rows"])
            i, c = ch["v1"]
            ok.append(abs(r["rows"][i]["P"] / Pt - 1) < TOL), c1.append(c), c2.append(ch["v2"][1])
        same(f"{key} N", n, rep[key]["n"])
        same(f"{key} accuracy", float(np.mean(ok)), rep[key]["learned"]["all"]["accuracy"])
        same(f"{key} v1 ECE", ece(c1, ok), rep[key]["learned"]["all"]["ece"], 1e-9)
        same(f"{key} v2 ECE", ece(c2, ok), post["v2"][key]["all"]["ece"], 1e-9)


def audit_fa(models):
    fa = J("false_alarm_report.json")["variants"]
    ctx = J("fa_context_report.json")
    for v in ("scrambled", "inverted"):
        recs = {r["toi"]: r for r in records(f"results/kaggle/fa/*/fa/fa_{v}_*.jsonl.gz")}
        conf = {k: ("error" not in r and choose(models, r["rows"])["v2"][1] >= models["v2"][1]["learned"]) for k, r in recs.items()}
        same(f"FA {v} N", len(recs), fa[v]["n"])
        same(f"FA {v} v2 confident", sum(conf.values()), fa[v]["v2"]["confident"])
        t = {r["toi"]: r for r in records(f"results/kaggle/fac/**/fac_tls_{v}_*.jsonl.gz")}
        b = {r["toi"]: r for r in records(f"results/kaggle/fac/**/fac_bls_gpu_{v}_*.jsonl.gz")}
        same(f"FA ctx {v} TLS coverage", len(t), len(recs))
        same(f"FA ctx {v} BLS coverage", len(b), len(recs))
        same(f"FA ctx {v} TLS SDE>=9", sum(r["tls_sde"] >= 9 for r in t.values()), ctx[v]["tls_sde_ge9"]["count"])
        same(f"FA ctx {v} BLS SDE>=9", sum(r["sde_a6"] >= 9 for r in b.values()), ctx[v]["a6_sde_ge9"]["count"])


def audit_b4():
    rt = J("b4_runtime.json")
    b = pd.DataFrame(records("results/kaggle/b4/*/b4/b4_bls_*.jsonl.gz"))
    r = pd.DataFrame(records("results/kaggle/b4/*/b4/b4_resolver_*.jsonl.gz"))
    m = b.merge(r, on="tic", suffixes=("_g", "_c"))
    tot = (m["load_s_g"] + m["prepare_s"] + m["bls_s"] + m["resolver_s"]).dropna()
    same("B4 complete", int(tot.size), rt["complete_pipeline"])
    same("B4 median total", float(np.median(tot)), rt["per_light_curve"]["total"]["median_s"], 1e-9)


def audit_audit():
    a = J("b5_audit_report.json")["all"]
    d = pd.read_csv(ROOT / "results/kaggle/b5_audit/b5audit/b5_audit.csv")
    same("Audit N", int(len(d)), a["n"])
    enough = (d["n_ecl_catalog"] >= 2) & (d["n_ecl_v2"] >= 2)
    same("Audit favor resolver", int(((d["delta_chi2"] > 10) & enough).sum()), a["later data support v2"]["count"])
    same("Audit favor catalog", int(((d["delta_chi2"] < -10) & enough).sum()), a["later data support the catalog"]["count"])


def separation():
    out = {}
    pool = pd.read_csv(ROOT / "data" / "injection_pool_screened.csv")
    pool = set(pool.loc[pool["in_pool"].astype(bool), "tic"])
    log = pd.read_csv(ROOT / "data" / "sample_log.csv", dtype={"toi": str})
    b2 = set(pd.read_csv(ROOT / "results" / "b2_per_target.csv")["tic"])
    b5log = pd.read_csv(ROOT / "data" / "b5_sample_log.csv")
    b5 = set(b5log.loc[b5log["decision"] == "include", "tic"])
    split = pd.read_csv(ROOT / "data" / "b1_split.csv").set_index("tic")["split"]
    same("Stars in both injection pool and B2", len(pool & b2), 0)
    same("Stars in both injection pool and B5", len(pool & b5), 0)
    same("Stars in both B2 and B5", len(b2 & b5), 0)
    cal = pd.read_csv(ROOT / "data" / "v2_calib_injections.csv")
    same("v2 calibration hosts in B1 test split", int((split.reindex(cal["tic"].unique()) == "test").sum()), 0)
    def overlap(df):
        return sum(bool(set(str(a).split(";")) & set(str(b).split(";"))) for a, b in zip(df["sectors_early"], df["sectors_later"]) if pd.notna(b))
    same("B2 targets with a sector in both windows", overlap(log), 0)
    same("B5 targets with a sector in both windows", overlap(b5log[b5log["decision"] == "include"]), 0)
    return out


def main():
    models = {v: RM.load(v) for v in ("v1", "v2")}
    audit_b5(models)
    audit_b2(models)
    audit_b1(models)
    audit_fa(models)
    audit_b4()
    audit_audit()
    separation()
    bad = [c for c in CHECKS if not c["ok"]]
    rep = {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "n_checks": len(CHECKS), "n_failed": len(bad), "failed": bad, "checks": CHECKS}
    (ROOT / "results" / "audit_recompute.json").write_text(json.dumps(rep, indent=1, default=float))
    print(f"{len(CHECKS)} checks, {len(bad)} failed")
    for c in bad:
        print("FAILED", c)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
