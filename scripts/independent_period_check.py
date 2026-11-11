"""POST HOC checks of B5 against information not used by B5 or its audit (decision Indep-1).
(a) Independent truth: Kostov et al. (2025) Table 4 measures TESS periods for known EBs. B5 is rescored against those
    periods, and the catalog-truth audit cases are compared with them.
(b) CLAUDE.md rule 6: every audited star is cross-matched with the ExoFOP TOI table and the NASA Exoplanet Archive.
(c) Reviewer point on half-period EBs: TOIs whose period is half that of an EB-catalog signal on the same star, by TFOPWG
    disposition.
Also writes the per-target B5 table (sectors, SNR, periods, correctness). Outputs: results/b5_independent_check.json,
results/b5_per_target.csv, results/b5_audit_crossmatch.csv."""

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
from b1_run import read_records  # noqa: E402
from b2_score import mcnemar, wilson  # noqa: E402
import resolver_models as RM  # noqa: E402

TOL = 1e-3


def kostov():
    rows = []
    for line in (ROOT / "data/raw/kostov2025_t10k_table4_20260930.dat").read_text().splitlines():
        rows.append({"tic": int(line[0:10]), "P_kostov": float(line[37:46])})
    return pd.DataFrame(rows).drop_duplicates("tic").set_index("tic")["P_kostov"]


def rate(k, n):
    return {"count": int(k), "n": int(n), "fraction": k / n if n else None, "ci95": wilson(int(k), int(n))}


def per_target():
    v2 = RM.load("v2")
    log = pd.read_csv(ROOT / "data/b5_sample_log.csv")
    log = log[log["decision"] == "include"].set_index("tic")
    runs = {}
    for p in glob.glob(str(ROOT / "results/kaggle/b5r/*/b5/b5_*.jsonl.gz")):
        for r in read_records(p):
            runs[(r["tic"], r["engine"])] = r
    out = []
    for tic, L in log.iterrows():
        t, b = runs[(int(tic), "tls")], runs[(int(tic), "bls_gpu")]
        if "error" in t:  # TLS itself failed on this star: no TLS seed, so no resolver answer; scored as incorrect (as in b5_score.py)
            Pc, P_b = float(L["period"]), b["seed"]["P_seed"]
            out.append({"tic": int(tic), "tier": L["tier"], "sectors_early": L["sectors_early"], "n_sectors_early": len(str(L["sectors_early"]).split(";")),
                        "eclipses_in_early_window": t["h3_eclipses"], "scored": t["h3_eclipses"] >= 2, "primary_sample": L["tier"] == "B" and t["h3_eclipses"] >= 2,
                        "P_catalog": Pc, "P_tls": np.nan, "tls_sde": np.nan, "P_bls": P_b, "P_resolver": np.nan, "confidence_v2": 0.0,
                        "abstain_v2": True, "snr_seed": np.nan, "depth_seed": np.nan,
                        "ok_tls": False, "ok_bls": abs(P_b / Pc - 1) < TOL, "ok_resolver": False})
            continue
        g = [(x["r"], x["P"], x["x"]) for x in t["rows"]]
        i, c = RM.decide(v2[0], g)
        seed = next(x["x"] for x in t["rows"] if x["r"] == "1")
        Pc = float(L["period"])
        P_r, P_t, P_b = t["rows"][i]["P"], t["seed"]["P_seed"], b["seed"]["P_seed"]
        out.append({"tic": int(tic), "tier": L["tier"], "sectors_early": L["sectors_early"], "n_sectors_early": len(str(L["sectors_early"]).split(";")),
                    "eclipses_in_early_window": t["h3_eclipses"], "scored": t["h3_eclipses"] >= 2, "primary_sample": L["tier"] == "B" and t["h3_eclipses"] >= 2,
                    "P_catalog": Pc, "P_tls": P_t, "tls_sde": t["seed"].get("tls_sde"), "P_bls": P_b, "P_resolver": P_r, "confidence_v2": c,
                    "abstain_v2": c < v2[1]["learned"], "snr_seed": seed.get("snr_all"), "depth_seed": seed.get("depth"),
                    "ok_tls": abs(P_t / Pc - 1) < TOL, "ok_bls": abs(P_b / Pc - 1) < TOL, "ok_resolver": abs(P_r / Pc - 1) < TOL})
    return pd.DataFrame(out)


def main():
    d = per_target()
    d.to_csv(ROOT / "results/b5_per_target.csv", index=False, float_format="%.8g")
    K = kostov()
    rep = {"label": "POST HOC (decision Indep-1)", "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()}
    # (a) B5 against Kostov periods
    p = d[d["primary_sample"]].copy()
    p["P_k"] = p["tic"].map(K)
    q = p[p["P_k"].notna()].copy()
    ok = lambda col: np.abs(q[col] / q["P_k"] - 1) < TOL
    rep["b5_vs_kostov"] = {"n_primary_with_kostov": int(len(q)), "kostov_equals_catalog": rate(ok("P_catalog").sum(), len(q)),
                           "raw_tls": rate(ok("P_tls").sum(), len(q)), "raw_bls": rate(ok("P_bls").sum(), len(q)), "resolver_v2": rate(ok("P_resolver").sum(), len(q)),
                           "mcnemar_vs_tls": mcnemar(ok("P_tls"), ok("P_resolver")), "mcnemar_vs_bls": mcnemar(ok("P_bls"), ok("P_resolver"))}
    # audit cases
    a = pd.read_csv(ROOT / "results/kaggle/b5r_audit/b5audit/b5_audit.csv")
    a["P_kostov"] = a["tic"].map(K)
    def side(r):
        if not np.isfinite(r["P_kostov"]):
            return ""
        if abs(r["P_v2"] / r["P_kostov"] - 1) < TOL:
            return "resolver"
        if abs(r["P_catalog"] / r["P_kostov"] - 1) < TOL:
            return "catalog"
        return "neither"
    a["kostov_agrees_with"] = a.apply(side, axis=1)
    rep["audit_vs_kostov"] = {v: {s: int(((a["verdict"] == v) & (a["kostov_agrees_with"] == s)).sum()) for s in ("resolver", "catalog", "neither")}
                              for v in ("later data support v2", "later data support the catalog", "inconclusive")}
    # (b) dispositions
    toi = pd.read_csv(ROOT / "data/raw/exofop_toi_20260930.csv", comment="#", low_memory=False)
    ps = pd.read_csv(ROOT / "data/raw/pscomppars_20260930.csv.gz", comment="#", low_memory=False)
    ps["tic"] = pd.to_numeric(ps["tic_id"].astype(str).str.replace("TIC ", ""), errors="coerce")
    tg = toi.groupby("TIC ID")
    a["toi"] = a["tic"].map(lambda t: ";".join(f"{x:.2f}" for x in tg.get_group(t)["TOI"]) if t in tg.groups else "")
    a["tfopwg_disposition"] = a["tic"].map(lambda t: ";".join(str(x) for x in tg.get_group(t)["TFOPWG Disposition"]) if t in tg.groups else "")
    a["toi_period_d"] = a["tic"].map(lambda t: ";".join(f"{x:.6f}" for x in tg.get_group(t)["Period (days)"]) if t in tg.groups else "")
    pg = ps.groupby("tic")
    a["exoplanet_archive_planet"] = a["tic"].map(lambda t: ";".join(str(x) for x in pg.get_group(t)["pl_name"]) if t in pg.groups else "")
    a[["tic", "verdict", "P_kostov", "kostov_agrees_with", "toi", "tfopwg_disposition", "toi_period_d", "exoplanet_archive_planet"]].to_csv(ROOT / "results/b5_audit_crossmatch.csv", index=False)
    dec = a[a["verdict"].isin(["later data support v2", "later data support the catalog"])]
    rep["decided_audit_cases"] = {"n": int(len(dec)), "are_tois": int((dec["toi"] != "").sum()), "in_exoplanet_archive": int((dec["exoplanet_archive_planet"] != "").sum()),
                                  "tfopwg_CP_KP": int(dec["tfopwg_disposition"].str.contains("CP|KP").sum())}
    # (c) half-period EBs among TOIs
    eb = pd.read_csv(ROOT / "data/raw/tess_ebs_prsa2022_v1.0_20260930.csv")[["tess_id", "period"]].rename(columns={"tess_id": "tic", "period": "P_eb"})
    eb = pd.concat([eb, K.rename("P_eb").reset_index()], ignore_index=True).dropna()
    t2 = toi[["TIC ID", "TOI", "TFOPWG Disposition", "Period (days)"]].rename(columns={"TIC ID": "tic", "TFOPWG Disposition": "disp", "Period (days)": "P"})
    t2 = t2[pd.to_numeric(t2["P"], errors="coerce") > 0]
    m = t2.merge(eb, on="tic")
    m["ratio"] = m["P"] / m["P_eb"]
    half = m[np.abs(m["ratio"] / 0.5 - 1) < 0.01].drop_duplicates("TOI")
    same = m[np.abs(m["ratio"] - 1) < 0.01].drop_duplicates("TOI")
    open_ = lambda df: df["disp"].isna() | df["disp"].isin(["PC", "APC"])
    rep["half_period_tois"] = {"tois_total_with_period": int(len(t2)), "tois_matching_eb_period": int(len(same)), "tois_at_half_the_eb_period": int(len(half)),
                               "half_period_dispositions": {str(k): int(v) for k, v in half["disp"].fillna("none").value_counts().items()},
                               "half_period_still_open_PC_APC_or_none": int(open_(half).sum()), "half_period_CP_KP": int(half["disp"].isin(["CP", "KP"]).sum()), "half_period_FP_FA": int(half["disp"].isin(["FP", "FA"]).sum())}
    (ROOT / "results/b5_independent_check.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
