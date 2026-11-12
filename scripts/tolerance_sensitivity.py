"""POST HOC sensitivity (decision Tol-1): headline results rescored with match tolerances 0.05 %, 0.1 % (primary), 0.2 % and
0.5 %. The frozen model's chosen alias and probabilities are unchanged; only the correctness test changes.
B1: test-split records (wrong-alias seeds; TLS and BLS search seeds). B5: rerun records, primary sample. B2: per-target table.
Output: results/tolerance_sensitivity.json."""

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
from b1_run import read_records  # noqa: E402
from b2_score import wilson, mcnemar  # noqa: E402
import resolver_models as RM  # noqa: E402

TOLS = (5e-4, 1e-3, 2e-3, 5e-3)


def acc(x):
    x = np.asarray(x, bool)
    return {"n": int(x.size), "accuracy": float(x.mean()), "ci95": wilson(int(x.sum()), int(x.size))}


def near(P, Pt, tol):
    P = np.asarray(P, float)
    return np.isfinite(P) & (np.abs(P / np.asarray(Pt, float) - 1) < tol)


def b1(lc):
    inj = pd.read_csv(ROOT / "data/b1_injections.csv").set_index("inj_id")
    out = {}
    for name, pat in (("wrong_alias_seed", "b1_wrong_*"), ("tls_seed", "b1_search_tls_*"), ("bls_seed", "b1_search_bls_gpu_*")):
        rows, seen = [], set()
        for p in sorted(glob.glob(str(ROOT / f"results/kaggle/b1/*/b1/{pat}.jsonl.gz"))):
            for r in read_records(p):
                if r["inj_id"] in seen or inj.loc[r["inj_id"], "split"] != "test":
                    continue
                seen.add(r["inj_id"])
                Pt = float(inj.loc[r["inj_id"], "P_true"])
                if "error" in r:
                    rows.append((Pt, (r.get("seed") or {}).get("P_seed", np.nan), np.nan))
                    continue
                i, _ = RM.decide(lc, [(x["r"], x["P"], x["x"]) for x in r["rows"]])
                rows.append((Pt, r["seed"]["P_seed"], r["rows"][i]["P"]))
        d = pd.DataFrame(rows, columns=["Pt", "seed", "res"])
        out[name] = {f"{t:g}": {"seed": acc(near(d["seed"], d["Pt"], t)), "resolver": acc(near(d["res"], d["Pt"], t)),
                                "mcnemar": mcnemar(near(d["seed"], d["Pt"], t), near(d["res"], d["Pt"], t))} for t in TOLS}
    return out


def b5():
    d = pd.read_csv(ROOT / "results/b5_per_target.csv")
    d = d[d["primary_sample"]]
    out = {}
    for t in TOLS:
        o, tl, bl = near(d["P_resolver"], d["P_catalog"], t), near(d["P_tls"], d["P_catalog"], t), near(d["P_bls"], d["P_catalog"], t)
        a, b = mcnemar(tl, o), mcnemar(bl, o)
        out[f"{t:g}"] = {"resolver": acc(o), "raw_tls": acc(tl), "raw_bls": acc(bl), "vs_tls": a, "vs_bls": b,
                         "criterion_met": bool(all(m["p_value"] < 0.01 and m["only_second_correct"] > m["only_first_correct"] for m in (a, b)))}
    return out


def b2():
    d = pd.read_csv(ROOT / "results/b2_per_target.csv")
    out = {}
    for t in TOLS:
        o, tl, bl, sp = (near(d[c], d["P_true"], t) for c in ("P_ours", "P_tls", "P_bls", "P_spoc1"))
        ms = [mcnemar(x, o) for x in (tl, bl, sp)]
        out[f"{t:g}"] = {"resolver": acc(o), "raw_tls": acc(tl), "raw_bls": acc(bl), "spoc1": acc(sp), "vs_tls": ms[0], "vs_bls": ms[1], "vs_spoc1": ms[2],
                         "criterion_met": bool(all(m["p_value"] < 0.01 and m["only_second_correct"] > m["only_first_correct"] for m in ms))}
    return out


def main():
    with open(ROOT / "models/resolver/learned.pkl", "rb") as fh:
        lc = pickle.load(fh)
    rep = {"label": "POST HOC (decision Tol-1)", "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "tolerances": list(TOLS), "B1": b1(lc), "B5": b5(), "B2": b2()}
    # the 0.1 % column must reproduce the reported results
    assert abs(rep["B5"]["0.001"]["resolver"]["accuracy"] - json.loads((ROOT / "results/b5_report.json").read_text())["primary"]["resolver_seeded_tls"]["v2"]["accuracy"]) < 1e-12
    (ROOT / "results/tolerance_sensitivity.json").write_text(json.dumps(rep, indent=1, default=float))
    for k in ("B5", "B2"):
        print(k, {t: (round(v["resolver"]["accuracy"], 3), round(v["raw_tls"]["accuracy"], 3), round(v["raw_bls"]["accuracy"], 3), v["criterion_met"], f"{v['vs_tls']['p_value']:.1e}", f"{v['vs_bls']['p_value']:.1e}") for t, v in rep[k].items()})
    for k, v in rep["B1"].items():
        print(k, {t: (round(x["seed"]["accuracy"], 3), round(x["resolver"]["accuracy"], 3)) for t, x in v.items()})


if __name__ == "__main__":
    main()
