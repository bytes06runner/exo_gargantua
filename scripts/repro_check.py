"""Reproducibility check (decision Repro-1; Kaggle CPU). Reruns the frozen resolver from the stored TLS seed on 60 B5 EBs
or 60 B2 TOIs (default_rng(20261108)) and compares with the stored Kaggle records; TLS is rerun on the first 10.
Output: repro_<sample>.json."""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from exogargantua import search as SE  # noqa: E402
from exogargantua.resolver import Star, alias_rows, resolve  # noqa: E402
from b1_run import read_records  # noqa: E402
import b2_resolve as BR  # noqa: E402
import b2_seeds as B2  # noqa: E402
import b5_run as B5  # noqa: E402
import resolver_models as RM  # noqa: E402

N, N_TLS, SEED = 60, 10, 20261108
M = {}


def init(sample, cache):
    M["v2"] = RM.load("v2")
    M["sample"] = sample
    (B5 if sample == "b5" else BR).init(cache)


def compare(stored, s, hyps):
    probs = np.array([s["hyp_probs"][h.key] for h in hyps])
    rows = [{"r": k, "P": P, "x": x} for k, P, x in alias_rows(hyps, probs)]
    g0 = [(x["r"], x["P"], x["x"]) for x in stored["rows"]]
    g1 = [(x["r"], x["P"], x["x"]) for x in rows]
    i0, c0 = RM.decide(M["v2"][0], g0)
    i1, c1 = RM.decide(M["v2"][0], g1)
    dP = max(abs(a["P"] / b["P"] - 1) for a, b in zip(stored["rows"], rows))
    dF = max(abs(a["x"]["d_logpost"] - b["x"]["d_logpost"]) for a, b in zip(stored["rows"], rows)
             if np.isfinite(a["x"]["d_logpost"]) and np.isfinite(b["x"]["d_logpost"]))
    return {"same_alias": i0 == i1, "abs_conf_diff": abs(c0 - c1), "max_frac_period_diff": dP, "max_abs_logpost_diff": dF}


def one(args):
    key, rec, tls_too = args
    out = {"id": key}
    try:
        if M["sample"] == "b5":
            L = rec["log"]
            t, f = B5.light_curve(int(L["tic"]), [int(x) for x in str(L["sectors_early"]).split(";")])
            stars, rr = B5.G["stars"], rec["run"]
            tic = int(L["tic"])
            st = stars.loc[tic] if tic in stars.index else None
            star = Star(float(st["rho"]) * B5.RHO_SUN, float(st["e_rho"]) * B5.RHO_SUN) if st is not None and np.isfinite(st["rho"]) else Star()
            P, stored = float(rr["seed"]["P_seed"]), rr
        else:
            r = rec["target"]
            t, f, m = B2.early_lc(BR.G["cache"], BR.G["excluded"], r, BR.G["toi_table"])
            t, f = t[m], f[m]
            ok = np.isfinite(t) & np.isfinite(f)
            t, f = t[ok], f[ok] / np.median(f[ok])
            tic = int(r["tic"])
            st = BR.G["stars"].loc[tic] if tic in BR.G["stars"].index else None
            star = Star(float(st["rho"]) * BR.RHO_SUN, float(st["e_rho"]) * BR.RHO_SUN) if st is not None else Star()
            P = float(rec["run"]["seeds"]["tls"])
            stored = rec["run"]["runs"][f"{P:.10g}"]
        s, hyps = resolve(t, f, P, star=star)
        out.update(compare(stored, s, hyps))
        if tls_too:
            tb, fb = SE.prepare(t, f)
            rs = float(st["rad"]) if st is not None else np.nan
            ms = float(st["mass"]) if st is not None else np.nan
            out["tls_period_rerun"] = float(SE.tls_peak(tb, fb, 1, rs, ms)["tls_period"])
            out["tls_period_stored"] = P
    except Exception as e:
        out["error"] = repr(e)[:300]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", choices=["b5", "b2"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", required=True)
    a = ap.parse_args()
    cache = str(B2.find_cache(None))
    rng = np.random.default_rng(SEED)
    if a.sample == "b5":
        runs = {r["tic"]: r for p in glob.glob(str(ROOT / "results/kaggle/b5/*/b5/b5_tls_*.jsonl.gz")) for r in read_records(p) if "error" not in r}
        log = pd.read_csv(ROOT / "data" / "b5_sample_log.csv").set_index("tic")
        ids = sorted(runs)
        pick = rng.choice(ids, N, replace=False)
        jobs = [(int(k), {"log": {"tic": int(k), "sectors_early": log.loc[k, "sectors_early"]}, "run": runs[int(k)]}, i < N_TLS) for i, k in enumerate(pick)]
    else:
        runs = {r["toi"]: r for p in glob.glob(str(ROOT / "results/kaggle/b2res/*/b2res/b2res_*.jsonl.gz")) for r in read_records(p)
                if "error" not in r and r["seeds"].get("tls") is not None and "error" not in r["runs"].get(f"{r['seeds']['tls']:.10g}", {"error": 1})}
        tg = B2.b2_targets().set_index("toi", drop=False)
        ids = sorted(k for k in runs if k in tg.index)
        pick = rng.choice(ids, N, replace=False)
        jobs = [(str(k), {"target": tg.loc[k].to_dict(), "run": runs[str(k)]}, i < N_TLS) for i, k in enumerate(pick)]
    with ProcessPoolExecutor(4, initializer=init, initargs=(a.sample, cache)) as ex:
        res = list(ex.map(one, jobs, chunksize=1))
    ok = [r for r in res if "error" not in r]
    tls = [r for r in ok if "tls_period_rerun" in r]
    rep = {"git_commit": a.commit, "sample": a.sample, "n": len(res), "errors": len(res) - len(ok),
           "same_chosen_alias": sum(r["same_alias"] for r in ok), "max_abs_conf_diff": max(r["abs_conf_diff"] for r in ok),
           "max_frac_period_diff": max(r["max_frac_period_diff"] for r in ok), "max_abs_logpost_diff": max(r["max_abs_logpost_diff"] for r in ok),
           "tls_rerun_n": len(tls), "tls_same_period_within_1e-6": sum(abs(r["tls_period_rerun"] / r["tls_period_stored"] - 1) < 1e-6 for r in tls), "per_target": res}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"repro_{a.sample}.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: v for k, v in rep.items() if k != "per_target"}, indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()
