"""POST HOC false-alarm context (decisions FA-ctx, FA-ctx.1, FA-ctx.2): on the FA1 inverted and scrambled light curves,
how often the searches themselves cross SDE >= 9 (TLS 1.32 SDE; A6 SDE of the GPU BLS spectrum, the search that
supplied the resolver's seed; raw BLS SDE of decision V1), and how often the v2 resolver is confident when they do and
when they do not. A record with an error counts as not crossing (reported).
Inputs: results/kaggle/fac/**/fac_{tls,bls_gpu}_*.jsonl.gz and the FA1 resolver records in results/kaggle/fa/.
Output: results/fa_context_report.json."""

import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from b1_run import read_records  # noqa: E402
from b2_score import wilson  # noqa: E402
import resolver_models as RM  # noqa: E402

THR = 9.0


def rate(k, n):
    return {"count": int(k), "n": int(n), "rate": k / n if n else None, "ci95": wilson(int(k), int(n))}


def load(engine, variant):
    out = {}
    for p in sorted(glob.glob(str(ROOT / f"results/kaggle/fac/**/fac_{engine}_{variant}_*.jsonl.gz"), recursive=True)):
        for r in read_records(p):
            out[r["toi"]] = r
    return out


def main():
    v2, thr = RM.load("v2")
    rep = {"label": "POST HOC (decision FA-ctx; not pre-registered)",
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "threshold_sde": THR}
    for variant in ("scrambled", "inverted"):
        bls, tls = load("bls_gpu", variant), load("tls", variant)
        fa = {r["toi"]: r for p in sorted(glob.glob(str(ROOT / f"results/kaggle/fa/*/fa/fa_{variant}_*.jsonl.gz"))) for r in read_records(p)}
        assert set(bls) == set(fa) and set(tls) == set(fa), (variant, len(bls), len(tls), len(fa))
        n = len(fa)
        conf = {}
        for k, r in fa.items():
            if "error" in r:
                conf[k] = False
                continue
            _, c = RM.decide(v2, [(x["r"], x["P"], x["x"]) for x in r["rows"]])
            conf[k] = c >= thr["learned"]
        a9 = {k for k, r in bls.items() if r.get("sde_a6") is not None and r["sde_a6"] >= THR}
        r9 = {k for k, r in bls.items() if r.get("sde_raw") is not None and r["sde_raw"] >= THR}
        t9 = {k for k, r in tls.items() if r.get("tls_sde") is not None and np.isfinite(r["tls_sde"]) and r["tls_sde"] >= THR}
        rep[variant] = {
            "n": n,
            "errors_bls": sum("error" in r for r in bls.values()),
            "errors_tls": sum("error" in r for r in tls.values()),
            "tls_sde_ge9": rate(len(t9), n),
            "resolver_confident_given_tls_sde_ge9": rate(sum(conf[k] for k in t9), len(t9)),
            "resolver_confident_given_tls_sde_lt9": rate(sum(conf[k] for k in fa if k not in t9), n - len(t9)),
            "resolver_confident_and_tls_sde_ge9": rate(sum(conf[k] for k in t9), n),
            "a6_sde_ge9": rate(len(a9), n),
            "tls_and_a6_sde_ge9": rate(len(set(t9) & set(a9)), n),
            "tls_or_a6_sde_ge9": rate(len(set(t9) | set(a9)), n),
            "raw_sde_ge9": rate(len(r9), n),
            "resolver_v2_confident": rate(sum(conf.values()), n),
            "resolver_confident_given_a6_sde_ge9": rate(sum(conf[k] for k in a9), len(a9)),
            "resolver_confident_given_a6_sde_lt9": rate(sum(conf[k] for k in fa if k not in a9), n - len(a9)),
            "resolver_confident_and_a6_sde_ge9": rate(sum(conf[k] for k in a9), n),
        }
    (ROOT / "results" / "fa_context_report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
