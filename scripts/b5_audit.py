"""B5 secondary catalog-truth audit (docs/b5_definition.md §6). For every scored B5 target where v2's TLS-seeded answer
differs from the catalog period by > 0.1 % (data/b5_audit_targets.csv), fit the SAME eclipsing-binary model to the
later-sector light curve under each candidate period and compare chi2.

Model per period P: primary trapezoid (depth, duration T, ingress tau, epoch; P refined within +-T*P/span) and a
secondary trapezoid with the primary's T and tau at a free phase (phase searched over all phases outside the
primary, then refined on a fine grid; depth >= 0). Equal parameter counts. Errors: robust white noise x red-noise
factor. Verdict: delta chi2 = chi2(P_cat) - chi2(P_v2) > 10 -> later data support v2; < -10 -> support the catalog;
otherwise inconclusive; "insufficient" if either period predicts < 2 primary eclipses with >= 50 % coverage in L.
Implementation note (disclosed): the per-hypothesis fitted duration of v2 was not stored by the B5 runs, so T is the
resolver's seed-ephemeris duration (summary T_seed of the TLS-seeded run), used for the detrending window
max(0.75 d, 3 T) and as the fit's starting value only.
Runs on Kaggle (CPU) after scripts/build_cache.py has cached the later sectors (data/b5_pins_audit.csv).
Output: b5_audit.csv (per target) and b5_audit_done.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from exogargantua import sample as S  # noqa: E402
from exogargantua.resolver import features as F, hypotheses as H, model as M  # noqa: E402
from b1_run import read_records  # noqa: E402

DCHI2 = 10.0


def eb_model_chi2(t, y, P, T, sigma, cad):
    """Fit primary (+ secondary with the primary's shape) at period P; return chi2, refined P, n primary eclipses."""
    t0, T0 = H.seed_ephemeris(t, y, P, sigma, T=T)
    h = H.refine(H.Hypothesis(Fraction(1), 0, P, t0, T0, T0 / 4), t, y, sigma)
    n_ecl = S.n_covered_transits(t, h.P, h.t0, h.T)
    w, dt, _ = M.window(t, h.P, h.t0, h.T / 2 + h.tau)
    m1 = np.where(w, M.trapezoid(dt, h.T, h.tau), 0.0)
    d1 = max(float(np.sum(m1 * y) / max(np.sum(m1 * m1), 1e-12)), 0.0)
    resid = y - d1 * m1
    _, raw = F.secondary(t, resid, h.P, h.t0, h.T, sigma, d1, cad)
    best = (np.sum(resid ** 2), None)
    if np.isfinite(raw.get("sec_phase", np.nan)):
        for ph in raw["sec_phase"] + np.linspace(-h.T / h.P, h.T / h.P, 41):
            ws, dts, _ = M.window(t, h.P, h.t0 + ph * h.P, h.T / 2 + h.tau)
            m2 = np.where(ws, M.trapezoid(dts, h.T, h.tau), 0.0)
            d2 = max(float(np.sum(m2 * resid) / max(np.sum(m2 * m2), 1e-12)), 0.0)
            c = np.sum((resid - d2 * m2) ** 2)
            if c < best[0]:
                best = (c, ph)
    return float(best[0] / sigma ** 2), h.P, int(n_ecl), best[1]


def main():
    import b2_seeds as B2
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", required=True)
    a = ap.parse_args()
    cache = B2.find_cache(a.cache)
    man = pd.read_csv(cache / "manifest.csv")
    excluded = set(zip(man.loc[man["excluded"].notna(), "tic"], man.loc[man["excluded"].notna(), "sector"]))
    tgt = pd.read_csv(ROOT / "data" / "b5_audit_targets.csv")
    tseed = {r["tic"]: r["summary"]["T_seed"] for p in glob.glob(str(ROOT / "results/kaggle/b5/*/b5/b5_tls_*.jsonl.gz"))
             for r in read_records(p) if "summary" in r}
    rows = []
    t_start = time.time()
    for r in tgt.itertuples():
        rec = {"tic": r.tic, "tier": r.tier, "P_catalog": r.P_true, "P_v2": r.P_v2_tls, "ratio": r.P_v2_tls / r.P_true}
        try:
            z = np.load(cache / "lc" / f"{r.tic}.npz")
            secs = [int(s) for s in str(r.sectors_later).split(";")]
            tt = [z[f"s{s:04d}_time"] for s in secs if (r.tic, s) not in excluded and f"s{s:04d}_time" in z.files]
            ff = [z[f"s{s:04d}_flux"] for s in secs if (r.tic, s) not in excluded and f"s{s:04d}_time" in z.files]
            if not tt:
                rec["verdict"] = "insufficient (no later data)"
                rows.append(rec)
                continue
            t, f = np.concatenate(tt), np.concatenate(ff)
            o = np.argsort(t)
            T = float(tseed.get(r.tic, 0.1))
            t, y = M.detrend(t[o], f[o], duration=T)
            cad = M.cadence(t)
            sw = M.white_noise(y)
            sigma = sw * M.red_noise_beta(t, y, max(T, 3 * cad), sw)
            c_cat, P_cat_ref, n_cat, _ = eb_model_chi2(t, y, r.P_true, T, sigma, cad)
            c_v2, P_v2_ref, n_v2, _ = eb_model_chi2(t, y, r.P_v2_tls, T, sigma, cad)
            d = c_cat - c_v2
            rec.update({"chi2_catalog": c_cat, "chi2_v2": c_v2, "delta_chi2": d, "n_ecl_catalog": n_cat, "n_ecl_v2": n_v2,
                        "P_catalog_refined": P_cat_ref, "P_v2_refined": P_v2_ref, "n_points": int(t.size), "T_used": T})
            if min(n_cat, n_v2) < 2:
                rec["verdict"] = "insufficient"
            elif d > DCHI2:
                rec["verdict"] = "later data support v2"
            elif d < -DCHI2:
                rec["verdict"] = "later data support the catalog"
            else:
                rec["verdict"] = "inconclusive"
        except Exception as e:
            rec["verdict"] = f"error: {e!r}"[:200]
        rows.append(rec)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "b5_audit.csv", index=False)
    (out / "b5_audit_done.json").write_text(json.dumps({"git_commit": a.commit, "n": len(rows), "wall_s": time.time() - t_start}))


if __name__ == "__main__":
    main()
