"""A6 draft sanity check on a synthetic 27-d, 2-min light curve (white noise, seed 0): raw SDE (V1), A6 SDE,
and the diagnostics (cell-minimum chi2, dense grid), without and with a box transit (P = 3.7 d, depth 2 ppt).
Small enough to run locally (~seconds). Output: results/a6_synthetic_check.json."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exogargantua import search as SE, sde as SD  # noqa: E402

SEED, NOISE, PERIOD, DEPTH, DUR = 0, 1e-3, 3.7, 2e-3, 0.1


def main():
    from astropy.timeseries import BoxLeastSquares
    rng = np.random.default_rng(SEED)
    t = np.arange(0, 27, 2 / 1440.0)
    f = 1 + rng.normal(0, NOISE, t.size)
    out = {"config": {"seed": SEED, "noise": NOISE, "period": PERIOD, "depth": DEPTH, "duration": DUR, "span_d": 27, "cadence_min": 2},
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(), "cases": {}}
    for case in ("noise_only", "with_transit"):
        ff = f.copy()
        if case == "with_transit":
            ff[((t - 1.0) % PERIOD) < DUR] -= DEPTH
        tb, fb = SE.prepare(t, ff)
        pmin, pmax, _ = SE.period_limits(tb)
        g = SE.bls_grid(tb, pmin, pmax)
        res = BoxLeastSquares(tb, fb).power(g, SE.BLS_DURATIONS, objective="likelihood")
        pw = np.asarray(res.power)
        iin = SD.ivar_in_from_depth(pw, np.asarray(res.depth))
        pt = SD.tls_periods(tb, 1.0, 1.0)
        s, pk = SD.sde_a6(g, pw, iin, fb, pt)
        out["cases"][case] = {"n_bls_periods": int(len(g)), "n_tls_periods": int(len(pt)), "sde_raw": SD.sde_raw_power(pw),
                              "sde_a6": s, "peak_period_a6": pk, "peak_period_bls": float(g[np.argmax(pw)]),
                              "sde_a6_cellmax": SD.sde_a6_cellmax(g, pw, iin, fb, pt),
                              "sde_dense_tlspts": SD.sde_dense(pw, iin, fb, len(pt), "tls_points"),
                              "sde_dense_scaled": SD.sde_dense(pw, iin, fb, len(pt), "scaled"),
                              "max_in_transit_fraction": float(np.max(iin) / fb.size)}
    (ROOT / "results" / "a6_synthetic_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
