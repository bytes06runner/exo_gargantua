"""Pre-registered seed searches (docs/sample_definition.md §6, docs/baselines.md §1-2).

`prepare` (wotan biweight 0.75 d, 10-min bins), `bls_peak` (astropy BLS, autoperiod grid) and
`tls_peak` are the single implementation used by the smoke run, the benchmark runs and the
tests. `bls_peaks_parallel` runs several independent `bls_peak` calls in worker processes
(amendment A1); it changes where BLS runs, never what it computes.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

BLS_DURATIONS = np.array([0.04, 0.06, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30])
P_MIN, P_MAX_CAP = 0.5, 30.0


def bin10(t, f, width=10.0 / 1440.0):
    idx = np.floor((t - t.min()) / width).astype(np.int64)
    counts = np.bincount(idx)
    keep = counts > 0
    tb = (np.bincount(idx, weights=t) / np.maximum(counts, 1))[keep]
    fb = (np.bincount(idx, weights=f) / np.maximum(counts, 1))[keep]
    return tb, fb


def prepare(time_, flux):
    """Detrend (wotan biweight, 0.75 d, break tolerance 0.5 d) and bin to 10 min."""
    from wotan import flatten
    flat = flatten(time_, flux, method="biweight", window_length=0.75, break_tolerance=0.5)
    ok = np.isfinite(flat)
    return bin10(time_[ok], flat[ok])


def period_limits(tb, pmax_cap=P_MAX_CAP):
    span = tb.max() - tb.min()
    return P_MIN, min(pmax_cap, 0.95 * span), span


def bls_grid(tb, pmin, pmax):
    from astropy.timeseries import BoxLeastSquares
    return BoxLeastSquares(tb, np.ones_like(tb)).autoperiod(
        BLS_DURATIONS, minimum_period=pmin, maximum_period=pmax, minimum_n_transit=2, frequency_factor=1.0)


def bls_peak(tb, fb):
    """Raw BLS peak on the pre-registered grid. Returns period, grid size and CPU time."""
    from astropy.timeseries import BoxLeastSquares
    t0 = time.time()
    pmin, pmax, _ = period_limits(tb)
    periods = bls_grid(tb, pmin, pmax)
    res = BoxLeastSquares(tb, fb).power(periods, BLS_DURATIONS, objective="likelihood")
    i = int(np.argmax(res.power))
    return {"bls_period": float(res.period[i]), "bls_index": i, "bls_n_periods": int(len(periods)),
            "bls_power_max": float(res.power[i]), "bls_s": time.time() - t0}


def _bls_peak_args(args):
    return bls_peak(*args)


def bls_peaks_parallel(inputs, workers=4):
    """Amendment A1: independent bls_peak calls, `workers` at a time. Output order = input order."""
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_bls_peak_args, inputs))


def tls_peak(tb, fb, n_threads, r_star=np.nan, m_star=np.nan, return_results=False):
    from transitleastsquares import transitleastsquares
    t0 = time.time()
    pmin, pmax, _ = period_limits(tb)
    kw = dict(period_min=pmin, period_max=pmax, use_threads=n_threads, show_progress_bar=False)
    if np.isfinite(r_star) and np.isfinite(m_star):
        kw.update(R_star=r_star, M_star=m_star, R_star_min=0.1, R_star_max=max(3.0, 1.5 * r_star),
                  M_star_min=0.1, M_star_max=max(2.5, 1.5 * m_star))
    res = transitleastsquares(tb, fb).power(**kw)
    out = {"tls_period": float(res.period), "tls_sde": float(res.SDE), "tls_n_periods": int(len(res.periods)),
           "tls_s": time.time() - t0}
    return (out, res) if return_results else out
