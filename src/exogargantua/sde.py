"""SDE definitions for the P08 pool screen.

raw (pre-registered, decision V1): (max(power) - mean(power)) / std(power) over the full BLS trial-period grid.

A6 (DRAFT, not in effect until approved; docs/drafts/A6_P08_draft.md): the Signal Detection Efficiency as
TLS computes it (Hippke & Heller 2019, A&A 623, A39, Sect. 2.5), using TLS's own code:
transitleastsquares 1.32 (5 Apr 2024), `transitleastsquares/stats.py::spectra(chi2, oversampling_factor)`,
which detrends the spectrum with `helpers.py::running_median` over
`oversampling_factor * tls_constants.SDE_MEDIAN_KERNEL_SIZE` (= 3 * 30 -> 91) trial periods.
The search is unchanged; only the statistic changes:
  1. the pre-registered BLS spectrum is read at TLS's own trial periods
     (`transitleastsquares.period_grid` with the arguments TLS would use on this light curve, oversampling 3),
     taking the nearest BLS trial period (the BLS grid is 70-770x finer);
  2. TLS needs chi2 per period. For astropy's 'likelihood' objective with dy=None (unit weights, data
     median-subtracted) power = 0.5 * delta-chi2, so chi2(P) = sum((y - median(y))**2) - 2 * power(P);
  3. `transitleastsquares.stats.spectra(chi2, 3)` is called unmodified; its SDE (5th output) is A6's SDE.
Diagnostics only (never used for decisions): `sde_dense_*`, the same TLS detrending applied directly to the
dense BLS grid with (a) the literal 91-point kernel and (b) a kernel scaled by the grid-density ratio.
"""

from __future__ import annotations

import numpy as np

TLS_SDE_MEDIAN_KERNEL_SIZE = 30  # tls_constants.SDE_MEDIAN_KERNEL_SIZE (TLS 1.32)
TLS_OVERSAMPLING = 3             # tls_constants.OVERSAMPLING_FACTOR (TLS 1.32)


def sde_raw_power(power) -> float:
    p = np.asarray(power, float)
    return float((p.max() - p.mean()) / p.std())


def tls_periods(tb, r_star=np.nan, m_star=np.nan) -> np.ndarray:
    """The period grid TLS itself searches on this light curve, with search.tls_peak's arguments
    (TLS defaults R_star = M_star = 1 when stellar parameters are missing)."""
    from transitleastsquares import period_grid
    from . import search as SE
    pmin, pmax, _ = SE.period_limits(tb)
    rs = r_star if np.isfinite(r_star) else 1.0
    ms = m_star if np.isfinite(m_star) else 1.0
    return np.asarray(period_grid(R_star=rs, M_star=ms, time_span=float(np.max(tb) - np.min(tb)),
                                  period_min=pmin, period_max=pmax, oversampling_factor=TLS_OVERSAMPLING,
                                  n_transits_min=2), float)


def bls_chi2(power, y) -> np.ndarray:
    y = np.asarray(y, float)
    return np.sum((y - np.median(y)) ** 2) - 2.0 * np.asarray(power, float)


def at_periods(period, power, targets) -> np.ndarray:
    """power at the trial period nearest each target period."""
    o = np.argsort(period)
    p, w = np.asarray(period, float)[o], np.asarray(power, float)[o]
    i = np.clip(np.searchsorted(p, targets), 1, len(p) - 1)
    i -= (np.abs(targets - p[i - 1]) <= np.abs(p[i] - targets))
    return w[i]


def sde_a6(period, power, y, periods_tls):
    """A6 SDE and the TLS-grid period of its peak. TLS's own spectra() is called unmodified."""
    from transitleastsquares.stats import spectra
    pw = at_periods(period, power, periods_tls)
    _, _, p_det, _, sde = spectra(bls_chi2(pw, y), TLS_OVERSAMPLING)
    return float(sde), float(periods_tls[int(np.argmax(p_det))])


# ---------- diagnostics only ----------

def running_median(data, kernel: int) -> np.ndarray:
    """Same output as transitleastsquares.helpers.running_median (TLS 1.32) without its (N x kernel)
    index matrix, so it fits in memory for spectra of millions of points (tests/test_sde.py)."""
    import bottleneck as bn
    data = np.asarray(data, float)
    med = bn.move_median(data, window=kernel)[kernel - 1:]
    missing = len(data) - len(med)
    front = int(missing * 0.5)
    return np.concatenate([np.full(front, med[0]), med, np.full(missing - front, med[-1])])


def tls_spectra(chi2, kernel: int):
    """transitleastsquares 1.32 stats.spectra with an explicit kernel (diagnostic use on the dense grid)."""
    SR = np.min(chi2) / chi2
    SDE_raw = (1 - np.mean(SR)) / np.std(SR)
    power_raw = SR - np.mean(SR)
    power_raw = power_raw * (SDE_raw / np.max(power_raw))
    if kernel % 2 == 0:
        kernel = kernel + 1
    if len(power_raw) > 2 * kernel:
        power = power_raw - running_median(power_raw, kernel)
        power = power - np.mean(power)
        SDE = np.max(power / np.std(power))
        power = power * (SDE / np.max(power))
    else:
        power, SDE = power_raw, SDE_raw
    return SR, power_raw, power, float(SDE_raw), float(SDE)


def sde_dense(power, y, n_tls: int, mode: str) -> float:
    base = TLS_OVERSAMPLING * TLS_SDE_MEDIAN_KERNEL_SIZE
    k = base if mode == "tls_points" else int(round(base * len(power) / max(n_tls, 1)))
    return tls_spectra(bls_chi2(power, y), k)[4]
