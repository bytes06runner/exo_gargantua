"""SDE definitions for the P08 pool screen.

raw (pre-registered, decision V1): (max(power) - mean(power)) / std(power) over the full BLS trial-period grid.

A6 (docs/decisions.md): the Signal Detection Efficiency as TLS computes it (Hippke & Heller 2019, A&A 623,
A39, Sect. 2.5), using TLS's own code: transitleastsquares 1.32 (5 Apr 2024),
`transitleastsquares/stats.py::spectra(chi2, oversampling_factor)`, which detrends the spectrum with
`helpers.py::running_median` over `oversampling_factor * tls_constants.SDE_MEDIAN_KERNEL_SIZE` (= 3 * 30 -> 91)
trial periods. The search is unchanged; only the statistic changes:
  1. chi2 per BLS trial period, exactly. astropy's likelihood power (bls.c, unit weights when dy is None) is
     power = 0.5 * ivar_in * (y_out - y_in)**2 = 0.5 * [chi2_ref(P) - chi2_box(P)], where chi2_box is the
     residual chi2 of the box model selected at P and chi2_ref(P) = sum((y - y_out(P))**2) is a constant
     model at that box's OUT-OF-TRANSIT mean, which differs from period to period (not the global mean or
     median; astropy's median subtraction cancels). Since chi2_ref(P) = S + ivar_in * 2 * power / W, with
     S = sum((y - mean y)**2) and W = sum(ivar) = N:
         chi2(P) = S - 2 * power(P) * (1 - ivar_in(P) / W).
     ivar_in is the number of in-transit points of the selected box: gpu_bls returns it; for astropy it is
     2 * power / depth**2, an integer to float precision (tests/test_sde.py).
  2. that chi2 is read at TLS's own trial periods (`transitleastsquares.period_grid`, TLS's arguments for this
     light curve, oversampling 3) by taking the nearest BLS trial period;
  3. `transitleastsquares.stats.spectra(chi2, 3)` is called unmodified; its SDE (5th output) is A6's SDE.
Diagnostics only (never used for decisions): `sde_a6_cellmax` (minimum chi2 within each TLS grid cell, i.e. the
maximum-power box when ivar_in is equal) and `sde_dense_*` (TLS detrending applied directly on the dense grid).
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


def ivar_in_from_depth(power, depth) -> np.ndarray:
    """In-transit point count of astropy's selected box: power = 0.5 * ivar_in * depth**2 (bls.c)."""
    power, depth = np.asarray(power, float), np.asarray(depth, float)
    out = np.zeros_like(power)
    ok = depth != 0
    out[ok] = 2.0 * power[ok] / depth[ok] ** 2
    return out


def bls_chi2(power, ivar_in, y) -> np.ndarray:
    """Exact residual chi2 of the box model behind each BLS likelihood power (unit weights)."""
    y = np.asarray(y, float)
    S = np.sum((y - np.mean(y)) ** 2)
    W = float(y.size)
    return S - 2.0 * np.asarray(power, float) * (1.0 - np.asarray(ivar_in, float) / W)


def nearest_index(period, targets) -> np.ndarray:
    """index of the trial period nearest each target period (ties to the shorter period)."""
    period = np.asarray(period, float)
    o = np.argsort(period, kind="stable")
    p = period[o]
    i = np.clip(np.searchsorted(p, targets), 1, len(p) - 1)
    i = i - (np.abs(targets - p[i - 1]) <= np.abs(p[i] - targets))
    return o[i]


def at_periods(period, values, targets) -> np.ndarray:
    return np.asarray(values, float)[nearest_index(period, targets)]


def sde_a6(period, power, ivar_in, y, periods_tls):
    """A6 SDE and the TLS-grid period of its peak. TLS's own spectra() is called unmodified."""
    from transitleastsquares.stats import spectra
    chi2 = bls_chi2(power, ivar_in, y)[nearest_index(period, periods_tls)]
    _, _, p_det, _, sde = spectra(chi2, TLS_OVERSAMPLING)
    return float(sde), float(periods_tls[int(np.argmax(p_det))])


def sde_a6_cellmax(period, power, ivar_in, y, periods_tls) -> float:
    """Diagnostic: per TLS trial period, the minimum chi2 over BLS trial periods in its cell (cell edges at
    the midpoints between neighbouring TLS periods); empty cells fall back to the nearest BLS period."""
    from transitleastsquares.stats import spectra
    chi2 = bls_chi2(power, ivar_in, y)
    period = np.asarray(period, float)
    o = np.argsort(period, kind="stable")
    ps, cs = period[o], chi2[o]
    ot = np.argsort(periods_tls, kind="stable")
    pt = np.asarray(periods_tls, float)[ot]
    edges = np.concatenate([[-np.inf], 0.5 * (pt[1:] + pt[:-1]), [np.inf]])
    lo, hi = np.searchsorted(ps, edges[:-1], "left"), np.searchsorted(ps, edges[1:], "left")
    cell = np.empty(pt.size)
    nonempty = hi > lo
    cell[nonempty] = np.minimum.reduceat(cs, lo[nonempty]) if nonempty.all() else [cs[a:b].min() for a, b in zip(lo[nonempty], hi[nonempty])]
    cell[~nonempty] = chi2[nearest_index(period, pt[~nonempty])]
    out = np.empty(pt.size)
    out[ot] = cell
    return float(spectra(out, TLS_OVERSAMPLING)[4])


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


def sde_dense(power, ivar_in, y, n_tls: int, mode: str) -> float:
    base = TLS_OVERSAMPLING * TLS_SDE_MEDIAN_KERNEL_SIZE
    k = base if mode == "tls_points" else int(round(base * len(power) / max(n_tls, 1)))
    return tls_spectra(bls_chi2(power, ivar_in, y), k)[4]
