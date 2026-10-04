"""Light-curve conditioning, the trapezoid transit template, noise estimates and linear fits.

All fits work on y = 1 - flux (dips positive) of the detrended light curve.
"""

from __future__ import annotations

import numpy as np

MIN_COVERAGE = 0.5      # an epoch counts as observed if >= 50 % of its expected in-transit cadences exist


def detrend(t, f, duration, min_window=0.75):
    """wotan biweight (Hippke et al. 2019), window max(0.75 d, 3 x duration) so that the transit is not
    absorbed by the trend (their recommendation of ~3 x T14). Returns time, y = 1 - flat flux."""
    from wotan import flatten
    w = max(min_window, 3.0 * float(duration))
    flat = flatten(np.asarray(t, float), np.asarray(f, float), method="biweight", window_length=w, break_tolerance=0.5)
    ok = np.isfinite(flat)
    return np.asarray(t, float)[ok], 1.0 - flat[ok]


def trapezoid(dt, T, tau):
    """Unit-depth trapezoid: total duration T, ingress/egress tau (0 < tau <= T/2), centred on dt = 0."""
    a = np.abs(dt)
    tau = np.clip(tau, 1e-6, T / 2)
    return np.clip((T / 2 - a) / tau, 0.0, 1.0)


def trapezoid_grad(dt, T, tau):
    """Unit-depth trapezoid and its partial derivatives with respect to T, tau and dt (tau clipped to T/2
    exactly as in `trapezoid`)."""
    a = np.abs(dt)
    sgn = np.sign(dt)
    if tau >= T / 2:  # triangle: m = 1 - 2|dt|/T
        m = np.clip(1 - 2 * a / T, 0.0, 1.0)
        on = (m > 0) & (m < 1)
        return m, np.where(on, 2 * a / T ** 2, 0.0), np.zeros_like(m), np.where(on, -sgn * 2 / T, 0.0)
    tau = max(tau, 1e-6)
    raw = (T / 2 - a) / tau
    m = np.clip(raw, 0.0, 1.0)
    on = (raw > 0) & (raw < 1)
    return m, np.where(on, 0.5 / tau, 0.0), np.where(on, -raw / tau, 0.0), np.where(on, -sgn / tau, 0.0)


def cadence(t):
    d = np.diff(np.sort(t))
    d = d[d > 0]
    return float(np.median(d)) if d.size else np.nan


def white_noise(y):
    """Robust per-point scatter: 1.4826 x MAD of y."""
    return float(1.4826 * np.median(np.abs(y - np.median(y))))


def red_noise_beta(t, y, timescale, sigma, mask=None, max_bins=2000):
    """Time-averaging red-noise factor (Pont et al. 2006): rms of y binned on `timescale`, divided by the
    white-noise expectation sigma / sqrt(n) * sqrt(M / (M - 1)); floored at 1."""
    if mask is not None:
        t, y = t[~mask], y[~mask]
    if t.size < 10:
        return 1.0
    idx = np.floor((t - t.min()) / timescale).astype(np.int64)
    n = np.bincount(idx)
    s = np.bincount(idx, weights=y)
    good = n >= max(3, int(0.5 * np.median(n[n > 0])))
    if good.sum() < 5:
        return 1.0
    means = s[good] / n[good]
    M = good.sum()
    expected = sigma / np.sqrt(np.mean(n[good])) * np.sqrt(M / (M - 1))
    return float(max(1.0, np.std(means) / expected)) if expected > 0 else 1.0


def epochs(tmin, tmax, P, t0, T):
    n0 = int(np.ceil((tmin - T / 2 - t0) / P))
    n1 = int(np.floor((tmax + T / 2 - t0) / P))
    n = np.arange(n0, n1 + 1)
    return n, t0 + n * P


def epoch_fits(t, y, P, t0, T, tau, sigma_eff, cad):
    """Per-epoch depth with the shape fixed: d_n = sum(m y) / sum(m^2), s_n = sigma_eff / sqrt(sum m^2).
    Returns dict of arrays over all predicted epochs; `covered` marks epochs with >= MIN_COVERAGE of the
    expected in-transit cadences present. Points within [tc - T/2 - tau, tc + T/2 + tau) of epoch tc count
    for that epoch. Vectorised when epoch windows cannot overlap (T + 2 tau < P); otherwise per epoch."""
    n, tc = epochs(t.min(), t.max(), P, t0, T)
    if n.size == 0:
        z = np.zeros(0)
        return {"n": n.astype(int), "tc": z, "d": z, "s": z, "covered": z.astype(bool), "cov_frac": z}
    half = T / 2 + tau
    if 2 * half < P:
        cyc = np.floor((t - t0) / P + 0.5).astype(np.int64)
        dt = t - (t0 + cyc * P)
        sel = (dt >= -half) & (dt < half) & (cyc >= n[0]) & (cyc <= n[-1])
        k = cyc[sel] - n[0]
        m = trapezoid(dt[sel], T, tau)
        sm2 = np.bincount(k, weights=m * m, minlength=n.size)
        smy = np.bincount(k, weights=m * y[sel], minlength=n.size)
        cov = np.bincount(k, weights=(np.abs(dt[sel]) < T / 2).astype(float), minlength=n.size) * cad / T
    else:
        order = np.argsort(t)
        ts, ys = t[order], y[order]
        lo, hi = np.searchsorted(ts, tc - half), np.searchsorted(ts, tc + half)
        sm2, smy, cov = np.zeros(n.size), np.zeros(n.size), np.zeros(n.size)
        for j in range(n.size):
            dtj = ts[lo[j]:hi[j]] - tc[j]
            mj = trapezoid(dtj, T, tau)
            sm2[j], smy[j] = np.sum(mj * mj), np.sum(mj * ys[lo[j]:hi[j]])
            cov[j] = np.sum(np.abs(dtj) < T / 2) * cad / T
    ok = sm2 > 0
    d = np.where(ok, smy / np.where(ok, sm2, 1.0), np.nan)
    s = np.where(ok, sigma_eff / np.sqrt(np.where(ok, sm2, 1.0)), np.nan)
    covered = (cov >= MIN_COVERAGE) & np.isfinite(d)
    return {"n": n.astype(int), "tc": tc, "d": d, "s": s, "covered": covered, "cov_frac": cov}


def window(t, P, t0, half):
    """Boolean mask of points within +-half of an epoch of (P, t0), and their offsets / cycle numbers."""
    cyc = np.round((t - t0) / P)
    dt = t - t0 - cyc * P
    return np.abs(dt) < half, dt, cyc


def matched_filter_grid(t, y, P, t0, T, tau, dP, dt0, sigma):
    """Folded matched-filter log-likelihood ratio 0.5 * (sum m y)^2 / (sigma^2 sum m^2) on a (dP, dt0) grid,
    evaluated on points near the epochs of (P, t0)."""
    half = T / 2 + tau + np.max(np.abs(dt0)) + np.max(np.abs(dP)) * (t.max() - t.min()) / P + 1e-9
    w, dt, cyc = window(t, P, t0, half)
    dt, cyc, yy = dt[w], cyc[w], y[w]
    out = np.zeros((dP.size, dt0.size))
    for i, a in enumerate(dP):
        x = dt[:, None] - cyc[:, None] * a - dt0[None, :]
        m = trapezoid(x, T, tau)
        num = (m * yy[:, None]).sum(0)
        den = (m * m).sum(0)
        out[i] = np.where(den > 0, 0.5 * num * np.abs(num) / np.where(den > 0, den, 1) / sigma ** 2, 0.0)
    return out


def fit_trapezoid(t, y, P, t0, T, tau, sigma, free_ephemeris=True, max_dt=None):
    """Least-squares trapezoid fit (depth, T, tau[, P, t0]) to points within 1.5 T of the epochs.
    Returns parameters, 1-sigma errors (covariance scaled by sigma^2) and chi2 improvement over y = 0."""
    from scipy.optimize import least_squares
    half = 1.5 * T + (max_dt or 0)
    w, _, cyc = window(t, P, t0, half)
    tt, yy = t[w], y[w]
    if tt.size < 8:
        return None
    # with points from fewer than two transit epochs the period is unconstrained: hold it fixed (a free
    # parameter with a zero Jacobian column would otherwise take an arbitrary step under x_scale="jac")
    fix_P = free_ephemeris and np.unique(cyc[w]).size < 2
    m0 = trapezoid(_dt(tt, P, t0), T, tau)
    d0 = float(np.sum(m0 * yy) / max(np.sum(m0 * m0), 1e-12))

    def unpack(p):
        if fix_P:
            return np.r_[p[:3], P, p[3]]
        if free_ephemeris:
            return p
        return np.r_[p, P, t0]

    def resid(p):
        dep, TT, ta, PP, tt0 = unpack(p)
        return (yy - dep * trapezoid(_dt(tt, PP, tt0), TT, ta)) / sigma

    p0 = [d0, T, min(tau, 0.49 * T)]
    lb, ub = [-np.inf, 0.2 * T, 1e-4], [np.inf, 3.0 * T, 1.5 * T]
    if fix_P:
        p0 += [t0]
        lb += [t0 - 0.5 * T]
        ub += [t0 + 0.5 * T]
    elif free_ephemeris:
        span = tt.max() - tt.min()
        p0 += [P, t0]
        lb += [P - 0.5 * T * P / max(span, P), t0 - 0.5 * T]
        ub += [P + 0.5 * T * P / max(span, P), t0 + 0.5 * T]
    p0 = np.clip(p0, np.array(lb) + 1e-12, np.array(ub) - 1e-12)
    def jac(p):
        dep, TT, ta, PP, tt0 = unpack(p)
        cyc = np.round((tt - tt0) / PP)
        m, dT, dta, ddt = trapezoid_grad(tt - tt0 - cyc * PP, TT, ta)
        cols = [-m, -dep * dT, -dep * dta]
        if fix_P:
            cols += [dep * ddt]                   # d(dt)/d(t0) = -1
        elif free_ephemeris:
            cols += [dep * ddt * cyc, dep * ddt]  # d(dt)/dP = -cyc
        return np.vstack(cols).T / sigma

    r = least_squares(resid, p0, jac=jac, bounds=(lb, ub), x_scale="jac")
    dep, TT, ta, PP, tt0 = unpack(r.x)
    ta = min(ta, TT / 2)
    J = r.jac
    try:
        cov = np.linalg.pinv(J.T @ J)
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        err = np.full(len(r.x), np.nan)
    chi2_0 = float(np.sum((yy / sigma) ** 2))
    chi2 = float(np.sum(r.fun ** 2))
    names = ["depth", "T", "tau", "t0"] if fix_P else ["depth", "T", "tau", "P", "t0"][:len(r.x)]
    return {"depth": float(dep), "T": float(TT), "tau": float(ta), "P": float(PP), "t0": float(tt0),
            "err": dict(zip(names, map(float, err))), "dchi2": chi2_0 - chi2, "n_points": int(tt.size)}


def _dt(t, P, t0):
    return t - t0 - np.round((t - t0) / P) * P
