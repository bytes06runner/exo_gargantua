"""Alias hypotheses around a seed period and their local refinement (brief §5).

A hypothesis is (r, j): period r * P0 with r = a/b from the alias set, and transit epochs
t0 + j * P0 + m * r * P0. Every hypothesis keeps the seed's reference transit at t0 + j * P0, and
j = 0..a-1 enumerates the distinct ephemerides of that period (e.g. for r = 2 the transits are either
the seed's even or its odd events).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from . import model as M

ALIAS_SET = (Fraction(1, 5), Fraction(1, 4), Fraction(1, 3), Fraction(1, 2), Fraction(2, 3), Fraction(1, 1),
             Fraction(3, 2), Fraction(2, 1), Fraction(3, 1), Fraction(4, 1), Fraction(5, 1))
SEED_DURATIONS_H = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0])


@dataclass
class Hypothesis:
    r: Fraction
    j: int
    P: float
    t0: float
    T: float
    tau: float
    fit: dict | None = None
    features: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @property
    def key(self):
        return f"{self.r.numerator}/{self.r.denominator}:{self.j}"


def seed_ephemeris(t, y, P0, sigma, t0=None, T=None):
    """Epoch and duration at the seed period when the seed gives only a period: box-shaped matched filter over
    phase (step = duration/8) and durations SEED_DURATIONS_H (< 0.25 P0). Given t0 and T are kept."""
    if t0 is not None and T is not None:
        return float(t0), float(T)
    durs = SEED_DURATIONS_H / 24.0
    durs = durs[durs < 0.25 * P0] if np.any(durs < 0.25 * P0) else durs[:1]
    if T is not None:
        durs = np.array([T])
    ph = (t - t.min()) % P0
    best = (-np.inf, None, None)
    for D in durs:
        nb = max(8, int(np.ceil(P0 / (D / 8))))
        idx = np.minimum((ph / P0 * nb).astype(int), nb - 1)
        s = np.bincount(idx, weights=y, minlength=nb)
        c = np.bincount(idx, minlength=nb).astype(float)
        k = max(1, int(round(D / (P0 / nb))))
        s2, c2 = np.r_[s, s[:k]], np.r_[c, c[:k]]
        cs, cc = np.cumsum(np.r_[0, s2]), np.cumsum(np.r_[0, c2])
        S = cs[k:k + nb] - cs[:nb]
        C = cc[k:k + nb] - cc[:nb]
        llr = np.where(C > 0, 0.5 * S * np.abs(S) / np.maximum(C, 1) / sigma ** 2, -np.inf)
        i = int(np.argmax(llr))
        if llr[i] > best[0]:
            best = (llr[i], t.min() + (i + k / 2) * P0 / nb, D)
    tc = best[1] if t0 is None else t0
    return float(tc), float(best[2])


def enumerate_hypotheses(P0, t0, T, alias_set=ALIAS_SET):
    out = []
    for r in alias_set:
        for j in range(r.numerator):
            out.append(Hypothesis(r=r, j=j, P=float(r) * P0, t0=t0 + j * P0, T=T, tau=T / 4))
    return out


TIE_NATS = 0.5  # grid points within this log-likelihood of the best are statistically indistinguishable


def pick_near_seed(g, dP, dt0, tol=TIE_NATS):
    """Among grid points within `tol` of the best log-likelihood, the one closest to the seed-implied
    ephemeris (smallest |dP|, then smallest |dt0|). When the data do not constrain the period (one transit
    in the data), the refinement therefore leaves the period at r x P0 instead of a grid edge."""
    ok = g >= g.max() - tol
    ii, kk = np.nonzero(ok)
    o = np.lexsort((np.abs(dt0[kk]), np.abs(dP[ii])))
    return int(ii[o[0]]), int(kk[o[0]])


def refine(h, t, y, sigma, n_steps=17):
    """Local refinement of (P, t0): the period within +-T*P/span (phase drift <= T over the baseline) and the
    epoch within +-T/2, maximising the folded matched-filter likelihood ratio; then a least-squares trapezoid
    fit with free depth, T, tau, P, t0."""
    span = t.max() - t.min()
    dPmax = h.T * h.P / max(span, h.P)
    dP = np.linspace(-dPmax, dPmax, n_steps)
    dt0 = np.linspace(-h.T / 2, h.T / 2, n_steps)
    g = M.matched_filter_grid(t, y, h.P, h.t0, h.T, h.tau, dP, dt0, sigma)
    i, k = pick_near_seed(g, dP, dt0)
    h.P, h.t0 = h.P + dP[i], h.t0 + dt0[k]
    # keep t0 near the middle of the data so P and t0 decorrelate in the fit
    mid = 0.5 * (t.min() + t.max())
    h.t0 = h.t0 + np.round((mid - h.t0) / h.P) * h.P
    fit = M.fit_trapezoid(t, y, h.P, h.t0, h.T, h.tau, sigma, free_ephemeris=True)
    if fit is not None and fit["depth"] > 0 and np.isfinite(fit["T"]):
        h.P, h.t0, h.T, h.tau = fit["P"], fit["t0"], fit["T"], fit["tau"]
    h.fit = fit
    return h
