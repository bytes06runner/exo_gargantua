"""Injection signal models for B1 (docs/injection_design.md).

- Quadratic TESS limb darkening from Claret (2017): ATLAS table 25 (Teff >= 3500 K) or
  PHOENIX-COND table 15 (Teff < 3500 K), solar metallicity, xi = 2 km/s, LSM coefficients,
  bilinear interpolation in (Teff, logg) on the table grid (clamped at the grid edges).
- Planet: batman transit, circular orbit, a/R* from Kepler's third law.
- EB: each eclipse drawn as a batman transit shape (primary at phase 0; secondary at the phase and
  duration implied by e, omega), depths set explicitly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
G = 6.67430e-11
R_SUN, M_SUN = 6.957e8, 1.98847e30


@lru_cache(maxsize=1)
def _ld_tables():
    raw = ROOT / "data" / "raw"
    out = {}
    for key, pat in (("atlas", "claret2017_tess_quadratic_atlas_table25_*.dat"),
                     ("phoenix", "claret2017_tess_quadratic_phoenix_table15_*.dat")):
        rows = []
        for line in sorted(raw.glob(pat))[0].read_text().splitlines():
            p = line.split()
            if len(p) < 6:
                continue
            logg, teff, z, xi, a, b = map(float, p[:6])
            if abs(z) < 1e-9 and abs(xi - 2.0) < 1e-9:
                rows.append((teff, logg, a, b))
        out[key] = np.array(rows)
    return out


def limb_darkening(teff: float, logg: float) -> tuple[float, float]:
    tab = _ld_tables()["phoenix" if teff < 3500 else "atlas"]
    teffs, loggs = np.unique(tab[:, 0]), np.unique(tab[:, 1])
    teff = float(np.clip(teff, teffs.min(), teffs.max()))
    logg = float(np.clip(logg, loggs.min(), loggs.max()))

    def at(t, g):
        m = (tab[:, 0] == t) & (tab[:, 1] == g)
        if not m.any():  # grid hole: nearest available point in logg at this Teff
            cand = tab[tab[:, 0] == t]
            r = cand[np.argmin(np.abs(cand[:, 1] - g))]
            return r[2], r[3]
        return tab[m][0, 2], tab[m][0, 3]

    i = np.searchsorted(teffs, teff).clip(1, len(teffs) - 1)
    j = np.searchsorted(loggs, logg).clip(1, len(loggs) - 1)
    t0, t1, g0, g1 = teffs[i - 1], teffs[i], loggs[j - 1], loggs[j]
    wt = 0.0 if t1 == t0 else (teff - t0) / (t1 - t0)
    wg = 0.0 if g1 == g0 else (logg - g0) / (g1 - g0)
    c = [np.array(at(t, g)) for t, g in ((t0, g0), (t1, g0), (t0, g1), (t1, g1))]
    u = (1 - wt) * (1 - wg) * c[0] + wt * (1 - wg) * c[1] + (1 - wt) * wg * c[2] + wt * wg * c[3]
    return float(u[0]), float(u[1])


def a_over_rstar(period_d: float, m_star: float, r_star: float) -> float:
    a = (G * m_star * M_SUN * (period_d * 86400.0) ** 2 / (4 * np.pi ** 2)) ** (1 / 3)
    return a / (r_star * R_SUN)


def _transit(time, period, t0, rp, a_rs, inc_deg, u, ecc=0.0, w=90.0):
    import batman
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = t0, period, rp, a_rs, inc_deg, ecc, w
    p.u, p.limb_dark = list(u), "quadratic"
    return batman.TransitModel(p, np.asarray(time, float)).light_curve(p)


def rp_for_depth(depth: float, b: float, u) -> float:
    """Rp/R* (bisection) such that the limb-darkened model depth at mid-transit, at impact parameter b, equals `depth`."""
    t = np.array([0.0])
    lo, hi = 1e-4, 0.9
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        a_rs = 20.0
        inc = np.degrees(np.arccos(min(b / a_rs, 1.0)))
        d = 1.0 - _transit(t, 10.0, 0.0, mid, a_rs, inc, u)[0]
        lo, hi = (mid, hi) if d < depth else (lo, mid)
    return 0.5 * (lo + hi)


def planet_flux(time, period, t0, depth, b, r_star, m_star, u):
    a_rs = a_over_rstar(period, m_star, r_star)
    b = min(b, 0.95)
    inc = np.degrees(np.arccos(b / a_rs))
    rp = rp_for_depth(depth, b, u)
    return _transit(time, period, t0, rp, a_rs, inc, u), {"a_rs": a_rs, "inc": inc, "rp_rs": rp}


def eb_flux(time, period, t0, depth1, depth2, b, r_star, m_star, u, ecc=0.0, omega=np.pi / 2):
    """Two eclipses per orbit. Secondary phase and duration ratio from (ecc, omega) (Winn 2010 eqs.)."""
    a_rs = a_over_rstar(period, m_star, r_star)
    inc = np.degrees(np.arccos(min(b, 0.95) / a_rs))
    # secondary phase offset (Sterne 1940 / Winn 2010 eq. 33, first order in e)
    dphi = 0.5 + (2 / np.pi) * ecc * np.cos(omega)
    dur_ratio = (1 + ecc * np.sin(omega)) / (1 - ecc * np.sin(omega)) if ecc > 0 else 1.0
    f1 = _transit(time, period, t0, rp_for_depth(depth1, b, u), a_rs, inc, u)
    # secondary: same shape family, duration scaled by dur_ratio through a/R* (approximation: the
    # impact parameter of the secondary scales as b / dur_ratio), centred at t0 + dphi * P
    f2 = _transit(time, period, t0 + dphi * period, rp_for_depth(depth2, b, u), a_rs / dur_ratio, inc, u)
    return 1.0 - (1.0 - f1) - (1.0 - f2), {"a_rs": a_rs, "inc": inc, "sec_phase": dphi, "sec_dur_ratio": dur_ratio}
