"""Fully synthetic TESS-like light curves for resolver development and unit tests (NOT the B1 benchmark,
which injects into real pool-star light curves; docs/injection_design.md).

Sampling: 2-min cadence, sectors of two 13.2-d orbits separated by a 1-d downlink gap, 1-6 sectors, consecutive
or with a gap of 1-10 sectors. Noise: white (log-uniform 200-3000 ppm) + AR(1) correlated noise (correlation time
0.05-0.5 d, amplitude 0-0.5 x white) + a sinusoidal stellar trend (period 2-15 d, amplitude 0-1 x white).
Signals follow the B1 class mix and parameter distributions (docs/injection_design.md §2) with the injection
models of exogargantua.inject. Stellar density is reported with a 15 % lognormal error, as a TIC value would be.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from .. import inject as I
from .hypotheses import ALIAS_SET

CLASSES = (("planet", 0.60), ("eb_twin", 0.15), ("eb_unequal", 0.15), ("eb_ecc", 0.10))
CAD = 2.0 / 1440.0


def sampling(rng):
    n_sec = int(rng.integers(1, 7))
    starts, t = [], 0.0
    for i in range(n_sec):
        if i > 0 and rng.uniform() < 0.3:
            t += 27.4 * int(rng.integers(1, 11))
        starts.append(t)
        t += 27.4
    seg = []
    for s in starts:
        seg.append(np.arange(s + 0.2, s + 13.4, CAD))
        seg.append(np.arange(s + 14.4, s + 27.2, CAD))
    return np.concatenate(seg), n_sec


def noise(rng, t):
    sw = float(np.exp(rng.uniform(np.log(200e-6), np.log(3000e-6))))
    tc = float(rng.uniform(0.05, 0.5))
    ar = float(rng.uniform(0, 0.5)) * sw
    phi = np.exp(-CAD / tc)
    from scipy.signal import lfilter
    e = rng.normal(0, 1, t.size)
    red = lfilter([np.sqrt(1 - phi ** 2)], [1, -phi], e)  # AR(1) across samples (gaps not special-cased)
    trendP = float(rng.uniform(2, 15))
    trendA = float(rng.uniform(0, 1)) * sw
    trend = trendA * np.sin(2 * np.pi * t / trendP + rng.uniform(0, 2 * np.pi))
    return rng.normal(0, sw, t.size) + ar * red + trend, {"sigma_white": sw, "red_amp": ar, "red_tc": tc}


def star(rng):
    m = float(rng.uniform(0.5, 1.4))
    r = float(m ** 0.8)
    teff = float(5772 * m ** 0.55)
    logg = float(4.438 + np.log10(m) - 2 * np.log10(r))
    rho = 1.41 * m / r ** 3
    rho_obs = rho * float(np.exp(rng.normal(0, 0.15)))
    return {"m": m, "r": r, "teff": teff, "logg": logg, "rho_true": rho, "rho": rho_obs, "rho_err": 0.15 * rho_obs}


def make_target(rng, seed_r=None, cls=None, p_none=0.0):
    """One synthetic target. cls 'none' (probability p_none when cls is not given): no signal at all, for the
    abstention check; its P_true is only used to draw a seed period."""
    t, n_sec = sampling(rng)
    st = star(rng)
    u = I.limb_darkening(st["teff"], st["logg"])
    if cls is None:
        cls = "none" if rng.uniform() < p_none else str(rng.choice([c for c, _ in CLASSES], p=[w for _, w in CLASSES]))
    P = float(np.exp(rng.uniform(np.log(0.5), np.log(30.0))))
    depth = float(np.exp(rng.uniform(np.log(200e-6), np.log(0.05))))
    b = float(rng.uniform(0, 0.95))
    t0 = float(t.min() + rng.uniform(0, P))
    info = {}
    if cls == "none":
        sig, depth = np.ones_like(t), 0.0
    elif cls == "planet":
        sig, info = I.planet_flux(t, P, t0, depth, b, st["r"], st["m"], u)
    elif cls == "eb_twin":
        sig, info = I.eb_flux(t, P, t0, depth, depth, b, st["r"], st["m"], u)
    elif cls == "eb_unequal":
        sig, info = I.eb_flux(t, P, t0, depth, depth * float(rng.uniform(0.05, 0.8)), b, st["r"], st["m"], u)
    else:
        e, w = float(rng.uniform(0.05, 0.5)), float(rng.uniform(0, 2 * np.pi))
        sig, info = I.eb_flux(t, P, t0, depth, depth * float(rng.uniform(0.3, 1.0)), b, st["r"], st["m"], u, ecc=e, omega=w)
        info.update({"ecc": e, "omega": w})
    nz, ninfo = noise(rng, t)
    f = sig * (1.0 + nz)
    n_in = int(np.sum(1.0 - sig > 0.5 * depth)) if depth > 0 else 0
    snr_true = float(depth / ninfo["sigma_white"] * np.sqrt(n_in)) if n_in else 0.0
    r = Fraction(seed_r) if seed_r is not None else ALIAS_SET[int(rng.integers(len(ALIAS_SET)))]
    return {"t": t, "f": f, "P_true": P, "t0_true": t0, "depth": depth, "b": b, "cls": cls, "n_sectors": n_sec,
            "seed_r": str(r), "P_seed": float(r) * P, "star": st, "noise": ninfo, "snr_true": snr_true, "n_in": n_in,
            "info": {k: float(v) for k, v in info.items()}}
