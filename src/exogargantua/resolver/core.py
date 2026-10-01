"""resolve(): seed period -> evidence for every alias hypothesis -> principled posterior (combine.principled).

Input: raw normalised light curve (time [d], flux ~ 1, NaN-free) and a seed period P0; optional seed epoch and
duration (estimated from the data when absent) and stellar density (g cm^-3) with its 1-sigma error.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import combine as C, features as F, hypotheses as H, model as M

FEATURES = ("F1_coverage", "F2_oddeven", "F3_secondary", "F4_density", "F5_coherence", "F6_shape")


@dataclass
class Star:
    rho: float = np.nan        # g cm^-3
    rho_err: float = np.nan


def resolve(t, f, P0, t0=None, T=None, star: Star | None = None, abstain=C.DEFAULT_ABSTAIN, alias_set=H.ALIAS_SET,
            detrended=False):
    t_start = time.time()
    star = star or Star()
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    if detrended:
        tt, y = t, 1.0 - f
    else:
        tt, y = M.detrend(t, f, duration=T if T is not None else 0.1)
    sig0 = M.white_noise(y)
    t0s, Ts = H.seed_ephemeris(tt, y, P0, sig0, t0, T)
    if not detrended and 3 * Ts > 0.75 and (T is None or abs(Ts - T) > 1e-9):
        tt, y = M.detrend(t, f, duration=Ts)
    cad = M.cadence(tt)
    # noise: mask the densest candidate grid (P0/5) around the seed epoch unless that removes > 80 % of the data
    msk, _, _ = M.window(tt, P0 / 5, t0s, Ts)
    if msk.mean() > 0.8:
        msk = np.zeros_like(msk)
    sigma_w = M.white_noise(y[~msk])
    beta = M.red_noise_beta(tt, y, max(Ts, 3 * cad), sigma_w, mask=msk)
    sigma = sigma_w * beta
    dens = F.DensityPrior(star.rho, star.rho_err)
    hyps = H.enumerate_hypotheses(P0, t0s, Ts, alias_set)
    for h in hyps:
        H.refine(h, tt, y, sigma)
        evaluate(h, tt, y, sigma, cad, dens)
    post = C.principled(hyps)
    summary = C.summarise(hyps, post, abstain)
    summary.update({"P0": float(P0), "t0_seed": t0s, "T_seed": Ts, "sigma_w": sigma_w, "beta": beta, "cadence": cad,
                    "n_points": int(tt.size), "runtime_s": time.time() - t_start})
    return summary, hyps


def evaluate(h, t, y, sigma, cad, dens):
    ef = M.epoch_fits(t, y, h.P, h.t0, h.T, h.tau, sigma, cad)
    F1, F2, raw = F.coverage_and_oddeven(ef)
    dp = raw["depth_all"] if np.isfinite(raw["depth_all"]) else 0.0
    w, dt, _ = M.window(t, h.P, h.t0, h.T / 2 + h.tau)
    resid = y - max(dp, 0.0) * np.where(w, M.trapezoid(dt, h.T, h.tau), 0.0)
    F3, rs = F.secondary(t, resid, h.P, h.t0, h.T, sigma, dp, cad)
    err = (h.fit or {}).get("err", {})
    sT = max(err.get("T", np.nan) if np.isfinite(err.get("T", np.nan)) else h.T, cad)
    stau = max(err.get("tau", np.nan) if np.isfinite(err.get("tau", np.nan)) else h.T / 2, cad / 2)
    F4, rd = F.density(dens, h.P, h.T, h.tau, sT, stau, dp)
    c = ef["covered"]
    events = [(ef["tc"][c], ef["d"][c], ef["s"][c], h.T, h.tau)]
    if rs["sec_detected"]:
        T2 = rs["sec_dur_ratio"] * h.T
        tau2 = (rs["sec_tau_over_T"] if np.isfinite(rs["sec_tau_over_T"]) else 0.25) * T2
        efs = M.epoch_fits(t, resid, h.P, h.t0 + rs["sec_phase"] * h.P, T2, tau2, sigma, cad)
        cs = efs["covered"]
        events.append((efs["tc"][cs], efs["d"][cs], efs["s"][cs], T2, tau2))
    F5, rt = F.coherence(t, y, events, h.T, h.tau, cad)
    F6, rsh = F.shape(t, y, h.P, h.t0, h.T, h.tau, sigma)
    h.features = dict(zip(FEATURES, (F1, F2, F3, F4, F5, F6)))
    h.raw = {**raw, **rs, **rd, **rt, **rsh, "P": h.P, "T": h.T, "tau_over_T": h.tau / h.T if h.T > 0 else np.nan,
             "depth": dp, "fit_dchi2": (h.fit or {}).get("dchi2", np.nan)}
    return h
