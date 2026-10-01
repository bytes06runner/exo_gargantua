"""The six per-hypothesis evidence features (brief §5; design notes in docs/resolver_design.md).

Each feature returns a log-likelihood term (natural log, used by the principled combiner) and raw
descriptors (used by the learned combiner). Terms are comparable across hypotheses of one target because
each is a log-ratio against a reference that does not depend on the hypothesis.
"""

from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr

from . import model as M

G_SI = 6.67430e-11
SEC_PHASE_SIGMA = 0.01      # width of the near-0.5 component of the secondary phase prior
P_NO_SECONDARY = 0.5        # prior probability of no secondary event
TIMING_ALT_FRAC = 0.25      # incoherent-ephemeris alternative: timing scatter of T/4
MIN_EPOCH_SNR_TIMING = 3.0
MIN_SEC_SNR = 7.0           # secondary counted as detected (for timing and shape descriptors)
N_DENSITY_DRAWS = 4000
MODEL_JITTER = 0.05         # event-to-event depth variability / model inadequacy, as a fraction of the depth
TIMING_JITTER = 0.01        # timing floor, as a fraction of the duration
SHAPE_JITTER = 0.05         # floor on T (fraction of T) and on tau/T (absolute) in F4 and F6


def jittered(d, s, eps=MODEL_JITTER):
    """Per-epoch depth errors with a model-error term in quadrature: s' = sqrt(s^2 + (eps * max(d, 0))^2).
    The term scales with each epoch's own measured depth, so it is the same for every hypothesis that predicts
    that epoch, and epochs where the data are flat keep their full precision (the missing-transit evidence)."""
    return np.sqrt(s ** 2 + (eps * np.maximum(d, 0.0)) ** 2)


# ---------------------------------------------------------------- F1 + F2: per-epoch depths
def coverage_and_oddeven(ef):
    """F1 (predicted-epoch coverage) and F2 (odd/even consistency) as an exact split of the per-epoch
    log-likelihood ratio (depth errors include the MODEL_JITTER term) of 'transits at every predicted epoch, common depth' against 'no transits':
        l_common = 0.5 * max(S, 0)^2 / W,  S = sum d/s^2, W = sum 1/s^2   (depth >= 0)
        l_parity = sum over parities g of 0.5 * max(S_g, 0)^2 / W_g
        F1 = l_parity,  F2 = l_common - l_parity  (<= 0; -2 F2 = odd/even chi^2 when unconstrained)."""
    c = ef["covered"]
    d, s, n = ef["d"][c], ef["s"][c], ef["n"][c]
    s = jittered(d, s)
    raw = {"n_pred": int(ef["n"].size), "n_cov": int(c.sum())}
    if c.sum() == 0:
        raw.update({"cov_frac": 0.0, "n_missing": 0, "depth_all": np.nan, "snr_all": 0.0, "oe_sigma": np.nan,
                    "depth_odd": np.nan, "depth_even": np.nan})
        return 0.0, 0.0, raw
    w = 1.0 / s ** 2
    S, W = float(np.sum(d * w)), float(np.sum(w))
    l_common = 0.5 * max(S, 0.0) ** 2 / W
    l_par, dep = 0.0, {}
    for g in (0, 1):
        m = (n % 2) == g
        if m.any():
            Sg, Wg = float(np.sum(d[m] * w[m])), float(np.sum(w[m]))
            l_par += 0.5 * max(Sg, 0.0) ** 2 / Wg
            dep[g] = (Sg / Wg, 1 / np.sqrt(Wg))
    dhat = S / W
    missing = int(np.sum((dhat - d) / s > 3.0)) if dhat > 0 else 0
    F2 = l_common - l_par
    raw.update({"cov_frac": float(c.sum() / max(ef["n"].size, 1)), "n_missing": missing, "depth_all": dhat,
                "snr_all": float(S / np.sqrt(W)), "oe_sigma": float(np.sqrt(max(-2 * F2, 0.0))) if len(dep) == 2 else np.nan,
                "depth_odd": dep.get(1, (np.nan,))[0], "depth_even": dep.get(0, (np.nan,))[0]})
    return float(l_par), float(F2), raw


# ---------------------------------------------------------------- F3: secondary event at any phase
def _log_phi_diff(a, b):
    """log(Phi(a) - Phi(b)) for a > b, stable."""
    la, lb = log_ndtr(a), log_ndtr(b)
    return la + np.log1p(-np.exp(np.minimum(lb - la, -1e-300)))


def secondary(t, resid, P, t0, T, sigma, depth_primary, cad):
    """F3. Locate the strongest secondary event: box search of the primary-subtracted, folded light curve over
    all phases outside the primary window, durations {T/2, T, 2T} (< P/3), refined by a trapezoid fit (free
    depth, duration, ingress, phase). Evidence: Bayes factor of 'secondary at that ephemeris' against 'none',
    from its per-epoch depths (MODEL_JITTER included, as for the primary):
        log BF = 0.5 max(S,0)^2 / W                                  (profile over a common depth)
                 + log(sqrt(2 pi) sigma_D / dmax) + log(Phi-window)   (depth prior uniform on [0, dmax])
                 + log(pi(phase) * D / P)                              (phase prior x resolution element)
    phase prior 1/2 uniform + 1/2 N(0.5, 0.01) wrapped; dmax = max(2 x primary depth, 10 sigma_D).
    F3 = log(P_NO_SECONDARY + (1 - P_NO_SECONDARY) BF)."""
    ph = ((t - t0) / P) % 1.0
    o = np.argsort(ph)
    ph, r = ph[o], resid[o]
    best = {"snr": 0.0, "phase": np.nan, "depth": np.nan, "dur": np.nan, "sigma": np.nan}
    for D in (0.5 * T, T, 2 * T):
        if D >= P / 3:
            continue
        w = D / P
        centres = np.arange(0.0, 1.0, w / 4)
        ph2, r2 = np.r_[ph - 1.0, ph, ph + 1.0], np.r_[r, r, r]
        cs = np.r_[0.0, np.cumsum(r2)]
        lo, hi = np.searchsorted(ph2, centres - w / 2), np.searchsorted(ph2, centres + w / 2)
        N = (hi - lo).astype(float)
        dist = np.minimum(centres, 1 - centres)
        ok = (N >= 3) & (dist > (T / P) * 0.5 + w / 2)
        if not ok.any():
            continue
        snr = np.where(ok, (cs[hi] - cs[lo]) / np.maximum(N, 1) / (sigma / np.sqrt(np.maximum(N, 1))), -np.inf)
        i = int(np.argmax(snr))
        if snr[i] > best["snr"]:
            best = {"snr": float(snr[i]), "phase": float(centres[i]), "dur": float(D), "tau": D / 4,
                    "depth": float((cs[hi[i]] - cs[lo[i]]) / N[i]), "sigma": float(sigma / np.sqrt(N[i]))}
    if not np.isfinite(best["phase"]):
        return float(np.log(P_NO_SECONDARY)), _sec_raw(best, depth_primary, T)
    if best["snr"] >= 3:
        f = M.fit_trapezoid(t, resid, P, t0 + best["phase"] * P, best["dur"], best["dur"] / 4, sigma, free_ephemeris=True)
        if f is not None and f["depth"] > 0 and f["dchi2"] > best["snr"] ** 2:
            best.update({"phase": float(((f["t0"] - t0) / P) % 1.0), "dur": f["T"], "tau": f["tau"], "tau_over_T": f["tau"] / f["T"]})
    ef = M.epoch_fits(t, resid, P, t0 + best["phase"] * P, best["dur"], best["tau"], sigma, cad)
    c = ef["covered"]
    if not c.any():
        return float(np.log(P_NO_SECONDARY)), _sec_raw(best, depth_primary, T)
    d, sj = ef["d"][c], jittered(ef["d"][c], ef["s"][c])
    S, W = float(np.sum(d / sj ** 2)), float(np.sum(1 / sj ** 2))
    dh, sD = S / W, 1 / np.sqrt(W)
    dmax = max(2.0 * max(depth_primary, 0.0), 10.0 * sD)
    lbf = (0.5 * max(S, 0.0) ** 2 / W + np.log(sD * np.sqrt(2 * np.pi) / dmax)
           + _log_phi_diff((dmax - max(dh, 0.0)) / sD, -max(dh, 0.0) / sD))
    dd = min(abs(best["phase"] - 0.5), 1 - abs(best["phase"] - 0.5))
    prior = 0.5 + 0.5 * np.exp(-0.5 * (dd / SEC_PHASE_SIGMA) ** 2) / (SEC_PHASE_SIGMA * np.sqrt(2 * np.pi))
    lbf += np.log(prior * min(best["dur"] / P, 1.0))
    best.update({"snr": float(dh / sD), "depth": float(dh), "sigma": float(sD)})
    F3 = float(np.logaddexp(np.log(P_NO_SECONDARY), np.log(1 - P_NO_SECONDARY) + lbf))
    return F3, _sec_raw(best, depth_primary, T)


def _sec_raw(b, dp, T):
    ratio = b["depth"] / dp if dp and dp > 0 and np.isfinite(b["depth"]) else np.nan
    det = b["snr"] >= MIN_SEC_SNR
    twin = bool(det and np.isfinite(ratio) and abs(b["depth"] - dp) < 3 * np.hypot(b["sigma"], dp / max(b["snr"], 1e-9))
                and abs(b["phase"] - 0.5) < 0.02 and abs(b["dur"] - T) < 0.5 * T)
    return {"sec_snr": b["snr"], "sec_phase": b["phase"], "sec_depth_ratio": ratio, "sec_tau_over_T": b.get("tau_over_T", np.nan),
            "sec_phase_off05": abs(b["phase"] - 0.5) if np.isfinite(b["phase"]) else np.nan,
            "sec_dur_ratio": b["dur"] / T if np.isfinite(b["dur"]) else np.nan, "sec_detected": bool(det), "twin": twin}


# ---------------------------------------------------------------- F4: stellar density
class DensityPrior:
    """Monte Carlo draws of stellar density (lognormal around the TIC value), impact parameter
    (uniform on [0, 1 + k]) and eccentricity (Beta(0.867, 3.03), Kipping 2013) with uniform omega,
    shared by all hypotheses of one target (common random numbers, fixed seed)."""

    def __init__(self, rho_cgs, rho_err_cgs, seed=20260930, n=N_DENSITY_DRAWS):
        self.ok = bool(np.isfinite(rho_cgs) and rho_cgs > 0)
        rng = np.random.default_rng(seed)
        sig = 0.1 if not np.isfinite(rho_err_cgs) or not self.ok else max(0.1, rho_err_cgs / rho_cgs)
        self.rho = (rho_cgs if self.ok else 1.0) * np.exp(rng.normal(0, sig, n)) * 1000.0  # kg m^-3
        self.u = rng.uniform(0, 1, n)
        self.e = rng.beta(0.867, 3.03, n)
        self.w = rng.uniform(0, 2 * np.pi, n)

    def durations(self, P_days, k):
        P = P_days * 86400.0
        aR = (G_SI * self.rho * P ** 2 / (3 * np.pi)) ** (1 / 3)
        b = self.u * (1 + k)
        esw = 1 + self.e * np.sin(self.w)
        eta = np.sqrt(1 - self.e ** 2) / esw                 # duration factor (Winn 2010 eq. 16)
        cosi = b / (aR * (1 - self.e ** 2) / esw)            # b = (a/R*) cos i (1 - e^2) / (1 + e sin w)
        ok = (np.abs(cosi) < 1) & (aR * (1 - self.e) > 1 + k)   # transits, and periastron outside the star
        sini = np.sqrt(np.clip(1 - cosi ** 2, 1e-12, 1))
        arg14 = np.clip(np.sqrt(np.clip((1 + k) ** 2 - b ** 2, 0, None)) / (aR * sini), 0, 1)
        arg23 = np.clip(np.sqrt(np.clip((1 - k) ** 2 - b ** 2, 0, None)) / (aR * sini), 0, 1)
        T14 = P_days / np.pi * np.arcsin(arg14) * eta
        T23 = P_days / np.pi * np.arcsin(arg23) * eta
        return T14, 0.5 * (T14 - T23), ok


def density(prior, P, T, tau, sT, stau, depth):
    """F4: log p(T_obs, tau_obs | P, star) under the forward model, Gaussian errors on the fitted T and tau
    (Seager & Mallen-Ornelas 2003; Kipping 2010 for the eccentric duration)."""
    if not prior.ok or not (np.isfinite(T) and np.isfinite(tau)):
        return 0.0, {"log_rho_ratio": np.nan}
    k = float(np.clip(np.sqrt(max(depth, 1e-8)), 0.005, 0.999))
    sT = float(np.hypot(sT, SHAPE_JITTER * T))
    stau = float(np.hypot(stau, SHAPE_JITTER * T))
    T14, taup, ok = prior.durations(P, k)
    ll = -0.5 * ((T - T14) / sT) ** 2 - 0.5 * ((tau - taup) / stau) ** 2 - np.log(2 * np.pi * sT * stau)
    ll = np.where(ok, ll, -np.inf)
    F4 = float(np.logaddexp.reduce(ll) - np.log(ll.size)) if np.isfinite(ll).any() else -700.0
    # descriptor: density implied by (P, T) for b = 0, circular, against the TIC value
    rho_impl = 3 * np.pi / (G_SI * (P * 86400) ** 2) * (P / (np.pi * T) * (1 + k)) ** 3 / 1000.0 if T > 0 else np.nan
    return max(F4, -700.0), {"log_rho_ratio": float(np.log(rho_impl / np.median(prior.rho / 1000.0))) if np.isfinite(rho_impl) else np.nan}


# ---------------------------------------------------------------- F5: ephemeris coherence
def epoch_times(t, y, tc, T, tau, half_range, n_grid=61):
    """Per-epoch mid-time by matched filter over +-half_range (depth free, positive), parabolic refinement."""
    order = np.argsort(t)
    ts, ys = t[order], y[order]
    grid = np.linspace(-half_range, half_range, n_grid)
    out = np.full(tc.size, np.nan)
    for i, c in enumerate(tc):
        lo, hi = np.searchsorted(ts, c - half_range - T), np.searchsorted(ts, c + half_range + T)
        dt = ts[lo:hi] - c
        if dt.size < 5:
            continue
        m = M.trapezoid(dt[:, None] - grid[None, :], T, tau)
        S = (m * ys[lo:hi, None]).sum(0)
        den = (m * m).sum(0)
        q = np.where(den > 0, np.maximum(S, 0) ** 2 / np.where(den > 0, den, 1), 0)
        j = int(np.argmax(q))
        if 0 < j < n_grid - 1:
            a, b_, c_ = q[j - 1], q[j], q[j + 1]
            den2 = a - 2 * b_ + c_
            off = 0.5 * (a - c_) / den2 if den2 != 0 else 0.0
            out[i] = c + grid[j] + np.clip(off, -1, 1) * (grid[1] - grid[0])
        else:
            out[i] = c + grid[j]
    return out


def coherence(t, y, events, T, tau, cad):
    """F5: per event family (primaries; detected secondaries), mid-times of events with observed depth SNR >= 3,
    a weighted linear ephemeris fitted to them, and for the residuals
        sum log N(res; 0, sigma_t) - log N(res; 0, sqrt(sigma_t^2 + (T/4)^2)),
    i.e. a coherent linear ephemeris against an incoherent one. sigma_t from Carter et al. (2008):
    sigma_t = T / Q * sqrt(tau / (2 T)), Q = d / s, floored at cadence / 2."""
    res_all, st_all = [], []
    for tc, d, s, TT, ta in events:
        sel = np.isfinite(d) & (d / s >= MIN_EPOCH_SNR_TIMING)
        if sel.sum() < 3:
            continue
        tm = epoch_times(t, y, tc[sel], TT, ta, half_range=TT / 3)
        Q = d[sel] / s[sel]
        st = np.hypot(np.maximum(TT / Q * np.sqrt(max(ta, 1e-4) / (2 * TT)), cad / 2), TIMING_JITTER * TT)
        good = np.isfinite(tm)
        if good.sum() < 3:
            continue
        x, oc, w = tc[sel][good], (tm - tc[sel])[good], 1 / st[good] ** 2
        A = np.vstack([np.ones_like(x), x - x.mean()]).T
        coef = np.linalg.lstsq(A * np.sqrt(w)[:, None], oc * np.sqrt(w), rcond=None)[0]
        res_all.append(oc - A @ coef)
        st_all.append(st[good])
    if not res_all:
        return 0.0, {"n_timed": 0, "oc_chi2_red": np.nan}
    res, st = np.concatenate(res_all), np.concatenate(st_all)
    alt = np.sqrt(st ** 2 + (TIMING_ALT_FRAC * T) ** 2)
    F5 = float(np.sum(-0.5 * (res / st) ** 2 - np.log(st) + 0.5 * (res / alt) ** 2 + np.log(alt)))
    return F5, {"n_timed": int(res.size), "oc_chi2_red": float(np.sum((res / st) ** 2) / max(res.size - 2 * len(res_all), 1))}


# ---------------------------------------------------------------- F6: shape of alternating events
def shape(t, y, P, t0, T, tau, sigma):
    """F6: separate trapezoid fits (depth, T, tau free; ephemeris fixed) to odd and to even events; chi^2 of
    the differences in duration and in tau/T; F6 = -chi^2 / 2. Zero when either parity has no usable fit."""
    cyc = np.round((t - t0) / P).astype(np.int64)
    fits = {}
    for g in (0, 1):
        m = (cyc % 2) == g
        f = M.fit_trapezoid(t[m], y[m], P, t0, T, tau, sigma, free_ephemeris=False) if m.sum() > 8 else None
        if f is None or not (f["depth"] > 0) or not np.isfinite(f["err"].get("depth", np.nan)) or f["depth"] < 3 * f["err"]["depth"]:
            return 0.0, {"dur_ratio_oe": np.nan, "shape_chi2": np.nan}
        fits[g] = f
    a, b = fits[0], fits[1]
    sTa, sTb = np.hypot(a["err"]["T"], SHAPE_JITTER * a["T"]), np.hypot(b["err"]["T"], SHAPE_JITTER * b["T"])
    qa, qb = a["tau"] / a["T"], b["tau"] / b["T"]
    sqa = np.hypot(qa * np.hypot(a["err"]["tau"] / max(a["tau"], 1e-6), a["err"]["T"] / a["T"]), SHAPE_JITTER)
    sqb = np.hypot(qb * np.hypot(b["err"]["tau"] / max(b["tau"], 1e-6), b["err"]["T"] / b["T"]), SHAPE_JITTER)
    chi2 = 0.0
    for x, y_, sx, sy in ((a["T"], b["T"], sTa, sTb), (qa, qb, sqa, sqb)):
        v = sx ** 2 + sy ** 2
        if np.isfinite(v) and v > 0:
            chi2 += (x - y_) ** 2 / v
    return float(-0.5 * chi2), {"dur_ratio_oe": float(a["T"] / b["T"]), "shape_chi2": float(chi2)}
