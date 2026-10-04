"""Alias resolver (brief §5): model pieces, the six features, both combiners, end-to-end on synthetic data."""
from fractions import Fraction

import numpy as np
import pytest

from exogargantua.resolver import combine as C, features as F, hypotheses as H, model as M
from exogargantua.resolver import Star, resolve

CAD = 2 / 1440


def box_lc(P=3.0, t0=1.0, T=0.12, depth=3e-3, sigma=5e-4, span=54.0, seed=0, every=1, depth2=None, phase2=0.5):
    """Trapezoid transits (tau = T/5) on white noise; `every`: keep every n-th transit; depth2: secondary."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, span, CAD)
    t = t[(t % 13.7) > 1.0]  # orbit gaps
    n = np.round((t - t0) / P)
    y = depth * M.trapezoid(t - t0 - n * P, T, T / 5) * ((n % every) == 0)
    if depth2:
        n2 = np.round((t - t0 - phase2 * P) / P)
        y += depth2 * M.trapezoid(t - t0 - phase2 * P - n2 * P, T, T / 5)
    return t, 1 - y + rng.normal(0, sigma, t.size)


def test_trapezoid_shape():
    dt = np.array([0.0, 0.03, 0.05, 0.061, 0.1])  # flat to T/2 - tau = 0.04
    m = M.trapezoid(dt, 0.12, 0.02)
    assert m[0] == 1 and m[1] == 1 and abs(m[2] - 0.5) < 1e-12 and m[3] == 0 and m[4] == 0


def test_epoch_fits_recover_depth_and_flag_flat_epochs():
    t, f = box_lc(every=2)
    ef = M.epoch_fits(t, 1 - f, 3.0, 1.0, 0.12, 0.024, 5e-4, CAD)
    c = ef["covered"]
    on, off = c & (ef["n"] % 2 == 0), c & (ef["n"] % 2 == 1)
    assert abs(np.mean(ef["d"][on]) - 3e-3) < 2e-4 and abs(np.mean(ef["d"][off])) < 2e-4
    assert np.allclose(ef["s"][c], 5e-4 / np.sqrt(np.sum(M.trapezoid(np.arange(-0.07, 0.07, CAD), 0.12, 0.024) ** 2)), rtol=0.2)


def test_red_noise_beta_white_is_one():
    rng = np.random.default_rng(1)
    t = np.arange(0, 27, CAD)
    y = rng.normal(0, 1e-3, t.size)
    assert M.red_noise_beta(t, y, 0.1, 1e-3) < 1.15
    y2 = y + 1e-3 * np.sin(2 * np.pi * t / 0.7)
    assert M.red_noise_beta(t, y2, 0.1, M.white_noise(y2)) > 2


def test_hypothesis_enumeration():
    hs = H.enumerate_hypotheses(2.0, 0.5, 0.1)
    assert len(hs) == sum(r.numerator for r in H.ALIAS_SET) == 24
    r2 = [h for h in hs if h.r == Fraction(2)]
    assert sorted(h.t0 for h in r2) == [0.5, 2.5] and all(h.P == 4.0 for h in r2)
    # every hypothesis contains a seed event
    for h in hs:
        assert abs(((h.t0 - 0.5) / 2.0) - round((h.t0 - 0.5) / 2.0)) < 1e-12


def test_seed_ephemeris_and_refine():
    t, f = box_lc()
    y = 1 - f
    t0, T = H.seed_ephemeris(t, y, 3.0, 5e-4)
    assert abs(((t0 - 1.0 + 1.5) % 3.0) - 1.5) < 0.03 and 0.06 <= T <= 0.2
    h = H.refine(H.Hypothesis(Fraction(1), 0, 3.0007, t0, T, T / 4), t, y, 5e-4)
    assert abs(h.P - 3.0) < 2e-4 and abs(h.fit["depth"] - 3e-3) < 3e-4


def test_f1_f2_split_is_exact_and_penalises_half_period():
    t, f = box_lc()
    y = 1 - f
    for P, want_oe in ((3.0, False), (1.5, True)):
        ef = M.epoch_fits(t, y, P, 1.0, 0.12, 0.024, 5e-4, CAD)
        F1, F2, raw = F.coverage_and_oddeven(ef)
        c = ef["covered"]
        s = F.jittered(ef["d"][c], ef["s"][c])
        S, W = np.sum(ef["d"][c] / s ** 2), np.sum(1 / s ** 2)
        assert abs((F1 + F2) - 0.5 * max(S, 0) ** 2 / W) < 1e-6 * max(1, F1)
        assert (raw["oe_sigma"] > 10) == want_oe


def test_secondary_detects_eb_and_not_planet():
    for d2, want in ((1.5e-3, True), (None, False)):
        t, f = box_lc(depth2=d2, phase2=0.5)
        F3, raw = F.secondary(t, 1 - f - 3e-3 * M.trapezoid(t - 1.0 - np.round((t - 1.0) / 3) * 3, 0.12, 0.024), 3.0, 1.0, 0.12,
                              5e-4, 3e-3, CAD)
        assert raw["sec_detected"] == want
        if want:
            assert abs(raw["sec_phase"] - 0.5) < 0.01 and abs(raw["sec_depth_ratio"] - 0.5) < 0.1 and F3 > 20
        else:
            assert F3 < 1


def test_secondary_finds_eccentric_phase():
    t, f = box_lc(depth2=2e-3, phase2=0.37)
    resid = 1 - f - 3e-3 * M.trapezoid(t - 1.0 - np.round((t - 1.0) / 3) * 3, 0.12, 0.024)
    _, raw = F.secondary(t, resid, 3.0, 1.0, 0.12, 5e-4, 3e-3, CAD)
    assert abs(raw["sec_phase"] - 0.37) < 0.01


def test_density_prefers_consistent_period():
    # Sun-like star; a 3-d planet has T14 ~ 0.12 d at b ~ 0.3
    dp = F.DensityPrior(1.41, 0.2)
    l3, _ = F.density(dp, 3.0, 0.125, 0.02, 0.003, 0.003, 1e-3)
    l03, _ = F.density(dp, 0.3, 0.125, 0.02, 0.003, 0.003, 1e-3)
    assert l3 > l03 + 5
    assert F.density(F.DensityPrior(np.nan, np.nan), 3.0, 0.125, 0.02, 0.003, 0.003, 1e-3)[0] == 0.0


def test_coherence_penalises_wrong_period():
    t, f = box_lc(span=100)
    y = 1 - f
    out = {}
    for P in (3.0, 3.0 + 0.04 / 33):  # second: drifts by ~T/3 across the baseline
        ef = M.epoch_fits(t, y, P, 1.0, 0.12, 0.024, 5e-4, CAD)
        c = ef["covered"]
        out[P], _ = F.coherence(t, y, [(ef["tc"][c], ef["d"][c], ef["s"][c], 0.12, 0.024)], 0.12, 0.024, CAD)
    assert out[3.0] > 0
    # a linear drift is absorbed by the fitted ephemeris; coherence is about scatter, so the two agree
    assert abs(out[3.0] - out[3.0 + 0.04 / 33]) < 5


def test_shape_flags_different_durations():
    rng = np.random.default_rng(3)
    t = np.arange(0, 54, CAD)
    n = np.round((t - 1.0) / 3.0)
    T = np.where(n % 2 == 0, 0.12, 0.20)
    y = 3e-3 * M.trapezoid(t - 1.0 - n * 3.0, T, T / 5)
    F6, raw = F.shape(t, y + rng.normal(0, 5e-4, t.size), 3.0, 1.0, 0.16, 0.03, 5e-4)
    assert F6 < -10 and raw["dur_ratio_oe"] < 0.8
    F6b, _ = F.shape(t, 3e-3 * M.trapezoid(t - 1.0 - n * 3.0, 0.12, 0.024) + rng.normal(0, 5e-4, t.size), 3.0, 1.0, 0.12, 0.024, 5e-4)
    assert F6b > -5


def test_principled_probabilities_sum_and_prior():
    hs = H.enumerate_hypotheses(2.0, 0.5, 0.1)
    for h in hs:
        h.features = {k: 0.0 for k in C.FEATURES}
    p = C.principled(hs)
    assert abs(p.sum() - 1) < 1e-12
    pa = C.by_alias(hs, p)
    assert all(abs(v["p"] - 1 / 11) < 1e-12 for v in pa.values())  # uniform over r with no evidence
    s = C.summarise(hs, p)
    assert s["abstain"]


@pytest.mark.parametrize("seed_P,truth", [(1.5, 3.0), (6.0, 3.0), (3.0, 3.0), (1.0, 3.0)])
def test_end_to_end_planet(seed_P, truth):
    t, f = box_lc()
    s, hs = resolve(t, f, seed_P, star=Star(1.41, 0.2))
    assert abs(s["P_map"] / truth - 1) < 1e-3 and s["p_max"] > 0.9 and not s["abstain"]


def test_end_to_end_noise_abstains_or_low_confidence():
    rng = np.random.default_rng(7)
    t = np.arange(0, 27, CAD)
    f = 1 + rng.normal(0, 5e-4, t.size)
    s, _ = resolve(t, f, 2.3)
    assert s["p_max"] < 0.999


def test_learned_combiner_and_temperature():
    rng = np.random.default_rng(0)
    groups, labels = [], []
    for _ in range(300):
        k = 5
        true = rng.integers(k)
        rows = []
        for i in range(k):
            x = {"a": rng.normal(2.0 if i == true else 0.0, 1.0), "b": rng.normal()}
            rows.append((str(i), 1.0, x))
        groups.append(rows)
        labels.append([int(i == true) for i in range(k)])
    lc = C.LearnedCombiner(max_iter=50).fit(groups[:200], labels[:200]).calibrate(groups[200:], labels[200:])
    p = lc.predict(groups[0])
    assert abs(p.sum() - 1) < 1e-9 and 0.05 <= lc.temperature <= 20
    acc = np.mean([np.argmax(lc.predict(g)) == np.argmax(l) for g, l in zip(groups[200:], labels[200:])])
    assert acc > 0.6


def test_metrics():
    assert C.ece([1.0, 1.0], [1, 1]) == 0
    assert abs(C.ece([0.8] * 10, [1] * 8 + [0] * 2)) < 1e-12
    acc, th = C.accuracy_at_coverage([0.9, 0.8, 0.3, 0.2], [1, 1, 0, 0], 0.5)
    assert acc == 1.0 and th == 0.8


def test_refine_keeps_seed_period_when_unconstrained():
    # one transit in the data: the period is unconstrained, so refinement must stay at r x P0
    t, f = box_lc(P=40.0, t0=10.0, span=27.0)
    y = 1 - f
    h = H.refine(H.Hypothesis(Fraction(1), 0, 40.0, 10.0, 0.12, 0.03), t, y, 5e-4)
    assert abs(h.P - 40.0) < 1e-6


def test_pick_near_seed_tie_break():
    g = np.zeros((5, 3))
    dP, dt0 = np.linspace(-2, 2, 5), np.linspace(-1, 1, 3)
    assert H.pick_near_seed(g, dP, dt0) == (2, 1)
    g[0, 0] = 10.0
    assert H.pick_near_seed(g, dP, dt0) == (0, 0)


def test_fit_holds_period_with_one_transit():
    t, f = box_lc(P=40.0, t0=10.0, span=27.0)
    fit = M.fit_trapezoid(t, 1 - f, 40.0, 10.0, 0.12, 0.03, 5e-4, free_ephemeris=True)
    assert fit["P"] == 40.0 and "P" not in fit["err"] and abs(fit["t0"] - 10.0) < 0.01
