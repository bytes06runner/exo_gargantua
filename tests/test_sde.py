"""A6 SDE: exact chi2 from astropy's BLS likelihood power, and TLS 1.32's spectrum code reproduced."""
import math

import numpy as np
import pytest
from astropy.timeseries import BoxLeastSquares

from exogargantua import sde

tls_helpers = pytest.importorskip("transitleastsquares.helpers")
tls_stats = pytest.importorskip("transitleastsquares.stats")

DURS = np.array([0.04, 0.08, 0.15])
OVERSAMPLE = 10


def bls_c_reference(t, y, period, durations, oversample=OVERSAMPLE):
    """astropy 8.0.1 bls.c (fast method, likelihood objective, unit weights) for ONE period, in plain Python,
    returning the selected box as a boolean in-transit mask over the points."""
    t = t - t.min()
    y = y - np.median(y)
    bin_d = durations.min() / oversample
    n_bins = int(math.ceil(period / bin_d)) + oversample
    ind = ((t - period * np.floor(t / period)) / bin_d).astype(int) + 1
    # bin -> which original bin it holds after the wrap padding (index n_bins - oversample + j holds bin 1 + j)
    holds = np.arange(n_bins + 1)
    holds[n_bins - oversample:n_bins] = np.arange(1, oversample + 1)
    holds[n_bins] = -1  # never filled
    my = np.zeros(n_bins + 1)
    mi = np.zeros(n_bins + 1)
    np.add.at(my, ind, y)
    np.add.at(mi, ind, 1.0)
    my[n_bins - oversample:n_bins] = my[1:oversample + 1].copy()
    mi[n_bins - oversample:n_bins] = mi[1:oversample + 1].copy()
    cy, ci = np.cumsum(my), np.cumsum(mi)
    sum_y, W = y.sum(), float(y.size)
    best, box = -np.inf, None
    for d in durations:
        dur = int(math.floor(d / bin_d + 0.5))
        for n in range(0, n_bins - dur + 1):
            yi, ii = cy[n + dur] - cy[n], ci[n + dur] - ci[n]
            if ii < np.finfo(float).eps or W - ii < np.finfo(float).eps:
                continue
            y_in, y_out = yi / ii, (sum_y - yi) / (W - ii)
            obj = 0.5 * ii * (y_out - y_in) ** 2
            if y_out >= y_in and obj > best:
                best, box = obj, (n, dur)
    n, dur = box
    mask = np.isin(ind, holds[n + 1:n + dur + 1])
    return best, mask


@pytest.mark.parametrize("seed,periods", [(3, [1.37, 2.71, 4.05]), (4, [0.83, 3.33, 6.1])])
def test_exact_chi2_from_astropy_power(seed, periods):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 20, 1500))
    y = 1 + rng.normal(0, 1e-3, t.size)
    y[((t - 0.3) % 2.71) < 0.08] -= 2e-3
    res = BoxLeastSquares(t, y).power(periods, DURS, objective="likelihood", oversample=OVERSAMPLE)
    power, depth = np.asarray(res.power), np.asarray(res.depth)
    iin = sde.ivar_in_from_depth(power, depth)
    # ivar_in recovered from (power, depth) is the integer in-transit count
    np.testing.assert_allclose(iin, np.round(iin), rtol=0, atol=1e-6)
    chi2 = sde.bls_chi2(power, iin, y)
    for k, P in enumerate(periods):
        ref_power, m = bls_c_reference(t, y, P, DURS)
        np.testing.assert_allclose(ref_power, power[k], rtol=1e-11)        # same box as astropy
        assert m.sum() == round(iin[k])
        y_in, y_out = y[m].mean(), y[~m].mean()
        chi2_box = np.sum((y[m] - y_in) ** 2) + np.sum((y[~m] - y_out) ** 2)
        chi2_ref = np.sum((y - y_out) ** 2)                                  # constant at the out-of-transit mean
        np.testing.assert_allclose(0.5 * (chi2_ref - chi2_box), power[k], rtol=1e-9)
        np.testing.assert_allclose(chi2[k], chi2_box, rtol=1e-12)
        # the reference is not the global mean (or median) model: that conversion would be off
        assert abs((np.sum((y - y.mean()) ** 2) - 2 * power[k]) - chi2_box) > 1e3 * abs(chi2[k] - chi2_box)


def test_gpu_ivar_in_matches_astropy():
    pytest.importorskip("torch")
    from exogargantua import gpu_bls, search as SE
    rng = np.random.default_rng(5)
    t = np.sort(rng.uniform(0, 27, 6000))
    y = 1 + rng.normal(0, 5e-4, t.size)
    y[((t + 0.4) % 3.1) < 0.08] -= 1.5e-3
    pmin, pmax, _ = SE.period_limits(t)
    grid = SE.bls_grid(t, pmin, pmax)
    res = BoxLeastSquares(t, y).power(grid, SE.BLS_DURATIONS, objective="likelihood")
    pw, iin = gpu_bls.bls_power(t, y, grid, SE.BLS_DURATIONS, device="cpu", max_elems=2e6, return_ivar_in=True)
    np.testing.assert_allclose(pw, np.asarray(res.power), rtol=1e-9, atol=1e-18)
    np.testing.assert_allclose(iin, sde.ivar_in_from_depth(res.power, res.depth), rtol=1e-6)


@pytest.mark.parametrize("n,k", [(5000, 91), (5001, 91), (20000, 301), (3000, 7)])
def test_running_median_equals_tls(n, k):
    x = np.random.default_rng(n).normal(size=n).cumsum()
    np.testing.assert_array_equal(sde.running_median(x, k), tls_helpers.running_median(x, k))


def test_dense_spectra_equals_tls_at_tls_kernel():
    rng = np.random.default_rng(1)
    chi2 = 1000 + rng.normal(0, 1, 30000).cumsum() * 0.01 + rng.normal(0, 0.5, 30000)
    chi2[12345] -= 40
    for a, b in zip(sde.tls_spectra(chi2, 91), tls_stats.spectra(chi2, 3)):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=0)


def test_nearest_index():
    p = np.array([3.0, 1.0, 2.0, 4.0])  # unsorted on purpose
    np.testing.assert_array_equal(sde.nearest_index(p, np.array([0.5, 1.4, 1.5, 1.6, 3.9, 9.0])), [1, 1, 1, 2, 3, 3])


def test_sde_a6_and_cellmax_find_injected_peak():
    rng = np.random.default_rng(2)
    y = rng.normal(0, 1e-3, 4000)
    P = np.linspace(0.5, 20, 200000)
    pw = np.abs(rng.normal(0, 1e-4, P.size))
    pw[np.abs(P - 7.3) < 0.002] += 5e-3
    iin = np.full(P.size, 40.0)
    ptls = np.linspace(20, 0.5, 3000)  # TLS order: descending
    s, peak = sde.sde_a6(P, pw, iin, y, ptls)
    assert abs(peak - 7.3) < 0.01 and s > 9
    assert sde.sde_a6_cellmax(P, pw, iin, y, ptls) > 9
