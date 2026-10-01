"""A6 SDE helpers reproduce TLS 1.32 exactly."""
import numpy as np
import pytest

from exogargantua import sde

tls_helpers = pytest.importorskip("transitleastsquares.helpers")
tls_stats = pytest.importorskip("transitleastsquares.stats")


@pytest.mark.parametrize("n,k", [(5000, 91), (5001, 91), (20000, 301), (3000, 7)])
def test_running_median_equals_tls(n, k):
    x = np.random.default_rng(n).normal(size=n).cumsum()
    np.testing.assert_array_equal(sde.running_median(x, k), tls_helpers.running_median(x, k))


def test_dense_spectra_equals_tls_at_tls_kernel():
    rng = np.random.default_rng(1)
    chi2 = 1000 + rng.normal(0, 1, 30000).cumsum() * 0.01 + rng.normal(0, 0.5, 30000)
    chi2[12345] -= 40
    ours = sde.tls_spectra(chi2, 91)
    theirs = tls_stats.spectra(chi2, 3)
    for a, b in zip(ours, theirs):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=0)


def test_at_periods_nearest():
    p = np.array([3.0, 1.0, 2.0, 4.0])  # unsorted on purpose
    w = np.array([30.0, 10.0, 20.0, 40.0])
    np.testing.assert_array_equal(sde.at_periods(p, w, np.array([0.5, 1.4, 1.6, 3.9, 9.0])), [10, 10, 20, 40, 40])


def test_sde_a6_finds_injected_peak():
    rng = np.random.default_rng(2)
    t = np.sort(rng.uniform(0, 50, 4000))
    y = rng.normal(0, 1e-3, t.size)
    P = np.linspace(0.5, 20, 200000)
    pw = np.abs(rng.normal(0, 1e-4, P.size))
    pw[np.abs(P - 7.3) < 0.002] += 5e-3
    ptls = np.linspace(0.5, 20, 3000)
    s, peak = sde.sde_a6(P, pw, y, ptls)
    assert abs(peak - 7.3) < 0.01 and s > 9
