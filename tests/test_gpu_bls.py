"""Amendment A2 (iv): the PyTorch BLS port must reproduce astropy's BLS (checked here on CPU)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from astropy.timeseries import BoxLeastSquares  # noqa: E402

from exogargantua import gpu_bls, search as SE  # noqa: E402


@pytest.mark.parametrize("seed,period", [(1, 2.3), (2, 7.9), (3, 0.83)])
def test_torch_bls_matches_astropy(seed, period):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 54.0, 6000))
    t = t[(t < 13) | (t > 14.5)]
    f = 1 + rng.normal(0, 5e-4, t.size)
    f[((t + 0.4) % period) < 0.08] -= 1.5e-3
    pmin, pmax, _ = SE.period_limits(t)
    grid = SE.bls_grid(t, pmin, pmax)
    ref = BoxLeastSquares(t, f).power(grid, SE.BLS_DURATIONS, objective="likelihood").power
    ours = gpu_bls.bls_power(t, f, grid, SE.BLS_DURATIONS, device="cpu", max_elems=2e6)
    np.testing.assert_allclose(ours, np.asarray(ref), rtol=1e-9, atol=1e-18)
    assert int(np.argmax(ours)) == int(np.argmax(ref))
