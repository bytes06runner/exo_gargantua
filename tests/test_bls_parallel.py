"""Amendment A1: parallel BLS must return exactly the serial BLS peaks."""
import glob
import json
from pathlib import Path

import numpy as np
import pytest

from exogargantua import search as SE

ROOT = Path(__file__).resolve().parents[1]


def _synthetic(seed, period):
    rng = np.random.default_rng(seed)
    t = np.arange(0, 27.0, 10 / 1440)
    f = 1 + rng.normal(0, 5e-4, t.size)
    f[((t + 0.3) % period) < 0.1] -= 2e-3
    return t, f


def test_parallel_equals_serial_synthetic():
    inputs = [_synthetic(s, p) for s, p in [(1, 2.3), (2, 3.7), (3, 5.1), (4, 1.3)]]
    serial = [SE.bls_peak(t, f) for t, f in inputs]
    parallel = SE.bls_peaks_parallel(inputs, workers=4)
    assert [s["bls_period"] for s in serial] == [p["bls_period"] for p in parallel]
    assert [s["bls_index"] for s in serial] == [p["bls_index"] for p in parallel]


def test_parallel_equals_serial_on_20_smoke_targets():
    """Kaggle check (scripts/gate2b_checks.py): parallel periods vs the smoke run's serial periods."""
    files = sorted(glob.glob(str(ROOT / "results/kaggle/gate2b/*/gate2b/gate2b_results.json")))
    if not files:
        pytest.skip("gate2b results not pulled yet")
    res = json.loads(Path(files[-1]).read_text())
    rows = res["bls_parallel"]["targets"]
    assert len(rows) == 20
    for r in rows:
        assert r["parallel_bls_period"] == r["serial_bls_period"], r["toi"]
