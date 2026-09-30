"""Tschudi (2026a) adapter: pinned commit, and a positive/negative control on a synthetic 2P lock."""
import types
from pathlib import Path

import numpy as np
import pytest

from exogargantua.baselines import tschudi as T


@pytest.mark.network
def test_adapter_resolves_2p_lock_and_keeps_true_period(tmp_path_factory):
    mod = T.load(T.fetch(tmp_path_factory.mktemp("ts") / "repo"))
    assert len(mod.KNOWN_PLANETS) == 0
    cfg = T.config(mod, 2)
    rng = np.random.default_rng(7)
    t = np.arange(0, 54, 10 / 1440)
    t = t[(t < 13) | (t > 14.5)]
    f = 1 + rng.normal(0, 3e-4, t.size)
    ph = (t - 0.7) % 2.0
    f[(ph < 0.06) | (ph > 2.0 - 0.06)] -= 2e-3
    wrong = T.correct(mod, cfg, types.SimpleNamespace(period=4.0, SDE=20.0, T0=0.7, duration=0.12), t, f, 1.0, 1.0)
    right = T.correct(mod, cfg, types.SimpleNamespace(period=2.0, SDE=20.0, T0=0.7, duration=0.12), t, f, 1.0, 1.0)
    assert wrong["tschudi_resolved"] and abs(wrong["tschudi_period"] / 2.0 - 1) < 1e-3
    assert not right["tschudi_resolved"] and right["tschudi_period"] == 2.0
