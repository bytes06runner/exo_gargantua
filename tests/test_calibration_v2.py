"""Decision V2-f: the v2 map changes only the probabilities, never the chosen alias, and stays normalised."""
import numpy as np
from sklearn.isotonic import IsotonicRegression

from exogargantua.calibration_v2 import IsotonicTopCalibrated


class _Base:
    def __init__(self, p):
        self.p = np.asarray(p, float)

    def predict(self, group):
        return self.p


def test_map_normalised_and_argmax_kept():
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit([0.2, 0.5, 0.9, 1.0], [0.0, 0.3, 0.6, 0.9])
    for p in ([0.7, 0.2, 0.1], [0.95, 0.03, 0.02], [0.4, 0.35, 0.25]):
        out = IsotonicTopCalibrated(_Base(p), iso).predict(None)
        i = int(np.argmax(p))
        assert out.sum() <= 1 + 1e-12 and out.max() == out[i]          # normalised or sub-normalised; MAP unchanged
        assert abs(out[i] - iso.predict([max(p)])[0]) < 1e-12
        rest, prest = np.delete(out, i), np.delete(np.asarray(p), i)
        if np.all(rest < out[i]):                                     # no cap active: proportional, sums to 1
            np.testing.assert_allclose(rest / rest.sum(), prest / prest.sum())
            assert abs(out.sum() - 1) < 1e-12
