"""Resolver v2 calibration map (decision V2-f): isotonic regression of the maximum alias probability of the v1
learned combiner (v1 classifier and temperature, unchanged), fitted on all v2 calibration runs. The other aliases
keep their relative v1 probabilities, rescaled to share 1 - calibrated maximum, each capped at the calibrated maximum
so the chosen alias never changes (V2-f.1); mass removed by the cap is left unassigned ("none of the candidates"),
so the alias probabilities may sum to less than 1."""

from __future__ import annotations

import numpy as np


class IsotonicTopCalibrated:
    def __init__(self, base, iso):
        self.base, self.iso = base, iso  # base: v1 LearnedCombiner; iso: fitted sklearn IsotonicRegression

    def predict(self, group):
        p = np.asarray(self.base.predict(group), float)
        i = int(np.argmax(p))
        top = float(np.clip(self.iso.predict([p[i]])[0], 0.0, 1.0))
        rest = p.sum() - p[i]
        out = p * ((1.0 - top) / rest) if rest > 0 else np.zeros_like(p)
        out = np.minimum(out, np.nextafter(top, 0.0))  # strictly below: no ties with the chosen alias
        out[i] = top
        return out
