"""Combiners (brief §5): (a) principled product of per-feature likelihood ratios; (b) learned gradient-boosted
classifier over the features, trained only on injections, with temperature calibration on a held-out
injection split. Both give a probability per alias factor r and abstain when the largest is below a threshold.
"""

from __future__ import annotations

import json
from fractions import Fraction

import numpy as np

DEFAULT_ABSTAIN = 0.9
FEATURES = ("F1_coverage", "F2_oddeven", "F3_secondary", "F4_density", "F5_coherence", "F6_shape")


# ---------------------------------------------------------------- (a) principled
def principled(hyps):
    """log p(h | data) = log prior(h) + sum_k F_k(h) + const. Prior: uniform over the alias factors r, then
    uniform over the a ephemerides (offsets j) of r = a/b. Independence assumptions: docs/resolver_design.md."""
    n_r = len({h.r for h in hyps})
    lp = np.array([-np.log(n_r) - np.log(h.r.numerator) + sum(h.features[k] for k in FEATURES) for h in hyps])
    lp -= np.logaddexp.reduce(lp)
    return np.exp(lp)


def by_alias(hyps, probs):
    """Sum hypothesis probabilities over offsets; the representative hypothesis of r is its most probable offset."""
    out = {}
    for h, p in zip(hyps, probs):
        k = str(h.r)
        if k not in out:
            out[k] = {"r": k, "p": 0.0, "best": h, "best_p": -1.0}
        out[k]["p"] += float(p)
        if p > out[k]["best_p"]:
            out[k]["best"], out[k]["best_p"] = h, float(p)
    return out


def summarise(hyps, probs, abstain=DEFAULT_ABSTAIN):
    a = by_alias(hyps, probs)
    top = max(a.values(), key=lambda x: x["p"])
    return {"p_alias": {k: v["p"] for k, v in a.items()},
            "P_alias": {k: v["best"].P for k, v in a.items()},
            "r_map": top["r"], "P_map": top["best"].P, "p_max": top["p"], "abstain": bool(top["p"] < abstain),
            "hyp_probs": {h.key: float(p) for h, p in zip(hyps, probs)}}


# ---------------------------------------------------------------- (b) learned
RAW_COLS = ("n_cov", "cov_frac", "n_missing", "snr_all", "oe_sigma", "sec_snr", "sec_depth_ratio", "sec_phase_off05",
            "sec_dur_ratio", "twin", "log_rho_ratio", "n_timed", "oc_chi2_red", "dur_ratio_oe", "shape_chi2",
            "tau_over_T", "depth")


def alias_rows(hyps, probs):
    """One row per alias factor (its most probable offset under the principled posterior). Features: each F_k
    and the principled log-score relative to the target's maximum, the raw descriptors, and log10(period).
    The seed-relative factor r itself is NOT a feature, so the learned model cannot learn the seed
    distribution of its training set."""
    a = by_alias(hyps, probs)
    keys = list(a)
    rows = []
    fmax = {k: max(h.features[k] for h in hyps) for k in FEATURES}
    lp_all = np.log(np.maximum(probs, 1e-300))
    lpmax = lp_all.max()
    for k in keys:
        h = a[k]["best"]
        x = {f"d_{f}": h.features[f] - fmax[f] for f in FEATURES}
        x["d_logpost"] = float(np.log(max(a[k]["p"], 1e-300)) - lpmax)
        for c in RAW_COLS:
            v = h.raw.get(c, np.nan)
            x[c] = float(v) if v is not None else np.nan
        x["log10_P"] = float(np.log10(h.P))
        rows.append((k, h.P, x))
    return rows


class LearnedCombiner:
    """sklearn HistGradientBoostingClassifier on alias rows (label: refined period within 0.1 % of the truth),
    then per-target softmax of the classifier logits divided by a temperature fitted on a calibration split."""

    def __init__(self, seed=20260930, **kw):
        self.seed = seed
        self.params = {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31, "l2_regularization": 1.0,
                       "random_state": seed, **kw}
        self.temperature = 1.0
        self.columns = None
        self.clf = None

    @staticmethod
    def _matrix(groups, columns):
        return np.array([[x.get(c, np.nan) for c in columns] for g in groups for (_, _, x) in g], float)

    def fit(self, groups, labels):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.columns = sorted({c for g in groups for (_, _, x) in g for c in x})
        X = self._matrix(groups, self.columns)
        yv = np.concatenate([np.asarray(l, int) for l in labels])
        self.clf = HistGradientBoostingClassifier(**self.params).fit(X, yv)
        return self

    def logits(self, group):
        X = self._matrix([group], self.columns)
        p = np.clip(self.clf.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
        return np.log(p / (1 - p))

    def predict(self, group):
        z = self.logits(group) / self.temperature
        z -= np.logaddexp.reduce(z)
        return np.exp(z)

    def calibrate(self, groups, labels, grid=np.exp(np.linspace(np.log(0.05), np.log(20), 200))):
        """Temperature minimising the negative log-likelihood of the true alias on the calibration split
        (targets whose candidate set contains the truth)."""
        Z = [self.logits(g) for g in groups]
        idx = [int(np.argmax(l)) if np.any(l) else -1 for l in labels]
        best = (np.inf, 1.0)
        for T in grid:
            nll = 0.0
            for z, i in zip(Z, idx):
                if i < 0:
                    continue
                zz = z / T
                nll -= zz[i] - np.logaddexp.reduce(zz)
            if nll < best[0]:
                best = (nll, T)
        self.temperature = float(best[1])
        return self


# ---------------------------------------------------------------- metrics
def ece(conf, correct, n_bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def accuracy_coverage(conf, correct, thresholds=np.linspace(0, 1, 101)):
    conf, correct = np.asarray(conf, float), np.asarray(correct, bool)
    out = []
    for th in thresholds:
        m = conf >= th
        out.append((float(th), float(m.mean()), float(correct[m].mean()) if m.any() else np.nan))
    return out


def accuracy_at_coverage(conf, correct, coverage=0.9):
    conf, correct = np.asarray(conf, float), np.asarray(correct, bool)
    o = np.argsort(-conf, kind="stable")
    k = max(1, int(np.ceil(coverage * conf.size)))
    return float(correct[o[:k]].mean()), float(conf[o[k - 1]])
