# Phase 3 report: alias resolver (2026-10-01)

Nothing has been run on the real injection pool. The B2 holdout is still sealed: no truth comparison
of any kind has been made.

## What was built
- **Code:** `src/exogargantua/resolver/`. Design notes for each feature are in
  `docs/resolver_design.md`.
- **Hypotheses:** the 11 alias factors × ephemeris offsets give 24 hypotheses, each refined locally
  in period and epoch.
- **Six features**, each a log-likelihood term plus raw descriptors:
  - F1 predicted-epoch coverage;
  - F2 odd/even (exact orthogonal split with F1);
  - F3 secondary at any phase (Bayes factor);
  - F4 stellar density (forward model with Kipping 2013 eccentricity prior);
  - F5 ephemeris coherence (Carter 2008 timing errors);
  - F6 odd/even shape.
- **Combiners:**
  - (a) principled: product of likelihood ratios, uniform prior over r;
  - (b) learned: gradient-boosted classifier over the features, with temperature calibration.
  - Both give a probability per alias and abstain below a threshold (default 0.9).

## Unit tests
- `tests/test_resolver.py`: 19 tests, all passing.
- Full suite: 72 tests, all passing.

## Synthetic accuracy and calibration check
Numbers are in `results/resolver_synth_check.json`, written by `scripts/resolver_synth_report.py`
from `results/kaggle/resolver_synth/`.
- **Data:** 3,000 fully synthetic TESS-like targets: the B1 class mix, wrong seeds uniform over R,
  and 10 % signal-free light curves. This is a code check, not the benchmark.
- **Learned combiner:** trained and calibrated on synthetic train/calibration splits. Only B1
  injections will train it for the paper.
- **Test split:** results below are for its 526 detectable signals (SNR ≥ 7.1).

| | Principled | Learned |
|---|---|---|
| Top-1 alias accuracy | 0.867 (0.835–0.893) | 0.918 (0.892–0.939) |
| ECE | 0.091 | 0.027 |
| Accuracy at 90 % coverage | 0.916 | 0.973 |
| Abstain rate at 0.9 | 0.122 | 0.141 |
| Accuracy when not abstaining | 0.929 (0.901–0.949) | 0.980 (0.963–0.989) |
| Signal-free light curves resolved confidently | 2/51 | 0/51 |

- **Paired McNemar:** learned better, 28 vs 1 discordant, p = 1.1e-7.
- **Weakest class, equal-depth EB twins:** accuracy 0.566 (principled) and 0.763 (learned).
- **Weakest SNR bin, 7.1–15:** accuracy 0.46–0.49, abstain rate 0.56–0.62.
- **Runtime:** median 20.7 s per synthetic target (p95 74.9 s), with 4 worker processes on a
  4-core Kaggle CPU.

## Smoke targets
`results/resolver_smoke_runtime.json`: execution test only. All 20 targets are sealed B2 members.
- 40/40 runs completed (TLS and BLS seeds).
- Median 2.5 s per run, p95 12.8 s, one process on a Kaggle CPU session.
- No resolved period has been compared with anything.
