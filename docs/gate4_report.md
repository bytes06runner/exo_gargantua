# Gate 4 report: B1 and B2 results (2026-10-20)

Sources:
- `results/b1_test_report.json`
- `results/b1_v2_posthoc_report.json`
- `results/b2_report.json`
- `results/v2_calibration_final.json`
- `results/v2_calibration_diagnostic.json`

Decisions: F1, V2-a … V2-score in `docs/decisions.md`. 95 % intervals are Wilson; McNemar tests are exact and paired.

Freeze tags:
- `resolver-frozen-v1`: 25264fa.
- `resolver-frozen-v2`: 88c4838. Recalibration only; the chosen alias is always v1's.

B2 was unsealed after both tags existed, with the primary comparison pre-registered (V2-e).

## B2 temporal holdout (1,169 TOIs: Tier A 1,068, Tier B 101)

Primary ("ours"): v2 seeded with the TLS peak, top-1 alias.

| | Accuracy [95 % CI] | Notes |
|---|---|---|
| **Ours (v2, TLS seed)** | **0.832 [0.810, 0.853]** | ECE 0.050; abstains on 26.5 %; 0.953 [0.937, 0.966] accurate on the 73.5 % it answers |
| Raw TLS | 0.814 [0.791, 0.836] | McNemar vs ours: 48 / 69 discordant, p = 0.064 |
| Raw BLS | 0.826 [0.804, 0.847] | 53 / 60, p = 0.57 |
| Tschudi 2026a | 0.814 [0.791, 0.836] | identical to TLS on this sample (its correction rarely triggers); p = 0.062 |
| SPOC-1 (single-sector TCE) | 0.675 [0.648, 0.701] | coverage 72.6 %; p = 1.7e-28 overall; where it gives a period (849): SPOC-1 0.929, McNemar 55 / 37 in SPOC's favour, p = 0.076 |
| SPOC-E | 0.765 [0.740, 0.788] | coverage 82.4 %; where given (963): 69 / 41 in SPOC's favour, p = 0.0097 |
| QLP-hist | 0.050 | coverage 5.1 % (catalog limits); where given (60): 0.967 |
| SPOC-hist | 0.439 | coverage 45.9 %; where given (536): 0.957, p = 4e-9 in SPOC's favour |

**Pre-registered success criterion (`docs/success_criteria.md`):** beat raw BLS, TLS and SPOC-1 with McNemar p < 0.01.
- **MISSED** against BLS and TLS: the differences are not significant, so the paper must say "comparable".
- Against SPOC-1 the overall win comes from coverage. Where SPOC-1 gives a period, it is at least as accurate as ours.

**Resolver seeded by each source vs that raw source.** The resolver helps a TLS seed but does not significantly help the others; it hurts BLS and catalog seeds:
- TLS: +21 net (69 vs 48), p = 0.064.
- BLS: −26 net (19 vs 45), p = 0.0016.
- SPOC-1: −23, p = 0.014.
- SPOC-E: −21, p = 0.040.
- SPOC-hist: −35, p = 1e-6.

The resolver turns some correct seeds into wrong aliases more often than it fixes wrong ones, when the seed is already usually right.

**Calibration (ECE), v1 → v2, by seed source.**

| Seed source | v1 | v2 |
|---|---|---|
| TLS | 0.053 | 0.050 |
| BLS | 0.074 | 0.031 |
| Tschudi | 0.053 | 0.050 |
| SPOC-1 | 0.024 | 0.031 |
| SPOC-E | 0.023 | 0.048 |
| QLP-hist | 0.006 | 0.005 |
| SPOC-hist | 0.011 | 0.030 |

v2 was calibrated on BLS + TLS seeds only, so for the catalog seeds it is out of distribution. Accuracy is identical for v1 and v2 by construction.

**Ceiling (truth among the 11 candidates), by seed source:**
- TLS 0.879
- BLS 0.847
- SPOC-1 0.707
- SPOC-E 0.801
- QLP-hist 0.049
- SPOC-hist 0.447

These are search- or coverage-limited (V2-d).

**By truth tier** (ours vs TLS):
- A (planets): 0.837 vs 0.824, p = 0.18.
- B (EBs): 0.782 vs 0.713, p = 0.21. Tier B ECE is 0.149, poorly calibrated.

**By early-sector stratum** (ours vs TLS):

| Stratum | n | Ours | TLS | p |
|---|---|---|---|---|
| 1 sector | 577 | 0.825 | 0.771 | 0.001 (58 vs 27) |
| 2-3 | 421 | 0.843 | 0.850 | 0.68 |
| 4-6 | 78 | 0.808 | 0.859 | 0.12 |
| 7-12 | 70 | 0.843 | 0.886 | 0.38 |
| 13+ | 23 | 0.870 | 0.870 | 1 |

These stratum tests are exploratory, not pre-registered, so the 1-sector result is a finding to report, not a claim.

## B1 injection-recovery (test split, 862 hosts)

| Seed | Raw seed alone | Resolver (v1 = v2) | ECE v1 | ECE v2 (post hoc) | Ceiling |
|---|---|---|---|---|---|
| Wrong alias (8,620) | 0.091 | 0.866 [0.859, 0.873] | 0.019 | 0.056 | 0.971 |
| BLS search (2,000) | 0.644 | 0.784 [0.765, 0.801] | 0.096 | 0.014 | 0.831 |
| TLS search (2,000) | 0.608 | 0.791 [0.773, 0.809] | 0.097 | 0.017 | 0.849 |

- The resolver beats the raw search seed on injections: McNemar p < 1e-60 against both BLS and TLS.
- **ECE < 0.05 criterion:** met by v1 on wrong-alias seeds. Missed by v1 on search seeds. Met by v2 on search seeds, which is POST HOC.

## Things that went wrong and were fixed, all logged

- The refinement tie-break (R1).
- TLS 12-hour session limits; resumed.
- The V2-a temperature calibration did not work; replaced by isotonic (V2-f, chosen on training data only).
- The V2-f.1 cap.
- Scoring had to apply V2-f.1's chosen-alias rule (V2-score).
- A scoring guard performance fix.
- Diagnostic label alignment.

Not started, as instructed: B3 audit, false-alarm runs, B4 runtime, paper writing.

## Post-Gate-4 results summary (post hoc unless stated; `results/b2_posthoc_report.json`)

- **Risk-coverage, stated plainly:** ranking raw TLS periods by TLS's own SDE nearly matches our risk-coverage
  curve on B2 (accuracy at 73.5 % coverage: ours 0.953, TLS-by-SDE 0.949; area under the accuracy-coverage curve
  0.956 vs 0.950). Our remaining advantage is **calibrated alias probabilities and an abstain decision**, not a better
  ranking of which TLS periods to trust.
- Selective accuracy on the 859 TOIs v2 answers: ours 0.953, raw TLS 0.912 (McNemar p = 6e-5), raw BLS 0.948 (p = 0.56).
- B1(ii) and B2 differ: B2 seeds are already right far more often (raw TLS 0.81 vs 0.61), with fewer EBs (9 % vs 40 %),
  a narrower period range and more single-sector targets.
- B3 dropped (decision B3-nogo).
