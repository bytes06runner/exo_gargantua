# B6: replication of the eclipsing-binary benchmark on unused binaries (pre-registration)

Written 2026-11-13, **before any B6 light curve is downloaded, searched or scored**. B6 exists because the B5 analysis
left three things that a confirmatory test should settle: B5 was run twice (decision B5-r1), the comparison with TLS
inside its search grid was post hoc (decision B5-floor), and the comparison of the BLS-seeded resolver with raw BLS was
secondary. B6 repeats B5 on binaries nobody has looked at, with those comparisons fixed in advance.

The resolver is the frozen v2 (`resolver-frozen-v2`, a3cbbd1). **Nothing about the resolver, its thresholds or the
searches changes, whatever B6 shows.** If a hypothesis below fails, the paper reports a failed replication.

## 1. Sample

Every EB of the Prša et al. (2022) catalog that passed the B5 rules Y01–Y05 and was excluded from B5 **only** by the
random cap S01 (`data/b5_sample_log.csv`, reason "S01: not in the pre-seeded subset"; written at the B5
pre-registration, commit 4435520, with `numpy.random.default_rng(20261021)`). That is the random complement of B5 in
the same population: **1,782 EBs** (1,623 Tier B, 159 Tier B-amb), no cap, no new rule, no star in common with B5.
`scripts/b6_select.py` writes `data/b6_sample_log.csv`, `data/b6_pinned_products.csv` (4,054 early-window and 10,112
later files, pinned by name from the frozen MAST listing of 2026-09-30) and `results/b6_selection.json`.

Early window, later sectors, the eligibility check H3 (at least two primary eclipses predicted by the catalog ephemeris
in the early window with at least half their cadences present) and Tier B-amb are defined exactly as in
`docs/b5_definition.md` §1–2. **Primary sample: Tier B, H3 pass.** B-amb is reported separately.

## 2. Truth and correctness

Truth is the catalog period. A period is correct iff |P / P_true − 1| < 0.001. A method that gives no period for a
star (including a failed search) counts as wrong.

## 3. Methods (identical to the B5 rerun)

- Light curve: pinned early-window SPOC 2-minute PDCSAP files, `search.prepare`.
- Seeds: GPU BLS on the pre-registered grid; TLS 1.32 through `search.tls_peak` with TIC v8.2 radius and mass.
  Stellar parameters are fetched from MAST into `data/b6_stellar_params.csv` **before** any light curve is downloaded
  and are not changed afterwards. Stars without TIC radius and mass run with TLS defaults, as in B2 and B5.
- Resolver: frozen v2, run from the TLS seed and from the BLS seed, with the TIC density where it exists.
- Code: `scripts/b5_run.py` with `--log data/b6_sample_log.csv --stars data/b6_stellar_params.csv`; the resolver and
  `src/exogargantua/search.py` must be byte-identical to the B5 rerun (checked by the scoring script).

## 4. Confirmatory hypotheses

Each is an exact two-sided paired McNemar test on the stated sample with the resolver correct more often than the
baseline. With three hypotheses, each must reach **p < 0.01 / 3** (Bonferroni). All three outcomes are reported in the
abstract whether they hold or not.

- **H1 (replication of the B5 criterion).** On the primary sample, the TLS-seeded v2 resolver beats raw TLS **and** raw
  BLS. Both tests must pass.
- **H2 (inside the TLS search grid; post hoc in B5).** On the primary binaries that have a TIC radius and mass and
  whose catalog period is at or above the TLS period floor, P_floor = 2π [(3R)³ / (GM)]^½ (the function
  `floor_days` of `scripts/b5_tls_floor.py`, same constants), the TLS-seeded v2 resolver beats raw TLS.
- **H3 (same seed).** On the primary sample, the BLS-seeded v2 resolver beats raw BLS.

## 5. Secondary analyses (descriptive; predictions stated so they can fail)

- **P1, calibration.** ECE (10 bins) of the TLS-seeded v2 resolver on the primary sample. Prediction from B5: above
  0.05, i.e. not calibrated on binaries.
- **P2, combiner and terms.** Accuracy of the sum-of-terms score of decision FI-1b on the stored alias rows, with all
  six terms and with each left out. Predictions from B5: within 0.02 of the learned combiner; leaving out F4 does not
  lower it; leaving out F1 or F2 lowers it by more than 0.2.
- **P3, catalog-truth audit.** Same procedure, model and Δχ² > 10 rule as `docs/b5_definition.md` §6, on every scored
  B6 binary where the TLS-seeded v2 period differs from the catalog. Prediction from B5: the numbers favoring the
  resolver and favoring the catalog are within a factor 1.5 of each other.
- Accuracy by period range and by number of early sectors; raw alias rates (decision Alias-1 rules); error categories
  (decision Err-1 rules); SPOC TCE periods (decision B5-spoc rules); Gaia DR3 periods (decision Indep-1 rules);
  the Tschudi (2026a) correction (decision B5-tsch rules) if compute allows. All with the rules already written for B5.
- **Pooled estimate.** B5 and B6 are random halves of one population, so accuracies and McNemar tests are also given
  for the two primary samples combined. The pooled numbers are secondary; the abstract quotes B5 and B6 separately.

## 6. What is not allowed

No retraining, recalibration or threshold change. No change to the sample after selection. No hypothesis added,
dropped or re-worded after the first B6 run record exists. No peeking: the scoring script is written before the search
jobs finish and is run once on the complete set of runs.

## 7. Compute (Kaggle; estimate from the B5 rerun)

Early-window cache: 4,054 files, one CPU job. GPU BLS with the BLS-seeded resolver: 2 GPU shards, about 5 h each.
TLS with the TLS-seeded resolver: 12 CPU shards of about 150 stars, about 2.5–3 h each. Audit: later sectors of the
disagreements, one CPU job, under 1 h. About 1.5 days of wall-clock time at five concurrent sessions.
