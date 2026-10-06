# Owner decisions log

| Date | Gate | Decision |
|---|---|---|
| 2026-09-30 | 1 | Proceed with repositioning. Framing is "calibrated and benchmarked", never "first". Rule-based harmonic correction of Tschudi (2026a) is added as a B2 baseline. |
| 2026-09-30 | — | Reviewer report for AAS80459 is kept in `docs/private/` (gitignored). Peer-review correspondence is never committed to this public repository; the response letter quotes only what the owner approves for the resubmission package. |
| 2026-09-30 | 1 | Close the novelty-check gap with an ADS full-text search. The ADS token is read from the `ADS_API_TOKEN` environment variable and never committed. |
| 2026-09-30 | — | Do not merge `v2-alias-resolver` into `main` yet. |
| 2026-09-30 | — | Tag `v1-aj-submission` created on `main` at 404d67c (the AAS80459 submission). `main` otherwise unchanged. |
| 2026-09-30 | 2 (recorded now, applies at Phase 2) | **Sector pinning.** When the sample is frozen, `data/sample_log.csv` must record, per target, the exact TESS sectors used (and product files / cadence), the MAST data release, and the download date, not just the target list. All later runs load **only** the pinned sectors. Reason: two of the five v1 re-runs (TIC 237222864, TIC 321669174) no longer reproduce because MAST served additional sectors after August 2026 (`results/legacy_audit/flagged_planet_reruns.json`). The sample-log columns will be extended in Phase 2 accordingly. |
| 2026-09-30 | 1 | **Gate 1 closed.** ADS full-text search (603 records, 8 fixed queries) found no calibrated alias resolver for short-period TESS signals. Phase 2 waits for the owner's go-ahead. |
| 2026-09-30 | 2 | **Freeze run of record = `exog-freeze-sample-v2`.** Decided before either run finished. The first run (`exog-freeze-sample`, commit 5560df3) is void: its Gaia step parsed 19-digit Gaia source ids through float64, which corrupts them and wrongly fails pool stars on P05 (found in a local 3-sector test; fixed in `scripts/freeze_sample.py` with `tests/test_freeze.py`). v2 also runs the TIC/Gaia queries in parallel with progress logging. Selection rules and seed are unchanged. |

## Pre-registration amendments (Gate 2, 2026-09-30)

At the time each amendment below was made, **no benchmark result existed**: no alias accuracy,
calibration, false-alarm or baseline-accuracy number had been computed for any method. The only
search outputs in existence are the 20 smoke-run BLS/TLS periods, recorded for timing and never
compared with truth (`scripts/smoke_run.py` computes no accuracy).

| ID | Date | Amendment | Reason |
|---|---|---|---|
| A1 | 2026-09-30 | **BLS parallelisation.** Seed BLS searches run 4 targets at a time (one astropy BLS per worker process); the BLS configuration itself is unchanged. A test (`tests/test_bls_parallel.py`) confirms parallel and serial runs return identical peak periods on the 20 smoke targets. | The smoke run showed astropy BLS uses 1 of 4 cores; this halves B2 wall time without changing any result. |
| A2 | 2026-09-30 | **B1 design.** (i) All 30,000 injections: the resolver is run from deliberately wrong seeds drawn from the alias family {P/5 … 5P} of the known injected period; no search is run. (ii) A random, pre-seeded subsample of 2,000 injections (seed `20260930`, drawn before any injection is run): full BLS + TLS searches exactly as pre-registered, to measure the realistic seed distribution. (iii) The paper reports (i) and (ii) separately. The BLS grid is **not** coarsened. (iv) Optional GPU BLS, hard cap 2 GPU hours for its validation: usable only if it reproduces astropy BLS peak periods on the 20 smoke targets within one step of the period grid; otherwise it is dropped and the outcome is logged here. | Full searches on all 30,000 injections were costed at ~3,000-5,700 h (`docs/gate2_report.md`), infeasible on Kaggle. |
| A3 | 2026-09-30 | **Truth source.** A TOI dispositioned CP or KP takes truth only from the NASA Exoplanet Archive (Tier A); it is never given Tier B truth from the EB catalog. Only the truth columns of `data/sample_log.csv` were re-frozen (`scripts/refreeze_truth.py`, which asserts every other column is unchanged); old and new checksums are in `data/FROZEN.json`. Six TOIs changed from Tier B to Tier C (catalog reference) and leave B2: **TOI 143.01, 1165.01, 1422.01, 1779.01, 2119.01, 4495.01** (`data/truth_amendment_log.csv`). B2 candidates: 1,204 → 1,198. | For TOI-4495.01 the EB-catalog period differs from the Exoplanet Archive period by 0.1 %, the benchmark's own tolerance; a confirmed planet's truth should come from the planet archive. |
| G1 | 2026-09-30 | **Kaggle compute: GPU (T4) sessions only.** Every Kaggle job from now on runs on a GPU T4 session; Kaggle CPU sessions are no longer used (owner instruction). | Owner decision. Consequence, stated for the record: CPU-bound work (TLS, astropy BLS, downloads) on a GPU session draws on the weekly GPU quota at the same wall-clock rate, so the weekly GPU quota becomes the binding constraint on the schedule. This supersedes the brief's instruction to move CPU-bound jobs to CPU sessions. |

## Gate 2 decisions, round 2 (2026-09-30)

No benchmark result existed when these were made (see the statement above the amendments table).

| ID | Date | Decision | Reason |
|---|---|---|---|
| G2 | 2026-09-30 | **Amends G1.** GPU (T4) sessions only for GPU work: GPU BLS, and later the resolver if it benefits. All CPU-bound work (TLS, detrending, downloads, compaction) runs on Kaggle CPU sessions, which have no weekly quota. Every job declares `"compute": "cpu"` or `"compute": "gpu"` in its `kernel-metadata.json`; `kaggle/push_job.py` refuses a GPU session for a CPU-tagged job and a CPU session for a GPU job. | Under G1, ~212 h of CPU-only TLS would have been billed to the weekly GPU quota. |
| A4 | 2026-09-30 | **Sealed holdout.** BLS, TLS, Tschudi, SPOC and QLP seeds for the B2 holdout are computed and stored, but **no accuracy (or any other comparison with holdout truth) is computed** until the resolver is frozen and the git tag `resolver-frozen-v1` exists. All scoring code goes through `exogargantua.scoring.require_frozen_resolver()`, which refuses to run unless that tag exists and is an ancestor of HEAD. Resolver design, feature development and training use injections (B1) only. | Prevents any tuning on the holdout. |
| D1 | 2026-09-30 | Download approved: 4 chunks, float64 flux in the compact cache. The owner creates the private Kaggle Dataset from the final chunk's output in the Kaggle UI; its slug and version are then recorded in `data/sample_log.meta.json` (a sidecar, because `data/sample_log.csv` is frozen by checksum). | Owner decision. |
| G2-m | 2026-09-30 | Measured: **5 concurrent batch CPU sessions** (5 probes overlapped for 163 s; a 6th push was refused with "Maximum batch CPU session count of 5 reached", `results/kaggle/concurrency_probe/result.json`). CPU-bound runs use up to 5 sessions at once. | Measurement for G2 planning. |
| O1 | 2026-10-01 | **Job orchestration moves to GitHub Actions** (`.github/workflows/kaggle-orchestrator.yml`, `kaggle/orchestrate.py`, `kaggle/queue.json`): 15-minute cron + `workflow_dispatch` + runs on pushes that change `kaggle/queue.json`; `concurrency` group without cancellation (runs never overlap); `permissions: contents: read` (no write-back of queue state); Kaggle credentials only from repository secrets `KAGGLE_USERNAME`, `KAGGLE_KEY` (added by the owner), never echoed. Guards: never re-push a job that exists on Kaggle in any state; push nothing when the Kaggle API does not answer; 300 s push timeout followed by a status check; 5 CPU / 1 GPU caps; G2 compute tags. Owner-approved option (a): the workflow file alone was added to `main` (commit 1e91c08) because scheduled runs fire from the default branch; `git diff v1-aj-submission main --stat` shows only that file. | Overnight on 2026-09-30/10-01 the laptop slept and stalled the local launchers. |

| A5 | 2026-10-01 | **P08 screening window (in effect; owner-approved option b).** The P08 pool screen (docs/sample_definition.md §5: "a BLS search (Section 6 configuration) of the uninjected, pinned light curve gives SDE < 9") is applied to the light curve each star's injections will use: its first <= 10 pinned sectors that start within 365.25 d of the first one (docs/injection_design.md §1), instead of all pinned sectors. Search configuration, SDE threshold (9) and the no-replacement rule are unchanged. | The literal wording cannot be applied as written: on full pinned spans (median 815 d, max 2,936 d) one star needs up to 21.0 h on one T4 and 101 stars exceed Kaggle's 12 h session limit, so their search cannot complete as specified; all 3,000 stars would cost 3,163 GPU-quota h (dual T4) or 14,511 h wall on 5 CPU sessions. The injection-window screen costs 36.7 GPU-quota h or 168.6 h wall on 5 CPU sessions, with no star over 0.09 h. `results/p08_cost.json`. No injection existed when this was made. |
| V1 | 2026-10-01 | **P08 SDE definition and GPU-port validation (pre-registered before any screen run).** SDE = (max(power) - mean(power)) / std(power) over the full BLS trial-period grid (likelihood objective; Kovács et al. 2002). P07 is applied first (sectors with CROWDSAP < 0.9 or X10/X11 dropped; star out if < 2 remain), then the A5 window. Validation: 50 pool stars drawn with `numpy.random.default_rng(20260930).choice(include, 50, replace=False)` (`data/p08_validation_stars.csv`, committed before the runs); SDE computed with astropy BLS on Kaggle CPU sessions and with the GPU port on a T4 session, same inputs and grid. Reported: max |SDE_gpu - SDE_astropy| and whether any star changes side of SDE = 9. If any star flips, the screen is not run and the owner decides. If none flips but differences exist, a borderline re-check rule is pre-registered before the screen. | Owner instruction (Gate 2, P08). |
| O2 | 2026-10-01 | **Orchestrator skip-guard test passed** (owner-confirmed). Launch: push-triggered run 36827603777 (06:57 UTC) pushed `orchestrator_smoke`, `p08_validate_cpu_1`, `p08_validate_cpu_2` and skipped `p08_validate_gpu` (COMPLETE). Skip while running: manual run #7, 36827899550 (07:00:49 UTC), and push run 36827853986 (07:00:24 UTC): all four jobs reported "already on Kaggle; never re-pushed", `pushed=[]`. Decision logs: `results/orchestrator_test/runs.json`. The smoke job is removed from `kaggle/queue.json` once it finishes. Note: the 15-minute cron had not fired by 07:00 UTC (~80 min after the workflow reached `main`); runs so far came from pushes and a manual dispatch. | Gate-2 orchestrator test. |
| A6 | 2026-10-01 | **P08 SDE on a TLS-detrended spectrum (in effect; owner-approved, TLS-grid reading). Supersedes V1's SDE definition for the P08 decision; V1's raw SDE is still computed and reported.** Full text in "Amendment A6" below. | The threshold 9 is TLS's (Hippke & Heller 2019), defined on a running-median-detrended spectrum; V1 applied it to the raw BLS spectrum, and in the 50-star validation raw-rule exclusions tracked window length, grid edges and the 13.7 d orbit rather than stellar signals (`results/p08_validation_report.json`). |

### Amendment A6 (in effect from this commit)

**Reason.** P08 excludes a pool star if an uninjected BLS search over its injection window gives
SDE >= 9. The threshold comes from TLS (Hippke & Heller 2019, A&A 623, A39), where SDE is computed
on a spectrum detrended with a running median. Decision V1 applied it to the raw BLS spectrum. In the
50-star validation (GPU half, raw rule), exclusions track the length of the screening window, and
the excluded stars' peaks sit at the edges of the trial-period grid or near the 13.7 d TESS orbit.

**Rule.** Only the SDE computation changes.

- **Unchanged:**
  - P07 and the A5 screening window;
  - §6 preparation (PDCSAP, wotan biweight 0.75 d, 10-min bins);
  - the BLS search (autoperiod grid, durations, likelihood objective, astropy or the validated GPU port);
  - the threshold (exclude if SDE >= 9);
  - the no-replacement rule.
- **Steps, per star:**
  1. **Exact chi² per BLS trial period.** Astropy's likelihood power (`bls.c`, astropy 8.0.1, unit
     weights when no uncertainties are given) is 0.5·ivar_in·(y_out − y_in)². Here y_in and y_out are
     the in- and out-of-transit means of the box selected at period P, and ivar_in is its in-transit
     point count.
     - This equals ½[χ²_ref(P) − χ²_box(P)], where χ²_box is the residual χ² of the selected box
       model.
     - χ²_ref(P) = Σ(y − y_out(P))² is the χ² of a constant at that box's **out-of-transit mean**. It
       is not the global mean, weighted mean or median, and it changes with P. Astropy's median
       subtraction cancels.
     - Hence, exactly, χ²(P) = χ²_box(P) = S − 2·power(P)·(1 − ivar_in(P)/W), with
       S = Σ(y − ȳ)² about the global mean and W = N.
     - ivar_in comes from `gpu_bls.bls_power(..., return_ivar_in=True)`. For astropy it is
       2·power/depth², an integer to float precision.
     - `tests/test_sde.py` rebuilds astropy's selected box point by point with a plain-Python `bls.c`
       and confirms all three: astropy's power (rtol 1e-11), the χ² identity (rtol 1e-9 on the power,
       1e-12 on χ²), and the GPU port's ivar_in.
  2. **TLS's own trial periods.** Take the grid TLS would search on this light curve:
     `transitleastsquares.period_grid(R_star, M_star, time_span, period_min, period_max,
     oversampling_factor=3, n_transits_min=2)`.
     - These are the arguments of `search.tls_peak`.
     - R_star and M_star come from `data/stellar_params.csv`, or default to TLS's 1.0 when missing.
     - The period limits are the §6 limits.
  3. **Reading χ² at those periods: nearest BLS trial period (decides).** Each TLS period takes the
     χ² of the nearest BLS trial period (ties to the shorter period).
     - Nearest is used because each value is then the χ² of a box model that was actually fitted:
       power and ivar_in belong to the same box.
     - Linear interpolation would blend two different boxes and produce a χ² of no model. BLS power
       is not smooth between trial periods, because phase and duration jump.
     - The BLS grid is 70–770× finer than TLS's on the smoke targets. The nearest BLS period is
       therefore within half a BLS step of the TLS period, below 1/140 of TLS's own spacing.
  4. **SDE.** Call `transitleastsquares.stats.spectra(chi2, 3)` unmodified (transitleastsquares
     **1.32**, 5 Apr 2024, file `transitleastsquares/stats.py`). It detrends with
     `transitleastsquares/helpers.py::running_median` over 3 × `SDE_MEDIAN_KERNEL_SIZE` (30) → 91
     trial periods. Its fifth output is the A6 SDE.
- **Implementation:** `src/exogargantua/sde.py::sde_a6`; recorded as `sde_a6` / `p08_a6` by
  `scripts/p08_screen.py`.
- **Diagnostics only, never decide:**
  - `sde_a6_cellmax`: per TLS period, the minimum χ² over the BLS periods in its cell (edges at
    midpoints between TLS periods).
  - `sde_dense_tlspts` and `sde_dense_scaled`: TLS detrending applied directly on the dense BLS
    grid. With the literal 91-point kernel the median follows transit peaks, and the noise-only SDE
    is inflated (`results/a6_synthetic_check.json`).

**Declarations.**
- **No benchmark results existed when A6 was made.** No injection has been generated, no injected
  search run, and no resolver built or scored. B2 seeds are sealed under A4.
- **P08 outcomes seen so far:** only those of the 50 validation stars under the raw rule.
- **Pool size is accepted as it falls.** We accept whatever pool size A6 produces, and P08 will not
  be changed again.
  - If the pool is small, B1 results are reported by sector-count stratum with wider intervals
    (option a).
  - Injections stay at 10 per star (option b, 20 per star, is rejected).
- **Sensitivity.** The paper reports pool size and injection results under both the raw rule (V1)
  and A6. Every screen row records both. The raw-rule analysis (`scripts/p08_validation_report.py`)
  is kept.

**Validation before the screen.**
- Re-run the 50 V1 stars with spectra saved (`kaggle/jobs/p08_a6val_*`), on GPU and on CPU astropy.
- For the raw rule and for A6, report:
  - flips across 9 between the two engines;
  - the maximum |ΔSDE| between engines;
  - peak-period agreement;
  - the A6 pool projection by sector stratum (Wilson 95%).
- The full screen is queued only after the owner approves these results.

## Phase 3 → B1 decisions (2026-10-04)

No injection had been generated and no B1 or B2 result existed when these were recorded. The P08
screen had finished (`results/p08_screen_report.json`): 2,155 pool stars under A6.

| ID | Date | Decision | Reason |
|---|---|---|---|
| F1 | 2026-10-04 | **Resolver freeze (owner rule).** The learned combiner is trained, and both combiners' abstain thresholds chosen, on the B1 training split only (docs/injection_design.md §6). Then the git tag `resolver-frozen-v1` is created. After the tag, nothing under `src/exogargantua/resolver/` or `models/resolver/` (trained combiner, temperature, thresholds) changes. Only after the tag may the B1 test split or the B2 holdout be scored. `exogargantua.scoring.require_frozen_resolver()` refuses unless the tag exists, is an ancestor of HEAD, and the frozen paths are identical at the tag, at HEAD and in the working tree (tests/test_scoring_guard.py). Any later change to the resolver needs a new tag, and every result scored with the old one is re-run and reported as such. | Owner instruction; prevents tuning on test or holdout data. |
| R1 | 2026-10-04 | **Resolver development fix before any B1 run.** On the Phase 3 synthetic check, 57 of 2,585 detectable targets had no alias candidate within 0.1 % of the true period. Cause: with one transit in the data the period is unconstrained, the refinement grid ties, and the tie went to the first grid point (the largest negative shift); the least-squares fit could then also move an unconstrained period. Fix: among grid points within 0.5 nats of the best, keep the one closest to r × P0; hold P fixed in the fit when fewer than two epochs are covered (commit fd48913). Before/after on the same 3,000 synthetic targets: `results/resolver_refine_fix_report.json`. | Found on synthetic data only; no B1 or B2 data involved. |
| A7 | 2026-10-04 | **B1 size and split (pre-registered before generation; docs/injection_design.md §2, §4, §6).** (i) Hosts are the 2,155 stars with `in_pool` in `data/injection_pool_screened.csv`; 10 injections each gives 21,550 (A6 pool accepted as it fell; 20 per star rejected). (ii) Injections are split by host star, stratified by sector stratum: 60 % of hosts training (of which 75 % fit / 25 % calibration), 40 % test. (iii) The B1(ii) full-search subsample (2,000) is drawn from the test-split injections only, so it is a held-out evaluation like B1(i) test. (iv) The B1 test split is sealed until `resolver-frozen-v1` (F1). | The pre-registered design assumed 30,000 injections and no split; a learned combiner needs a training split, and owner instruction requires a host-level split. |
| F1-tag | 2026-10-05 | **`resolver-frozen-v1` created at 25264fa** (01:05 IST). Learned combiner trained on 9,690 fit injections, temperature 1.457 and abstain thresholds (principled 0.99, learned 0.54) chosen on 3,240 calibration injections (`results/b1_training_report.json`). No B1-test or B2 truth had been read. From here the resolver and `models/resolver/` do not change (F1). | Freeze before scoring. |

## Resolver v2: recalibration only (2026-10-06, owner decision; recorded before any v2 data exists)

Context: on the B1 test split, v1's learned combiner met ECE < 0.05 with wrong-alias seeds (B1(i): 0.019)
but missed it with search seeds (B1(ii): 0.096 BLS, 0.097 TLS; `results/b1_test_report.json`). The B2 holdout
is still sealed; no B2 truth has been read.

| ID | Date | Decision | Reason |
|---|---|---|---|
| V2-a | 2026-10-06 | **v2 changes only the calibration map and the abstain thresholds.** Byte-identical to v1: everything under `src/exogargantua/resolver/` (features, candidate set, refinement, principled combiner) and the trained gradient-boosted classifier (`models/resolver/learned.pkl`). The calibration map stays a single temperature on the classifier logits (same family as v1, refitted). The principled combiner has no calibration map; only its abstain threshold is refitted. Abstain-threshold rule unchanged: smallest t in {0.50, ..., 0.99} with accuracy >= 0.95 on the calibration data (0.99 if none). v2 parameters live in `models/resolver_v2/`; v1 files are not modified. | Owner decision: fix calibration under realistic seeds without touching what the resolver computes. |
| V2-b | 2026-10-06 | **Calibration data (pre-registered).** About 2,000 fresh injections on TRAINING-split hosts only (fit + calib hosts in `data/b1_split.csv`; never a test host), parameters from docs/injection_design.md §2 and §1 sector rule, drawn with a new seed `numpy.random.default_rng(20261006)`: host uniform over training hosts (with replacement), then the same per-injection draws as B1. Each injection is searched with GPU BLS and with TLS exactly as in B1(ii); the resolver runs from both seeds. Temperature and thresholds are fitted on all these runs pooled (BLS- and TLS-seeded). Nothing else is fitted. | Calibration must reflect the seeds the resolver gets in practice (search peaks, not uniform wrong aliases). |
| V2-c | 2026-10-06 | **Order and reporting.** Fit v2 → tag `resolver-frozen-v2` → report v2 on the B1(ii) test split, labelled POST HOC (that split's v1 results had been seen when v2 was decided), next to v1. Then unseal B2 and score BOTH v1 and v2 with the baselines (raw BLS, TLS, Tschudi 2026a, SPOC TCE, QLP); **v2 is the primary resolver for B2, v1 secondary**; both are reported, and neither is chosen after unsealing. | Prevents choosing a version on holdout results. |
| V2-d | 2026-10-06 | **Not addressed in v2:** in 15-17 % of search-seeded B1(ii) test injections the true period is not among the 11 alias candidates (the search locked onto a non-alias peak). The paper reports this as a search-limited ceiling and future work. | Owner decision. |
