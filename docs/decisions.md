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
