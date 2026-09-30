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
