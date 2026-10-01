# DRAFT limitations note: quiet-star (and short-window) bias of the injection hosts

Not final text; numbers to be regenerated from results files when the screen is complete.

The injection-recovery benchmark (B1) uses host stars selected to be quiet and uncrowded: no TOI,
CTOI or SPOC TCE on the star, absent from the Prša et al. (2022) and Kostov et al. (2025) EB
catalogs, not flagged variable in Gaia DR3, RUWE < 1.4, CROWDSAP >= 0.9 in every sector used, and
an uninjected BLS search with SDE < 9 over the data the injections use (P08, amendment A5).
Injected signals therefore sit on the least problematic light curves TESS provides. Alias
resolution on real TOI hosts (B2) involves more active, more crowded and more systematics-affected
stars, so B1 accuracy and calibration should be read as upper bounds for such stars; B2 is the
check on them, and the paper reports both.

The P08 screen is not neutral with respect to light-curve length. In the 50-star validation
(`results/p08_validation_report.json`) the screen removed 20 of 45 stars, and removal tracked the
screening window: median 7.5 sectors (271 d) for removed stars versus 2 sectors (76 d) for kept
ones; 8 of 9 stars with >= 13 pinned sectors were removed. The removed stars' BLS peaks lie near the
edges of the trial-period grid (11 below 1 d, close to the 0.5 d minimum; 5 within 10 % of the
maximum period) or near the 13.7 d TESS orbit, rather than at astrophysical periods. This is
consistent with the pre-registered SDE being computed on the raw BLS spectrum with a fixed
threshold, so that longer windows (more trial periods, more red-noise structure) exceed SDE = 9
more often. The consequence is that the surviving pool is weighted towards short windows: B1
results for injections spanning many sectors rest on fewer host stars and carry wider intervals,
which the paper must state explicitly (per-sector-count results with their CIs).

Amendment A6 changes how the P08 SDE is computed but not what was found. SDE is now computed from a
running-median-detrended spectrum on TLS's own period grid, with TLS 1.32's own code. The threshold
of 9, the search and the no-replacement rule are unchanged. A6 reduces the window-length dependence
but does not remove it.
- **Fewer exclusions.** In the same 50-star validation (`results/p08_a6_validation_report.json`),
  A6 excludes 11 of the 45 searched stars, against 20 under the raw rule.
- **Peaks of the A6 exclusions.** Most sit at instrumental rather than stellar timescales:
  - 5 near one TESS sector (26.7–28.4 d);
  - 3 near the 13.7 d spacecraft orbit (13.2–14.0 d);
  - 3 at 1.0–1.5 d.
- **Window length.** Excluded stars still have longer screening windows (median 7 sectors) than kept
  ones (median 3.5).

The screen was not changed again: the pool is accepted as A6 leaves it (decisions.md, A6). So the
injection hosts remain weighted towards shorter windows and away from stars with strong
sector- or orbit-period systematics. B1 results for many-sector light curves, and for light curves
with such systematics, should be read with that in mind.
- **Reporting.** The paper reports B1 by sector-count stratum with Wilson intervals, and gives the
  pool size and injection results under both the raw rule and A6.
