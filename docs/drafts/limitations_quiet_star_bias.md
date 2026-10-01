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
