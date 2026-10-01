# DRAFT amendment A6: P08 SDE computed on a TLS-detrended spectrum

**Status: draft, NOT in effect.** It takes effect only when the owner approves it and it is copied into
`docs/decisions.md` with the approving commit. Until then, P08 is the rule pre-registered in
`docs/sample_definition.md` §5 as amended by A5, with the SDE definition in decision V1.

## Reason

P08 excludes a pool star if an uninjected BLS search over its injection window gives SDE >= 9. The
threshold 9 comes from TLS (Hippke & Heller 2019, A&A 623, A39), where SDE is computed on a spectrum
that has first been detrended with a running median. Our pre-registration (decision V1) applied the
threshold to the raw BLS spectrum, (max − mean)/std, with no detrending. In the 50-star validation
(`results/p08_validation_report.json`, GPU half), exclusions track the length of the screening window,
and the excluded stars' peaks sit at the edges of the trial-period grid or near the 13.7 d TESS orbit,
not at stellar periods. The raw-spectrum SDE therefore responds to red-noise structure and grid edges
that TLS's detrending is designed to remove. The threshold of 9 has its stated meaning only on the kind
of spectrum it was calibrated on.

## New rule

P08 is unchanged except for how SDE is computed:

- **Same search.** Same light curve and screening window (P07, A5); same §6 preparation (PDCSAP,
  wotan biweight 0.75 d, 10-min bins); same BLS (autoperiod grid, durations, likelihood objective);
  same engines.
- **Same threshold and same rule.** Exclude if SDE >= 9. No replacement of excluded stars.
- **New SDE.** SDE is computed with TLS's own spectrum-detrending code, called unmodified:
  - transitleastsquares **1.32** (5 Apr 2024), file `transitleastsquares/stats.py`, function
    `spectra(chi2, oversampling_factor)`.
  - It uses `transitleastsquares/helpers.py::running_median` with a kernel of
    `oversampling_factor × tls_constants.SDE_MEDIAN_KERNEL_SIZE` = 3 × 30 → 91 trial periods.
  - Method: Hippke & Heller (2019), Sect. 2.5.
  - Implementation: `src/exogargantua/sde.py::sde_a6`; tests in `tests/test_sde.py`.

  Steps, per star:

  1. **TLS's own period grid.** Build the grid TLS itself would search on this light curve:
     `transitleastsquares.period_grid(R_star, M_star, time_span, period_min, period_max,
     oversampling_factor=3, n_transits_min=2)`.
     - The arguments are those our TLS seeds use (`search.tls_peak`).
     - R_star and M_star come from `data/stellar_params.csv`, or default to TLS's 1.0 when missing.
     - period_min and period_max are our §6 limits.
  2. **Read the BLS spectrum on that grid.** For each TLS trial period, take the BLS power at the
     nearest BLS trial period. The BLS grid is 70–770× finer than TLS's on the smoke targets, so this
     equals evaluating our BLS at TLS's periods to within a small fraction of TLS's grid spacing.
  3. **Convert to chi2.**
     - astropy BLS with the likelihood objective and no uncertainties uses unit weights on
       median-subtracted flux, so power = ½ Δχ².
     - Therefore χ²(P) = Σ(y − median y)² − 2·power(P).
  4. **Compute SDE.** Call `transitleastsquares.stats.spectra(chi2, 3)` unmodified. Its detrended SDE
     (fifth return value) is the A6 SDE.

  Why TLS's grid and not the BLS grid:
  - TLS's kernel is 91 points of *its* grid. Its SDE distribution, and hence the meaning of 9,
    depends on that grid's density.
  - Applying the 91-point kernel literally on the 70–770× denser BLS grid is not TLS's method. The
    kernel is then narrower than a transit peak, so the median follows the peaks.
  - On simulated white noise with no signal, that literal variant inflates the noise SDE well above
    the raw and A6 values (`scripts/a6_synthetic_check.py` → `results/a6_synthetic_check.json`). That variant is recorded in validation as a diagnostic only
    (`sde_dense_tlspts`, `sde_dense_scaled`) and never decides anything.

## Declarations

- **No benchmark results existed when this was drafted.** No injection had been generated, no injected
  search had run, and no resolver had been built or scored. The B2 holdout seeds are sealed under A4:
  no accuracy against holdout truth has been computed. The only P08 outcomes seen are those of the
  50-star validation sample (`data/p08_validation_stars.csv`), under the raw rule.
- **Pool size is accepted as it falls.** We accept whatever pool size A6 produces. No further change
  to P08 will be made, whatever the screen yields.
  - If the A6 pool is small, results are reported by sector-count stratum with wider intervals
    (option a). Injections per star stay at 10 (option b is rejected).
- **Sensitivity.** The paper reports pool size and injection results under both the raw-SDE rule (V1)
  and the A6 rule. Every screen output records both (`sde`/`p08` and `sde_a6`/`p08_a6`).
  `scripts/p08_validation_report.py` keeps the raw-rule analysis.

## Validation (before the screen is queued)

Re-run the 50 validation stars on the GPU (two T4s) and on CPU astropy, saving spectra. For each
engine, report:
- flips across 9 under A6;
- the maximum |ΔSDE| between engines;
- agreement of the peak period;
- the projected pool by stratum under A6 (Wilson 95% intervals).
