# Legacy inventory (Phase 0)

Scope: every module and script of the rejected v1 project (AAS80459). All of it now lives in
`legacy/` (read only). Scripts that were deleted in the final v1 commit `404d67c` were restored
from `404d67c^` into `legacy/deleted_in_404d67c/` because they are the only record of how the
submitted numbers were produced.

Line numbers below refer to files under `legacy/`.

**Summary verdict: none of the v1 code is reused as-is.** The v1 benchmark cannot be reproduced
from the committed code, several reported results were edited by hand, and the "Bidirectional
Harmonic Validator" (the one component the reviewer found interesting) made period recovery no
better than the plain BLS peak, and on clean synthetic data it moves correct periods to wrong
aliases. Section 3 gives the root causes. The *idea* behind the validator (test the alias family
of a detected period instead of trusting the periodogram peak) is kept and rebuilt from scratch
in Phase 3.

---

## 1. Module inventory

| File | What it does | Reusable? | Known bugs / problems |
|---|---|---|---|
| `exoplanet_pipeline/ingestion.py` | Downloads SPOC 2-min LCs (all sectors) with lightkurve, de-duplicates 20 s vs 120 s per sector, normalises, "undilutes" by CROWDSAP, upward 5σ clip, stitches. Also `fetch_stellar_parameters` (TIC → NASA Archive → FITS header → solar fallback). | No. Rewrite. | **Loads `SAP_FLUX`, not PDCSAP** (L144-148), although the manuscript says PDCSAP. This line was added in the *final* commit, after the benchmark ran; the benchmark-era version (`ac43691`) instead applied `lk.CBVCorrector` **on top of PDCSAP** (double cotrending). Neither is plain PDCSAP. SAP is divided by CROWDSAP (L161-163) with no FLFRCSAP correction. 5σ clip is applied before detrending (L165-168), so it clips real variability. Stellar parameters silently fall back to the Sun (Rs = 1 R☉) and that fallback feeds the radius cap. |
| `exoplanet_pipeline/denoise.py` | Per-segment exponential "thermal decay" fit, then a quick BLS for a transit mask, then a Savitzky-Golay (2-day window, order 2) baseline with transits masked. | No. Replaced by wotan (Hippke et al. 2019). | The quick masking BLS (L82-108) runs on the **uncorrected** `lc`, not on the thermal-corrected flux, so mask and baseline come from different light curves. The mask uses the single global BLS maximum with no alias handling; when it picks the wrong period the real transits are not masked and the SG filter partly fits through them (depth suppression, V-shaping; likely contributor to the spurious b = 0.93 "grazing" fit). The "convexity bias" claim in the manuscript has no test or statistic anywhere in the code. |
| `exoplanet_pipeline/detection.py` | BLS search (`run_bls_search`), the "Bidirectional Harmonic Validator" (`resolve_fundamental_period`), a bootstrap FAP, alert banner. | Idea only. Rebuilt from scratch. | See Section 3.1. Also: the returned `'snr'` key holds the BLS *power* (a Δlog-likelihood in flux² units), not an SNR, and downstream code thresholds it at 7.1 as if it were an SNR. SDE is computed (L193-204) but the final choice uses power, not SDE (L259). |
| `exoplanet_pipeline/vetting.py` | Odd/even depth difference, secondary-eclipse significance at phase 0.45-0.55, TPF centroid shift. | No. Odd/even and secondary are re-implemented as *cited features* (Twicken et al. 2018; Thompson et al. 2018). | Odd/even epoch index uses `floor((t - t0)/P)` (L66-70), so every transit is **split across the odd and even sets** (points before mid-transit get the previous epoch number). Depth is `1 - min(binned flux)` (L75-80), a noise-biased statistic. Pass/fail is a fixed 0.005 (5000 ppm) absolute difference, independent of noise. Secondary search only at phase 0.5 (misses eccentric EBs). Centroid uses 1 sector and a fixed 0.333 px threshold. |
| `exoplanet_pipeline/ml_vetting.py` | Random forest "planet probability". | No. Discarded. | Trained on **hand-drawn synthetic feature distributions** (L187-254), not on light curves; e.g. planets are generated with `bls_power ≈ snr ∈ [10, 3000]`, but at inference `bls_power` is the astropy BLS power (different units and scale). Any "critical flag" caps the probability at ≤ 0.05 (L140-148), so the model output is mostly overwritten by the flags. |
| `exoplanet_pipeline/pipeline.py` | Orchestrates everything; applies "hard physical overrides" (L220-254) and post-MCMC radius/grazing rules (L300-357). | No. | Overrides: "SNR" < 7.1 (really BLS power); secondary > 3σ unless P < 10 d and depth > 1 %; odd/even > 0.005; Rp > 25 R⊕ from √depth × Rs (Rs possibly the solar fallback). The radius cap duplicates SPOC DV's > 25 R⊕ flag without citing it. |
| `exoplanet_pipeline/estimation.py` | batman + emcee fit of P, t0, Rp/Rs, a/Rs, i; Monte Carlo physical parameters. | No. | Uniform prior in *i* (not cos *i*), no stellar-density prior, circular orbit; the a/Rs-b degeneracy is unconstrained, so b is poorly determined, and input transits may already be distorted by the SG detrending. `np.random.seed(42)` at import time. |
| `exoplanet_pipeline/validate.py`, `validate_injection.py` | Small validation / injection harnesses. | No. | `injection_results.csv` (restored) has 2 rows, both not recovered: the injection-recovery test was never actually run at scale. Limb darkening fixed at [0.3, 0.1]. |
| `exoplanet_pipeline/report.py`, `math_utils.py`, `debug_plateau.py` | Report text, tiny helpers, a debug stub (imports a non-existent `pipeline.run_pipeline`). | No. | — |
| `run_pipeline.py`, `tests/` | CLI wrapper; two unit-test files (math helpers, ML vetter). | No. | Tests cover helpers only; nothing tests detection, the validator, or any reported number. |
| `deleted_in_404d67c/run_120_real_benchmark.py` | The script that produced the submitted benchmark. | No. | See Section 2. |
| `deleted_in_404d67c/fix_jsonl.py`, `patch_ml_vetting.py`, `patch_results.py` | Post-hoc edits of the results file. | No. | `fix_jsonl.py` **sets `exo_rejected = True` by hand for TIC 279999655** and then regenerates the benchmark table from the edited file. The manuscript itself says "two verified targets were manually patched into the dataset". |
| `deleted_in_404d67c/generate_appendix.py` | Writes `appendix_table.tex` from the jsonl. | No. | Column labelled "Vanilla BLS Period" is the `spoc_period` field; the summary table calls the same field "SPOC Baseline". |
| `manuscript.tex`, `appendix_table.tex`, figures, `output/` | Submitted paper and its artefacts. | No. | Numbers not reproducible from committed code (Section 2). |

## 2. How the submitted benchmark was actually produced

From `legacy/deleted_in_404d67c/run_120_real_benchmark.py`:

- **The "SPOC baseline" is not SPOC.** It is astropy BLS on undetrended PDCSAP flux, over a
  *linear* grid of 10 000 periods between 0.5 and 20 d (L137-145). For a 4-sector baseline the
  required frequency resolution is roughly 1/(3 × 100 d); a linear period grid of this size is
  far too coarse at short periods, so the baseline was handicapped.
- **Unequal data.** The baseline downloads `search_result[:4]` (the first four search rows,
  which can include duplicate 20 s/120 s products of the same sector) (L33), while the pipeline
  under test downloads *all* sectors (`ingestion.py` L62-64).
- **Aliases counted as successes.** "Recovered" means within 5 % of P, P/2 or 2P (L151, L169).
  A benchmark of a period-alias method that scores P/2 and 2P as correct cannot measure
  alias resolution. (The manuscript later states a 1 % tolerance "of the catalog period or a
  primary harmonic alias", which has the same problem.)
- **Centroid vetting was switched off** (`run_centroid=False`, L159), although the manuscript
  presents centroid vetting. This is why TIC 239372503 (a centroid-offset background object per
  the reviewer) is listed as "Candidate".
- `exo_rejected` = disposition FALSE_POSITIVE **or** BLS power < 7.1 (L174).
- Failed downloads were silently skipped (L123-125); 11 of 120 targets are missing and there is
  no log of which. The manuscript's "109" includes 2 targets patched in by hand.
- No seed, config or git hash was stored with the results.

Recomputing from the restored `real_benchmark_results.jsonl` (60 CP/KP planets, 1 % tolerance to
the catalog period, aliases *not* counted): plain BLS 26/60, v1 pipeline 27/60. The validator
fixed 8 periods and broke 7. Five of the 7 it broke have catalog periods inside the "perigee
danger zones" (Section 3.1). (Computed by the ad-hoc snippet in the Phase 0 session from the
legacy file; kept here only to explain the root cause, never for the paper.)

## 3. Root causes of the Table 2 errors

### 3.1 Non-harmonic period mismatches

`resolve_fundamental_period` (`detection.py` L19-145) does the following:

1. Trial multipliers `1/7 … 7` **including non-integers** 1.5, 2.5, 3.5, 4.5 and their
   reciprocals (L30-33). A 1.5× or 1/2.5× "alias" is not an alias of a strictly periodic transit.
2. Around each trial period it re-runs BLS over **±5 %** (`linspace(0.95 P, 1.05 P, 2000)`,
   L50) and keeps whatever peak it finds. The output can therefore sit anywhere in
   [0.95 m, 1.05 m] × P_seed, which produces ratios such as 2.90 (= 3 × 0.968) or 1.42
   (= 1.5 × 0.947).
3. It then applies a **"TESS perigee veto"** that multiplies the score by **0.01** for any
   candidate period in **10-17 d or 6.5-7.5 d** (L120-136; the same veto is also applied to the
   initial BLS peak, L225-247). Any true period in those ranges is almost never selectable.
4. The winner is the candidate with the highest lightkurve BLS `snr` (L61, L143). `D_max` is
   computed (L118) and printed but **never used**: the "95 % depth plateau" described in the
   manuscript (§Phase 4) is not implemented.
5. Candidates failing a 2σ/10 % odd/even check are **dropped** (L108-109). The manuscript says
   the pipeline "automatically doubles the period"; the code never does.

The two cases named in the brief:

| Target (legacy file) | Catalog P | v1 P | Ratio | Mechanism |
|---|---|---|---|---|
| TIC 232976128 | 13.0792 d | 4.3598 d | 0.3333 | Plain BLS found 13.0788 d. 13.08 d lies in the 10-17 d "danger zone", score × 0.01, so the 1/3 multiplier wins. |
| TIC 317548889 | 6.8660 d | 19.9388 d | 2.904 | 6.866 d lies in the 6.5-7.5 d zone, score × 0.01; the m = 3 candidate wins, and its ±5 % re-search lands at 0.968 × 3P. |

**Reproduced in code** (`scripts/legacy_audit/demo_validator_regression.py` →
`results/legacy_audit/validator_regression.json`): on synthetic two-sector light curves with a
clean 1500 ppm box transit, and the *true* period given as the seed, the legacy validator returns
13.0792 → 19.6154 d (×1.5), 6.8660 → 2.2886 d (×1/3), 11.1453 → 3.7152 d (×1/3), while 5.2 d
and 3.3 d are left alone.

### 3.2 Known planets labelled "Rejected (FP)"

Contributing mechanisms, all in code:

- **Broken odd/even test** (`vetting.py` L66-81, threshold L239 in `pipeline.py`): the parity
  split mixes halves of every transit, the depth statistic is `1 - min(bin)`, and the threshold
  is an absolute 5000 ppm. For deep or noisy targets this fires on genuine planets.
- **Wrong period first, then vetting at the wrong period.** Of the five planets the reviewer
  lists, three were vetted at a period that is not the planet's: TIC 237222864 (10.2889 →
  4.1156 d, ×0.4, a non-integer "alias" from step 1 above), TIC 36724087 (0.7684 → 18.671 d),
  TIC 321669174 (10.5054 → 5.3207 d, ≈ P/2 from the danger-zone penalty). Folding a planet at a
  wrong period produces fake secondaries and odd/even differences.
- **"SNR < 7.1" veto on BLS power**, a quantity that is not an SNR.
- **Radius cap with the solar fallback radius** when TIC has no radius (the TIC 279999655 case;
  that target was additionally flipped to "rejected" by hand in `fix_jsonl.py`).
- **ML vetter** trained on synthetic feature distributions whose `bls_power` scale does not match
  inference-time values.

Which specific veto fired per target was never logged in the results file. A re-run of the
benchmark-era code (`ac43691`) on the flagged targets is reported in Section 5.

### 3.3 Other reviewer points traced to code

- *TIC 239372503 (centroid-offset background object) listed as Candidate:* centroid vetting
  disabled in the benchmark run (Section 2).
- *TIC 425561347 "grazing, b = 0.932":* b comes from the emcee fit in `estimation.py` with a
  uniform-in-i prior, no density prior, on SG-detrended data whose transit mask may be wrong
  (`denoise.py`). SPOC finds b ≈ 0.3. The case study is dropped.
- *Figure 3 shows no significant odd/even difference:* consistent with the broken parity split
  and the absolute threshold; no significance was ever computed.
- *No false-alarm testing:* confirmed; no inverted/scrambled light-curve runs exist.
- *No code release/DOIs:* PyPI release existed, but not a Zenodo/MAST DOI.

## 4. What carries over to v2

- The **question** (is the detected period the true one, or a member of its alias family?).
- Nothing else. Detrending → wotan. Search seeds → BLS/TLS/TPS/QLP periods as given.
  Odd/even, secondary, density → re-implemented as *features* with cited provenance.
  All numbers → regenerated by committed scripts.

## 5. Re-run of benchmark-era code on flagged targets

See the Gate 0 report; results (if any) are in `results/legacy_audit/`.
