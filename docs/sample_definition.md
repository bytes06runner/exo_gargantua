# Sample definition (pre-registered)

Written and committed on 2026-09-30, **before any catalog snapshot or light curve was downloaded**
for this project. Every rule below is applied by code (`scripts/freeze_sample.py` and the
Kaggle jobs); every inclusion and exclusion is written to `data/sample_log.csv` automatically,
with reason, stage, UTC timestamp and git commit. A target may be excluded later **only** by a
criterion listed here (codes `X…`); anything else is a protocol deviation and is reported as such.

Checked facts this document relies on (documentation only, no sample data):
- MAST per-sector 2-min light-curve bulk scripts `tesscurl_sector_<N>_lc.sh` exist for sectors
  1-106 and list exact product filenames (`tess<stamp>-s<NNNN>-<TIC16>-0120-s_lc.fits`).
- MAST SPOC TCE statistics files `tess<stamp>-s<AAAA>-s<BBBB>_dvr-tcestats.csv` exist for every
  single-sector run and for multi-sector runs (e.g. s0001-s0013, s0014-s0026, s0001-s0096).
- The TESS orbit-times table `https://tess.mit.edu/public/files/TESS_orbit_times.csv` gives
  start/end of every orbit per sector.
- The published TESS EB catalog (Prša et al. 2022; VizieR J/ApJS/258/16; MAST HLSP `tess-ebs`
  v1.0) has **no ambiguous-period column** (see Section 3.2 for the pre-registered proxy).

## 1. Snapshots (all saved raw in `data/raw/` with download date in the filename)

| ID | Source | File |
|---|---|---|
| S-TOI | ExoFOP TOI table, CSV export (`exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv`) | `data/raw/exofop_toi_<YYYYMMDD>.csv` |
| S-CTOI | ExoFOP CTOI table, CSV export | `data/raw/exofop_ctoi_<YYYYMMDD>.csv` |
| S-PS | NASA Exoplanet Archive `pscomppars` via TAP (all columns) | `data/raw/pscomppars_<YYYYMMDD>.csv` |
| S-EB | TESS EB catalog v1.0 (Prša et al. 2022), MAST HLSP CSV | `data/raw/tess_ebs_prsa2022_v1.0_<YYYYMMDD>.csv` |
| S-EB2 | TESS Ten Thousand EB catalog (Kostov et al. 2025) | `data/raw/kostov2025_t10k_<YYYYMMDD>.csv` (exclusion of injection-pool stars only) |
| S-ORB | TESS orbit times | `data/raw/tess_orbit_times_<YYYYMMDD>.csv` |
| S-LC | MAST 2-min LC bulk scripts, sectors 1-106 (parsed to a table; raw scripts kept in the Kaggle dataset) | `data/raw/mast_lc_products_<YYYYMMDD>.csv.gz` |
| S-TCE | All MAST SPOC `dvr-tcestats.csv` files listed on the TCE bulk page | `data/raw/spoc_tcestats/…` (Kaggle dataset) + index in git |

Snapshots are taken once, by a script, and never refreshed for this paper. The snapshot date is
recorded in every results file.

## 2. Real-data sample (benchmarks B2 and B3)

Unit: **one TOI** (one signal). A host with several TOIs contributes several units.

Inclusion (all must hold; failures are logged with the code shown):

| Code | Rule |
|---|---|
| X01 | TOI's TIC has SPOC 120-s light-curve products in **≥ 2 sectors** in S-LC. (20-s "fast" products are not used.) |
| X02 | Catalog period finite and **0.5 ≤ P ≤ 30 d**. Blank or zero periods (single transits) and longer periods are out of scope. |
| X03 | Not a duplicate: if two TOIs share a TIC and their periods agree within 0.1 % and epochs within 0.1 d, keep the lower TOI number. |

All TFOPWG dispositions (CP, KP, PC, APC, FP, FA, blank) are kept. Disposition decides the truth
tier (Section 3), not inclusion.

Pinned sectors: for each included TOI, **all** SPOC 120-s sectors present in S-LC for its TIC,
with exact filenames. Later code loads only these files (see `docs/decisions.md`).

Download-time exclusions (applied by the download job, logged automatically):

| Code | Rule |
|---|---|
| X10 | A pinned file fails to download after 3 retries: the *sector* is dropped (logged). |
| X11 | A sector with < 50 % finite PDCSAP cadences after the lightkurve `default` quality bitmask: the *sector* is dropped (logged). |
| X12 | After X10/X11 fewer than 2 sectors remain: the *TOI* is excluded. |
| X13 | A downloaded file's MD5 or `DATA_REL` differs from the pinned manifest on a later run: the run **aborts** (not an exclusion; indicates MAST changed a file). |

Crowding (`CROWDSAP`) is recorded per sector but is **not** an exclusion for the real sample.

## 3. Ground truth (B2)

Truth is assigned per TOI from independent catalogs, never from any method under test.
Alias set used for matching: R = {1/5, 1/4, 1/3, 1/2, 2/3, 1, 3/2, 2, 3, 4, 5}.

### 3.1 Tier A: confirmed planets (NASA Exoplanet Archive, S-PS)

A TOI gets Tier-A truth P_true = `pl_orbper` if a `pscomppars` row with the same `tic_id`
satisfies both:
- |P_TOI / (r · P_PS) − 1| < 0.01 for some r ∈ R, and
- the TOI epoch coincides with a transit of the PS ephemeris: the distance from the TOI epoch to
  the nearest PS-predicted transit is < max(0.5 × TOI duration, 3σ of the propagated PS epoch).

Additionally `pl_orbpererr1 / pl_orbper < 1e-3`. If **more than one** PS planet matches, the TOI
gets no Tier-A truth (code T-AMB, logged).

### 3.2 Tier B: eclipsing binaries (Prša et al. 2022, S-EB)

Same matching rule against the EB ephemeris (`period`, `bjd0`), using `signal_id` rows of the
same TIC. P_true = catalog `period`.

The published catalog has no ambiguous-period flag. **Pre-registered proxy:** an EB is
"period-ambiguous" (tier **B-amb**) if its period is missing, or if both the polyfit primary and
secondary depths exist with |Dp − Ds| / Dp < 0.10 and the secondary phase separation from the
primary is within 0.02 of 0.5 (equal-depth twins at phase 0.5, where P and P/2 cannot be told
apart from photometry alone). Tier B-amb targets are reported separately and **never** count
toward the primary B2 metric.

### 3.3 Tier C: catalog reference (secondary analysis only)

Any other TOI with TFOPWG disposition PC/APC/CP/KP: reference period = TOI catalog period
(derived by SPOC/QLP from multi-sector data). Tier C is **not** ground truth; it is used only in
a secondary analysis that is labelled as agreement with the catalog.

Primary B2 metric: Tier A ∪ Tier B (excluding B-amb). If a TOI qualifies for A and B, A wins.

## 4. Temporal holdout split (B2)

Per TOI, with sector start = start of the sector's first orbit in S-ORB:
- t_first = start of the earliest pinned sector.
- **Early (input) sectors E** = pinned sectors whose start < t_first + 365.25 d.
- **Later (check) sectors L** = all other pinned sectors.

B2 eligibility (all must hold, logged as `b2_eligible` with reason):

| Code | Rule |
|---|---|
| H1 | Tier A or Tier B truth (not B-amb). |
| H2 | L is non-empty. |
| H3 | The truth ephemeris predicts ≥ 2 transits in E with ≥ 50 % of their in-transit cadences present after X11 (computed from the downloaded pinned data). This selects on the truth period identically for every method compared. |

Everything else about the target (sector count, crowding, disposition) is recorded but not used
to select.

## 5. Injection pool (B1)

Target pool size: **3,000 stars**, drawn with seed `20260930`.

Eligibility (from S-LC, TIC v8.2, Gaia DR3, S-TCE, S-TOI, S-CTOI, S-EB, S-EB2):

| Code | Rule |
|---|---|
| P01 | SPOC 120-s products in ≥ 2 sectors. |
| P02 | Not a TOI host (S-TOI) and not a CTOI host (S-CTOI). |
| P03 | No SPOC TCE on the TIC in any single- or multi-sector `tcestats` file (S-TCE). |
| P04 | Not in the Prša et al. 2022 or Kostov et al. 2025 EB catalogs. |
| P05 | Gaia DR3 source (via TIC `GAIA` id) not flagged `phot_variable_flag = 'VARIABLE'`, and RUWE < 1.4. |
| P06 | TIC v8.2: 3200 ≤ Teff ≤ 7000 K, logg ≥ 4.0, 7 ≤ Tmag ≤ 13, radius and mass present. |
| P07 | At download: CROWDSAP ≥ 0.9 in every pinned sector (sectors failing it are dropped; star excluded if < 2 remain). |
| P08 | At download, before any injection: a BLS search (Section 6 configuration) of the uninjected, pinned light curve gives SDE < 9. Otherwise excluded as `intrinsic_signal`. |

Stratification by number of pinned sectors n: {2-3, 4-6, 7-12, ≥ 13}, 750 stars per stratum
drawn uniformly at random from eligible stars (if a stratum has fewer, all are taken and the
shortfall is reported). Stars excluded by P07/P08 are **not** replaced (the realised pool size
is reported). Injections then use sector subsets of each star to cover 1 to ≥ 10 sectors.

## 6. Search configuration for seeds and baselines (fixed now)

- Flux: PDCSAP, lightkurve `default` quality bitmask, per-sector median normalisation.
- Detrending: wotan `biweight`, window 0.75 d, `break_tolerance` 0.5 d (Hippke et al. 2019).
- Searches (BLS and TLS alike) run on 10-minute bins of the detrended light curve; resolver
  features use unbinned data.
- Period range: 0.5 d to min(30 d, 0.95 × time span of the searched data).
- BLS (astropy `BoxLeastSquares`): durations {0.04, 0.06, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30} d;
  frequency grid from `autoperiod` with `minimum_n_transit=2`, `frequency_factor=1.0`;
  objective `likelihood`; "raw BLS peak" = argmax power.
- TLS: package defaults (oversampling factor 3, duration grid step 1.1), stellar radius/mass
  ranges from TIC; "TLS peak" = the returned best period.
- Exact versions and commits: `docs/baselines.md`.

## 7. What is recorded per TOI in `data/sample_log.csv`

`toi, tic, decision, reason, stage, timestamp_utc, git_commit, snapshot_date, sectors_all,
sectors_early, sectors_later, product_files, lc_listing_date, data_release, download_date,
truth_tier, truth_period, truth_source, b2_eligible, b2_reason`

Per-file pins (filename, source bulk script, its Last-Modified date, and after download the
FITS `DATA_REL` and MD5) are in `data/pinned_products.csv`. Injection-pool stars are logged in
`data/injection_pool_log.csv` with the same decision/reason/stage/timestamp/commit columns.
