# Baselines for B2 (pinned 2026-09-30, before any benchmark run)

Every baseline is run on (or taken for) the **same TOIs** and, where it runs on light curves,
on the **same pinned early-sector files** as our resolver. Each is scored with the same rule:
correct iff |P / P_true − 1| < 0.001 (`docs/success_criteria.md`). A baseline that gives no
period for a TOI counts as incorrect; accuracy is also reported on the subset where the
baseline returned a period, with its coverage.

Shared software versions (from `pyproject.toml`, installed from the pinned commit on Kaggle):
numpy 2.0.2, scipy 1.17.1, astropy 8.0.1, lightkurve 2.6.0, wotan 1.10, transitleastsquares
1.32, batman-package 2.5.3. Each results file records the installed versions it actually ran.

## 1. Raw BLS peak

- Implementation: `astropy.timeseries.BoxLeastSquares`, astropy **8.0.1**.
- Input: early sectors only, PDCSAP, `default` quality mask, wotan biweight 0.75 d, 10-min bins.
- Grid: `autoperiod(durations, minimum_n_transit=2, frequency_factor=1.0)`, periods clipped to
  [0.5 d, min(30 d, 0.95 × span)]; durations {0.04, 0.06, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30} d;
  objective `likelihood`.
- Output: period at maximum power.
- Multi-TOI hosts: transits of the *other* TOIs on the host are masked (catalog ephemerides,
  ±1 duration) before the search, identically for every light-curve baseline and our resolver.

## 2. TLS peak

- Implementation: `transitleastsquares` **1.32** (PyPI), Hippke & Heller (2019).
- Same input and masking as BLS. Package defaults (oversampling factor 3, duration grid step
  1.1, `use_threads` = all cores); `R_star`, `M_star` and their ranges from TIC v8.2
  (`data/stellar_params.csv`); period limits as BLS.
- Output: `results.period`.

## 3. Rule-based harmonic correction (Tschudi 2026a)

- Code of record: `gitlab.com/yohanntschudi/mdwarf-transit-survey`, commit
  **ff62f3643ed12f36f32244165fae7242a0a47307** (2026-08-31), MIT license:
  `pipeline/03_search_tls.py` (`__version__ = "10.0.0"`, the TLS search with the harmonic
  breaker: equal-depth secondary, odd/even, equal-depth thirds, transit-count correlation) and
  `pipeline/03b_harmonic_cleanup.py` (v1.3).
- Why this repository: the 2026a paper's own repository (`mdwarf-noise-frontier`, commit
  2be009507a74e0c5fc299e0b8f7d3930c1d8d490) contains data and results only, no code; the
  survey repository holds the detection chain that the 2026b paper describes as the 2026a
  pipeline applied to TOI hosts.
- Integration: a thin adapter (to be written in Phase 4, `src/exogargantua/baselines/tschudi.py`)
  feeds our pinned early-sector light curves into its data structures and calls its harmonic
  correction on the TLS peak from Section 2. Two deliberate changes, both to prevent truth
  leakage and both reported in the paper:
  1. its built-in `KNOWN_PLANETS` lookup (`check_known_planet`) is emptied;
  2. no catalog information other than what our resolver also receives is passed in.
  Everything else stays at its defaults, including thresholds tuned for M dwarfs. The paper
  will state that the method was designed for M dwarfs and that we apply it outside that domain.
- If the adapter cannot reproduce the behaviour on the 2026a paper's own worked examples, we
  report that instead of silently changing its code.
- Implementation (2026-09-30): `src/exogargantua/baselines/tschudi.py` calls
  `resolve_harmonic_alias` exactly as their pipeline does (`model=None`, their TLSConfig values for
  the keys it reads, harmonic thresholds at their in-function defaults, `found_periods=[]`), on the
  same TLS result as our TLS seed, inside the B2 TLS shards (one TLS run per TOI). Positive/negative
  control in `tests/test_tschudi_adapter.py`: a synthetic 2.0 d signal seeded at 4.0 d is corrected
  to 2.0 d (trigger: equal-depth secondary); seeded at 2.0 d it is left unchanged.

## 4. SPOC TCE periods

Source: MAST SPOC `dvr-tcestats.csv` files (list and download date in
`results/kaggle/freeze_sample/.../freeze_manifest.json`), TCEs on TOI hosts saved to
`data/spoc_tces_toi_hosts.csv.gz`.

- **SPOC-1 (single-sector; the baseline named in `docs/success_criteria.md`):** the TCE from
  the single-sector run of the TOI's *first* early sector.
- **SPOC-E (best early multi-sector):** among runs whose whole sector span lies inside the early
  window E, the one with the most sectors; ties go to the later processing date.
- TCE ↔ TOI match: same TIC; |P_TCE / (r · P_TOI) − 1| < 0.01 for some r in the alias set R;
  and the TCE epoch within max(0.5 × TOI duration, 0.1 d) of a TOI-predicted transit. If several
  TCEs match, the one with the highest `tce_model_snr`. No match → the baseline has no period.

## 5. QLP (and SPOC) historical catalog periods

The current ExoFOP TOI table holds periods *updated* with later data, so it cannot serve as an
early-data baseline. Instead:

- **QLP-hist / SPOC-hist:** the published TOI catalog of Guerrero et al. (2021), release
  2020-10-18 (VizieR J/ApJS/254/39, snapshot `data/raw/guerrero2021_toi_table2_20260930.dat`,
  1,098 QLP and 1,143 SPOC TOIs, with the sectors each entry was based on). For a TOI whose
  table sectors are all within its early window E, the table period is taken as the QLP-hist
  (or SPOC-hist) baseline according to the table's `Pipeline` column. Other TOIs: no period.
- Coverage is limited to TOIs ≤ 2321 first observed in Sectors 1-26; the paper reports QLP-hist
  on that subset only.

## 6. Reference only (not a baseline)

- Current ExoFOP TOI period: Tier C reference (`docs/sample_definition.md` §3.3).
- The legacy v1 "Bidirectional Harmonic Validator": not a baseline (its failure modes are
  documented in `docs/legacy_inventory.md`; the reviewer did not ask for it to be compared).
