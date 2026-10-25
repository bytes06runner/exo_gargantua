# Gate 2 report (2026-09-30)

Every number below is copied from a committed file named next to it. No benchmark accuracy has
been computed or looked at.

## 1. Pre-registration

`docs/sample_definition.md` was committed at `0961d05` **before** any catalog snapshot or light
curve was downloaded. Later changes to the pipeline were bug fixes (below), not rule changes.

## 2. Frozen sample (`data/FROZEN.json`, manifest `results/kaggle/freeze_sample/v2_20260930T1025/freeze/freeze_manifest.json`)

Run of record: Kaggle kernel `exog-freeze-sample-v2`, code commit `835f583`, 190 s on 4 CPU cores.

| Item | Count |
|---|---|
| TOI snapshot rows (ExoFOP, 2026-09-30) | 8,148 |
| Excluded X01 (< 2 SPOC 120-s sectors) | 2,230 |
| Excluded X02 (period missing/zero or outside 0.5-30 d) | 580 |
| Excluded X03 (duplicates) | 0 |
| **Included TOIs** | **5,338** |
| Truth tier A (pscomppars) / B (Prša EB) / B-amb / C / none | 1,146 / 110 / 1 / 3,334 / 747 |
| B2 candidates (H1 + H2 pass; H3 checked at download) | 1,204 (1,096 A, 108 B) |
| Injection pool (P01-P06 pass) | 3,000 (750 per sector stratum) |
| Pinned LC files | 63,559 (34,340 TOI hosts, 29,219 pool) |

MAST listing covered sectors 1-106 (`mast_lc_scripts` Last-Modified dates in the manifest); SPOC
TCEs: 131 `tcestats` files, 340,505 rows. Every file's SHA-256 is in `data/FROZEN.json` and
`tests/test_frozen.py` fails if any frozen file changes.

Protocol notes (all logged in `docs/decisions.md`):
- The first freeze run (`exog-freeze-sample`) is void: a float64 parsing bug corrupted 19-digit
  Gaia ids and wrongly failed pool stars on P05. Found in a local test before either run
  finished; v2 was declared the run of record before its results were seen.
- MAST filenames carry a varying 4-digit configuration id; an early parser accepted only
  `0120` (caught by the 3-sector debug run, test added).

### Items for your decision (not changed by me)

1. **Confirmed planets with Tier-B truth.** Six CP/KP TOIs got Tier B (EB-catalog) truth because
   their pscomppars epoch or period did not pass the Tier-A match: TOI 143.01, 1165.01 (KELT-23 A b),
   1422.01 (TOI-1422 b), 1779.01 (LP 261-75 b), 2119.01, 4495.01. The EB-catalog period equals the
   TOI period to < 0.03 % for five of them. For **TOI-4495.01** (CP, 670 ppm) the EB-catalog period
   (5.179905 d) differs from pscomppars TOI-4495 c (5.18553 d) by 0.1 %, which is the benchmark's
   correctness tolerance. Options: (a) keep the rule as frozen and discuss these six in the paper;
   (b) pre-results amendment: Tier B only for TOIs not dispositioned CP/KP (they would drop out of
   B2 or need a Tier-A fix). Rule 6 in `CLAUDE.md` requires a flagged discussion either way.
2. **EB ambiguity proxy fires once** (TOI 1127.01). The published Prša catalog has no
   ambiguous-period flag; the pre-registered proxy is strict. This is reported as is.

## 3. Baselines (`docs/baselines.md`)

BLS (astropy 8.0.1), TLS 1.32, Tschudi 2026a (`mdwarf-transit-survey` @ `ff62f364`, MIT,
`KNOWN_PLANETS` lookup emptied to avoid truth leakage), SPOC-1 and SPOC-E TCE periods from the
MAST `tcestats` files, and QLP-hist / SPOC-hist from the Guerrero et al. (2021) release
(2020-10-18) table.

## 4. Kaggle smoke run (`results/kaggle/smoke_run/v1_20260930T1241/smoke/smoke_results.json`)

Kernel `exog-smoke-run`, code commit `2d27dda`, 20 targets drawn with seed 20260930 from the B2
candidates. Hardware: 4 CPU cores, 33.7 GB RAM, **no GPU**; Python 3.12.13.

| Measure | Value |
|---|---|
| Total wall-clock | 2.18 h |
| Wall per target: median / p95 / max | 22 s / 1,407 s / 3,009 s |
| Download per target | < 2.5 s (22.8 MB/s) |
| BLS CPU use | 1.00 core (astropy BLS is single-threaded) |
| TLS CPU use | 3.78 of 4 cores |
| GPU use | none (no GPU attached; neither search uses one) |
| Raw FITS per target: median / mean | 10.7 / 18.6 MB (1.94 MB per file) |
| Slim cache per target: median / mean | 1.07 / 1.75 MB (0.183 MB per file) |
| H3 (≥ 2 truth transits in early data) | 18 of 20 pass |
| File exclusions | 1 sector (X11) |

Cost is dominated by the early-window **span**: the pre-registered BLS grid
(`autoperiod`, `frequency_factor = 1`, 0.04 d shortest duration) grows with span, so a 354-d
window costs ~2,400 s of BLS alone (TOI 1453.01), while single-sector targets take ~13 s.

Private Kaggle Dataset created from the smoke cache: `srijeetbanerjee/exog-lc-cache-smoke`
(slim arrays + `smoke_results.json` + `FROZEN.json`).

## 5. Cost estimate (`results/cost_estimate.json`, `scripts/estimate_costs.py`)

Model: cost = c × N_points × N_trial_periods per method, with the exact grid sizes of the
pre-registered searches reproduced per target; it reproduces the 20 measured runtimes to within
about 20 %.

| Item | Estimate |
|---|---|
| Full download (63,559 files) | 123 GB, ~1.5 h in one session |
| Slim cache (Kaggle Dataset size) | 11.6 GB |
| B2 seed searches (1,204 targets), as pre-registered | BLS 138 h + TLS 71 h = 209 h wall (4-core sessions) |
| … of which targets with 240-400 d early span (227) | 189 h |
| … with BLS run 4-at-a-time across cores (same results) | ~105 h wall |
| B1, 30,000 injections, sectors within one year | ~5,700 h serial, ~3,000 h with parallel BLS |

Current Kaggle quota (`results/kaggle/quota_20260930.json`, from `kaggle quota`):
GPU 1.57 h used, **28.43 h remaining** of 30 h; TPU 20 h; refresh 2026-10-03. The API reports
no CPU quota and no session limits; those are not assumed here.

**Conclusion:** B2 is feasible on CPU (~105 h wall, i.e. about nine 12-hour sessions, or fewer
if Kaggle allows concurrent CPU sessions on your account). **B1 is not feasible with the
pre-registered search configuration on Kaggle CPU.** It needs a decision (Section 6).

## 6. Options that need your approval (none taken)

1. Run BLS 4 targets at a time across cores. Same results, ~2× less B2 wall time. *Recommended.*
2. For B1, benchmark a GPU batched BLS (the brief's intended GPU use) on a 50-injection subset
   before committing GPU quota; TLS remains CPU.
3. Amend the search configuration *before any results*: a coarser BLS frequency grid
   (e.g. `frequency_factor` > 1) and/or a smaller B1 grid (e.g. 10,000 injections). This changes
   pre-registered settings, so it must be your decision and would be logged as an amendment.
4. Decide the Tier-B confirmed-planet question (Section 2).

Nothing heavy has been launched beyond the freeze and the smoke run.
