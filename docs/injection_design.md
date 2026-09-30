# B1 injection design (pre-registered)

Committed on 2026-09-30, before any injection is generated or run. Implements the brief's B1 and
amendment A2. Anything not fixed here is decided by code in `scripts/make_injections.py` (to be
written), which reads only this document's parameters and the frozen files.

## 1. Hosts and data

- Hosts: the 3,000 frozen injection-pool stars (`data/injection_pool_log.csv`, decision include),
  after the download-time rules P07 (CROWDSAP >= 0.9 per sector) and P08 (uninjected BLS SDE < 9).
- Each injection uses k pinned sectors of one host, k drawn uniformly from 1..10 (capped by the
  host's available sectors), all starting within 365.25 d of the first sector used (the same
  one-year window as the B2 early data). Sector choice: the first k eligible sectors.
- Signals are multiplied into the compact-cache PDCSAP flux (real noise, real gaps), before
  detrending; everything downstream is identical to B2.

## 2. Grid

**30,000 injections**, 10 per pool star, drawn with `numpy.random.default_rng(20260930)` in the
order of `data/injection_pool_log.csv`.

| Parameter | Distribution |
|---|---|
| True period P | log-uniform, 0.5-30 d |
| Primary depth | log-uniform, 200 ppm - 5 % |
| Impact parameter b | uniform, 0-0.95 |
| Epoch | uniform within the first period of the data |
| Stellar R*, M*, Teff, logg | TIC v8.2 of the host (`data/stellar_params.csv`) |
| Limb darkening | quadratic, TESS band, Claret (2017) table interpolated in Teff/logg (snapshot committed before generation) |

Signal classes (fractions of the 30,000):

| Class | Fraction | Model |
|---|---|---|
| Planet | 60 % | `batman`, circular, a/R* from P and M*, R* (Kepler's law); Rp/R* from depth |
| EB, equal-depth twins | 15 % | two identical eclipses per orbit at phases 0 and 0.5 (the P vs P/2 case) |
| EB, unequal depths | 15 % | secondary depth = primary x U(0.05, 0.8), phase 0.5 |
| EB, eccentric | 10 % | e ~ U(0.05, 0.5), omega ~ U(0, 2pi); secondary phase and duration ratio from e, omega; secondary depth = primary x U(0.3, 1.0) |

EB eclipses are modelled with `batman` transit shapes for each eclipse (the brief's "equal-depth
twins, unequal depths, eccentric with offset secondaries"); the EB's true period is the orbital
period P.

## 3. B1(i): wrong-seed resolver runs (all 30,000)

- One seed per injection: P_seed = r x P_true with r drawn uniformly from the alias set
  R = {1/5, 1/4, 1/3, 1/2, 2/3, 1, 3/2, 2, 3, 4, 5} (seeded, stratified so each r has 30,000/11
  +/- 1 injections). r = 1 is included so the resolver is also scored when the seed is already right.
- Correct answer for the resolver: P_true (within 0.1 %); its candidate set around P_seed contains
  P_true because 1/r is in R for every r in R.
- No transit search is run for B1(i).

## 4. B1(ii): full-search subsample (2,000)

- Drawn from the 30,000 with `numpy.random.default_rng(20260930)`, uniformly, **before any
  injection is run**; the list is committed (`data/b1_search_subsample.csv`).
- Seeds: raw BLS peak (GPU BLS, accepted under A2 iv) and TLS peak, exactly as in B2.
- Reported separately from B1(i) (A2 iii).

## 5. Metrics (fixed by `docs/success_criteria.md`)

Alias accuracy within 0.1 %, confusion matrix by alias factor, ECE and reliability diagram,
abstain rate and accuracy-coverage curve, Wilson/bootstrap 95 % CIs. Injections are the only data
used to design and train the resolver (A4).
