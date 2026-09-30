# Related work (Phase 1)

All keys refer to `paper/refs.bib`, which is generated from `paper/bib_sources.csv` by
`scripts/build_bib.py` and checked by `scripts/verify_bib.py` (69/69 entries resolve at
Crossref/DataCite/arXiv; title overlap ≥ 0.6). The structured version is
`docs/related_work_matrix.csv`; the gap analysis is `docs/novelty_check.md`.

"Checked" below means what was actually read: *full text* (PDF text grepped for the relevant
sections), or *abstract* only.

## 1. Transit search

- **SPOC TPS** (Jenkins2002; Jenkins2010; Jenkins2016): wavelet-based adaptive matched filter
  run per sector and multi-sector; produces TCEs with MES. Not FFT-based periodogram
  "aliasing"; the brief's point that the v1 "Fourier alias trapping" wording was wrong stands.
  *Abstract.*
- **SPOC / Kepler DV** (Twicken2018; Li2019): diagnostic tests on each TCE, including odd/even
  depth and epoch tests, weak-secondary test, ghost diagnostic, centroid tests; Twicken2018
  notes that circular EBs can be "folded onto primary events" so the correct EB period "would
  be twice the period reported by DV". Li2019: transit-model fitting and multiple-planet search.
  *Full text (grep).*
- **QLP** (Huang2020a; Huang2020b; Kunimoto2021; Kunimoto2022; Kunimoto2023qlp): FFI photometry
  and BLS search; since Sector 59 a GPU-based search. *Abstracts.*
- **BLS** (Kovacs2002), **TLS** (Hippke2019tls): TLS returns `odd_even_mismatch`,
  `transit_count`, `distinct_transit_count`, `empty_transit_count` and per-transit SNRs.
  *Full text (grep).*
- **wotan** (Hippke2019wotan): our default detrending; no novelty claimed.
- **Independent Kepler searches at the detection limit** (Ivashtenko2025; Robnik2026):
  permutation/injection-based false-alarm probabilities per TCE, per-transit vetting, Bayes
  factor statistic. Relevant as the model for how to test false alarms honestly. *Abstracts.*
- **SHERLOCK** (DevoraPajares2024): end-to-end user pipeline. *Abstract.*

## 2. Vetting

- **Kepler Robovetter** (Coughlin2016; Thompson2018) and **ephemeris matching** (Coughlin2014).
  Relevant details: model-shift uniqueness; `PERIOD_ALIAS` flag (N:1, N ≥ 3, informational);
  `PLANET_PERIOD_IS_HALF`; reliability measured with simulated false alarms (inverted and
  scrambled light curves). Ephemeris matching accepts 2:1 / 1:2 period matches. *Full text.*
- **LEO-Vetter** (Kunimoto2025leo): automated flux- and pixel-level vetting for TESS occurrence
  rates. Odd/even used to fail EBs at half period; data-gap test (≥ 50 % of transits near gaps
  → fail) aimed at 13.7 d scattered light; SNR-consistency (CHI) test; model-shift uniqueness.
  *Full text.* The v1 "perigee veto" was an unprincipled, uncited version of the data-gap test.
- **ExoMiner / ExoMiner++ / ExoMiner++ 2.0** (Valizadegan2022; Valizadegan2025; Martinho2026):
  deep-learning planet/non-planet classifiers and vetting catalogs. ExoMiner++ lists incorrect
  (2× / ½×) TCE periods as a source of label noise and misclassification. *Full text (grep) for
  ExoMiner++; abstracts otherwise.*
- **AstroNet family** (Shallue2018; Yu2019; Tey2023), **DART-Vetter** (Fiscale2025),
  **WATSON-Net** (DevoraPajares2025): neural vetters on folded light curves. *Abstracts.*
- **RAVEN** (Hadjigeorghiou2025): GBDT + GP classifiers trained on planets and 8 FP scenarios
  injected into real TESS light curves, combined into posterior probabilities with priors.
  Methodological precedent for our learned, injection-trained, calibrated combiner. *Abstract.*
- **TRICERATOPS** (Giacalone2021), **DAVE** (Kostov2019), centroid methods (Bryson2013),
  **Transit-APP** (Bryson2025; note: the brief attributed Transit-APP to Kunimoto; Crossref
  shows Bryson et al. 2025, RNAAS 9, 81). *Abstracts.*

## 3. Period ambiguity and aliases (core area)

- **Tschudi 2026a,b** (Tschudi2026a; Tschudi2026b): closest prior work. Rule-based harmonic
  correction for 2P, 3P and P/2 in TESS (equal-depth phase-0.5 events, odd/even > 3σ,
  equal-depth thirds, transit-count ratios, TLS re-runs at P/2-P/4), plus an event-time O−C
  coherence test, applied to 461 M-dwarf TOI hosts. *Full text of the relevant sections.*
- **Long-period aliases**: Cooke2019 and Cooke2021 (duotransit aliases, ~38 per system, follow-up
  strategy), Hawthorn2024 (85 duotransit candidates), MonoTools via Osborn2022 (Bayesian
  marginalisation over aliases using stellar density and window function), Garai2023 and
  Becker2019 (individual systems), Bass2025 (ruling out short aliases for > 100 d TOIs using TESS
  data). *Abstracts + grep.* All out of our scope (short-period, many transits) and cited as
  such.
- **TESS EB catalog** (Prsa2022): human classifiers could flag EBs "where the period was
  ambiguous (whether there are two nearly identical primary and secondary eclipses or no visible
  secondary eclipse)". Useful ground truth and a source of hard EB cases. *Full text (grep).*
- **RAVEN on TESS-SPOC FFIs** (Lafarga2026): aliases resolved by visual inspection; 80 of 465
  unrecovered TOIs sit at 2P, P/2, 3P or P/3 of the ExoFOP period. *Full text (grep).*
- **Single-transit period posteriors** (Javed2026): period from transit duration with a neural
  network marginalising over geometry; single transits only. *Abstract.*
- **Kepler period-accuracy catalog** (Lissauer2024) and **TESS Ten Thousand EB catalog**
  (Kostov2025, updated EB ephemerides). *Abstracts.*
- **Ephemeris maintenance** (Dragomir2020). *Abstract.*
- **Rotation-period alias classifier** (Boyle2026, TARS): a learned half-period alias
  corrector for rotation periods; different signal class, but a precedent for learned alias
  classification in TESS. *Abstract.*
- **Stellar density / photo-eccentric**: Seager2003 (unique solution for ρ⋆ from transit
  shape and P), Kipping2010 (duration expressions), Kipping2014 (asterodensity profiling),
  Dawson2012 (photo-eccentric effect). These give our density-consistency alias likelihood; the
  eccentricity degeneracy (Dawson2012) is the main reason this feature cannot be decisive alone.

## 4. Catalogs, data, modelling, statistics

TESS mission (Ricker2015), TOI catalog (Guerrero2021), TIC 8 (Stassun2019), Gaia DR3 (GaiaDR3),
TESS light-curve DOI (TESSdataDOI), batman (Kreidberg2015), limb darkening (Claret2017;
Kipping2013ld), Kepler injection-recovery (Christiansen2013), Astropy (Astropy2022). Statistics:
McNemar1947 (paired test), Wilson1927 (binomial CIs), Guo2017 (temperature scaling / ECE),
Zadrozny2002 (isotonic calibration), Chow1970 (reject option / abstain), Friedman2001 (gradient
boosting).

## 5. Not yet covered (to do before writing)

- ~~ADS full-text search~~: done 2026-09-30, see docs/novelty_check.md Section 5.
- SPOC TESS-specific TPS/DV documentation for multi-sector runs (the TESS DV release notes are
  not DOI-registered; decide how to cite).
- Tuson et al. CHEOPS duotransit programme (found only as an EPSC abstract so far).
- Kepler "Coughlin 2017" model-shift document (a KSCI technical document; cite via Thompson2018
  unless a DOI/bibcode is found).
