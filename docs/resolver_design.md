# Alias resolver: design notes (Phase 3)

Code: `src/exogargantua/resolver/` (`core.py` pipeline, `hypotheses.py`, `model.py`, `features.py`,
`combine.py`, `synth.py`). Tests: `tests/test_resolver.py`. Citation keys refer to `paper/refs.bib`;
every key resolves through `scripts/build_bib.py` / `scripts/verify_bib.py`.

**What is reused and what is new.**
- **Reused (cited):**
  - odd/even depth test (Twicken2018);
  - secondary-eclipse search;
  - stellar density from transit shape (Seager2003, Kipping2010);
  - timing precision (Carter2008);
  - red-noise factor (Pont2006);
  - wotan detrending (Hippke2019wotan);
  - gradient boosting (Pedregosa2011);
  - temperature scaling (Guo2017).
- **New:**
  - their combination into a posterior over the alias family, with an abstain outcome;
  - the benchmark that calibrates it;
  - the catalog audit.

## 1. Problem and output

**Input:**
- a light curve (normalised PDCSAP flux, NaN-free);
- a seed period P0 from any search (BLS, TLS, SPOC TPS, QLP, or a deliberately wrong seed in B1(i));
- optionally the seed epoch and duration, which are estimated from the data when absent;
- optionally the TIC stellar density (g cm⁻³) and its 1σ error.

**Output:**
- the probability of each alias factor r ∈ R = {1/5, 1/4, 1/3, 1/2, 2/3, 1, 3/2, 2, 3, 4, 5};
- the refined period of each alias;
- the MAP period and its probability;
- an abstain flag (maximum probability below a threshold; default 0.9, to be chosen on B1 training
  data only, A4);
- all per-hypothesis evidence.

**Hypotheses** (`hypotheses.py`).
- A hypothesis is (r, j): period r·P0 with transits at t0 + j·P0 + m·r·P0. For r = a/b,
  j = 0 … a−1 enumerates the distinct ephemerides that keep one of the seed's events. For example,
  for r = 2 the transits are either the even or the odd seed events. This gives 24 hypotheses.
- Each hypothesis is refined locally, because true periods are rarely exact rational multiples of
  a noisy P0 (this also fixes v1's non-harmonic mismatches):
  - a (period, epoch) grid of ±T·P/span in period (phase drift ≤ T over the baseline) and ±T/2 in
    epoch, maximising the folded matched filter;
  - then a least-squares trapezoid fit with free depth, duration T, ingress τ, P and t0.

**Seed ephemeris.** When only P0 is given, t0 and T come from a box matched filter at P0 over
phase and the durations {0.5 … 12} h (< P0/4). Every alias hypothesis shares the seed's reference
event, so a wrong seed still yields a correct candidate.

**Light-curve conditioning** (`model.py`).
- wotan biweight with window max(0.75 d, 3T) (Hippke2019wotan), so the transit is not absorbed.
- White noise: 1.4826 × MAD.
- Red-noise factor β by time-averaging on the transit timescale (Pont2006).
- The noise is estimated with the densest candidate grid (P0/5) masked.

**Model-error term.** Each event's depth error has ε·max(dₙ, 0) added in quadrature (ε = 0.05).
- **What it stands for:** event-to-event variability, detrending residuals, and the trapezoid not
  being a limb-darkened shape.
- **Why it is needed:** without it, likelihood differences at high SNR (10⁵ nats) are dominated by
  0.1 % fit differences between nested models. During development the equal-depth EB twin flipped
  between aliases with p = 1.000 either way.
- **Why it scales with each epoch's own depth:** the term is then the same for every hypothesis
  that predicts that epoch. A hypothesis-level depth would favour diluted short-period aliases.
  Flat epochs keep full precision, which is the missing-transit evidence.
- **Matching floors:** timing σ_t gets 0.01 T; duration and τ/T in F4 and F6 get 0.05 T.
- **Status:** these constants are fixed design choices made before any B1 or B2 result. They are
  not tuned on holdout data and will not be tuned on B1 test data.

## 2. The six features

Each feature returns a natural-log likelihood term for the principled combiner and raw descriptors
for the learned one. Every term is a log-ratio against a reference that is the same for all
hypotheses of a target, so the terms can be compared across hypotheses.

### F1 — Predicted-epoch coverage ("missing transit" test)
- **Method:** list every predicted epoch, and count it as observed when at least 50 % of its
  expected in-transit cadences exist. Fit each epoch's depth with the hypothesis' shape fixed
  (linear): dₙ ± sₙ, with s including β and the model-error term.
- **Term:** F1 = Σ over parity groups g of ½·max(S_g, 0)²/W_g, where S = Σ d/s² and W = Σ 1/s².
  This is the log-likelihood ratio of "transits at every predicted epoch" (depth ≥ 0, one depth per
  parity) against "no transits".
- **What it catches:**
  - A P/k alias predicts transits where the data are flat. Within each parity group, its flat and
    real epochs then disagree, and S_g/√W_g collapses.
  - A k·P alias leaves real events unexplained, so it explains less likelihood. Unexplained events
    can only be recovered through F3.
- **Descriptors:** number of predicted and covered epochs, coverage fraction, number of "missing"
  epochs (more than 3σ below the common depth), overall depth and SNR, per-parity depths.

### F2 — Odd/even depth consistency
- **Term:** F2 = l_common − l_parity ≤ 0, where l_common is the same likelihood ratio with one
  common depth.
- **Exact split:** F1 + F2 equals the common-depth likelihood ratio (tested in
  `test_f1_f2_split_is_exact_and_penalises_half_period`). −2·F2 is the odd/even χ² (1 dof) when the
  depths are unconstrained.
- **Significance:** reported as σ = √(−2·F2), the convention SPOC DV uses, flagged at 3σ
  (Twicken2018).
- **Role:** this is the main discriminator of P/2 against P, and of unequal-EB P against 2P.

### F3 — Secondary event at any phase
- **Search:** box search of the primary-subtracted, folded light curve over all phases outside the
  primary window, with durations {T/2, T, 2T} (< P/3). This covers eccentric EBs whose secondary is
  away from phase 0.5. The strongest event is refined with the same trapezoid model as the primary
  (free depth, duration, ingress, phase).
- **Evidence:** a Bayes factor from the secondary's per-epoch depths, on the same footing as F1:
  - ½·max(S,0)²/W, the profile over a common depth;
  - plus the depth Occam factor for a uniform depth prior on [0, d_max], with
    d_max = max(2 × primary depth, 10 σ_D);
  - plus log of the phase prior times the resolution element D/P. The phase prior is
    ½ uniform + ½ N(0.5, 0.01), the near-0.5 part being circular orbits.
- **Term:** F3 = log(½ + ½·BF), where ½ is the prior probability of no secondary.
- **What it means for each alias:**
  - For a k·P alias, F3 is how the unexplained events are recovered, as a "secondary" at phase 1/k.
    This works only for k = 2. For k ≥ 3 there are several unexplained phases and a single secondary
    cannot explain them.
  - For an equal-depth EB twin seen at P0, the 2P0 hypothesis explains the data with a phase-0.5
    secondary of equal depth and shape.
- **Descriptors (the brief's depth / duration / shape comparison):** secondary SNR, depth ratio,
  phase offset from 0.5, duration ratio, τ/T, and a "twin" flag (equal depth within 3σ, |phase −
  0.5| < 0.02, similar duration).

### F4 — Stellar-density consistency
- **Method:** a forward model, not a point estimate. Monte Carlo draws, fixed seed, shared by all
  hypotheses of a target:
  - stellar density from the TIC value (lognormal, error floored at 10 %);
  - impact parameter uniform on [0, 1 + k];
  - eccentricity from Beta(0.867, 3.03) (Kipping2013ecc), ω uniform.
  - Orbits whose periastron is inside the star are excluded.
- **Prediction:** for each draw, T14 and the ingress τ = (T14 − T23)/2 follow from a/R* (Kepler's
  third law), with the eccentric duration factor √(1−e²)/(1 + e sin ω) (Winn2010; Kipping2010).
- **Term:** F4 = log of the mean over draws of N(T_obs; T14, σ_T)·N(τ_obs; τ, σ_τ). That is
  p(T, τ | P, star): the density argument of Seager2003 turned into a likelihood per alias.
- **Behaviour:** short aliases with too long a duration are excluded. Long aliases are constrained
  through τ, which limits b.
- **No stellar parameters:** F4 = 0 for every hypothesis.
- **Descriptor:** log of the b = 0, circular implied density over the TIC density.

### F5 — Ephemeris coherence across sectors
- **Method:** for each event family (primaries, and secondaries when detected at SNR ≥ 7), take
  events with observed depth SNR ≥ 3 and fit each event's mid-time by matched filter over ±T/3 with
  parabolic refinement. Fit a weighted linear ephemeris to these times.
- **Term:** F5 = Σ [log N(res; 0, σ_t) − log N(res; 0, √(σ_t² + (T/4)²))], comparing a coherent
  linear ephemeris against an incoherent one. Timing errors follow Carter2008,
  σ_t = (T/Q)·√(τ/2T), floored as above.
- **Role:** penalises candidates whose events do not line up over the multi-sector baseline, for
  example wrong cycle counts across sector gaps or incoherent "events" made of noise.
- **Descriptors:** number of timed events, reduced O−C χ².

### F6 — Event-shape consistency between alternating events
- **Method:** separate trapezoid fits to odd and even events (depth, T, τ free; ephemeris fixed).
- **Term:** F6 = −½·χ² of the differences in duration and in τ/T, with floors. It is zero when
  either parity has no fit with depth SNR ≥ 3.
- **What it catches:** EBs whose alternating eclipses differ in shape (partial against total,
  eccentric durations) even when their depths match.
- **Descriptors:** odd/even duration ratio, shape χ².

## 3. Combiners (`combine.py`)

### (a) Principled
- **Posterior:** log p(h | data) = log prior(h) + Σ_k F_k(h). The prior is uniform over r, then
  uniform over the a offsets of r = a/b. p(r) = Σ_j p(r, j).
- **Independence assumptions:**
  1. F1 and F2 are an exact orthogonal split of the per-epoch depth likelihood, so they do not
     double count.
  2. F3 uses only the primary-subtracted residuals at phases outside the primary window, which are
     disjoint from F1/F2's data. It double counts only through the shared noise estimate.
  3. F4 uses the fitted (T, τ) and stellar parameters, not the per-epoch depths. It is treated as
     independent of F1–F3. Approximate: T and τ come from the same fit as the depth.
  4. F5 uses event times. For a symmetric template, the time derivative of the model is orthogonal
     to the model, so timing and depth estimates are uncorrelated to first order.
  5. F6 uses shape parameters per parity. Approximate: the trapezoid's depth and τ are correlated,
     so F6 and F2 are not exactly independent.
- **Known limitation:** for equal-depth EB twins the photometric likelihoods of P0 (planet) and
  2P0 (EB with an equal secondary) are nearly identical. The uniform depth prior for secondaries
  then favours P0 by a factor of order the secondary SNR, so the principled combiner tends to choose
  the half period. Information that separates twins is in the descriptors (twin flag, V-shape τ/T,
  depth, density mismatch); the learned combiner can use it, the principled one does not.

### (b) Learned
- **Model:** sklearn `HistGradientBoostingClassifier` (Pedregosa2011), fixed seed 20260930 and
  fixed hyper-parameters, on one row per (target, alias). The row is the alias' most probable
  offset.
- **Row features:**
  - each F_k and the principled log-posterior, relative to the target's maximum;
  - the raw descriptors;
  - log10 of the period.
- **Excluded on purpose:** the seed-relative factor r. Otherwise the model would learn its training
  set's seed distribution (uniform over r in B1(i), mostly r = 1 for real searches).
- **Label:** refined period within 0.1 % of the true period.
- **Calibration:** per-target softmax of the classifier logits divided by a temperature, chosen by
  minimum negative log-likelihood of the true alias on a held-out calibration split (Guo2017).
- **Training data:** only B1 injections (brief §5(b), A4). The Phase 3 synthetic check trains on
  synthetic data purely to exercise the code.

### Abstain
Both combiners abstain when the maximum alias probability is below the threshold. Accuracy against
coverage curves are reported across all thresholds.

## 4. Validation status (Phase 3)
- **Unit tests:** `tests/test_resolver.py` covers model pieces, every feature's sign and behaviour,
  the exact F1/F2 split, combiner normalisation and temperature fit, metrics, and end to end on
  synthetic planets from wrong seeds.
- **Synthetic check:** `scripts/resolver_synth_run.py` (Kaggle CPU) and
  `scripts/resolver_synth_report.py` → `results/resolver_synth_check.json`.
  - Data: fully synthetic TESS-like light curves, B1 class mix and parameter ranges, wrong seeds
    uniform over R, plus 10 % signal-free light curves.
  - It is a code check, not the benchmark: B1 injects into real pool-star light curves.
- **Smoke targets:** `scripts/resolver_smoke.py` is an execution and runtime test on the 20 Phase-2
  smoke targets, which are members of the sealed B2 holdout. It makes no truth comparison (A4).
