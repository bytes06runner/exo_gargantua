# Novelty check (Gate 1)

**Question (from the brief):** Has anyone already built a calibrated, general alias resolver for
short-period TESS TCEs/TOIs with a public benchmark and catalog audit?

**Answer: No, not as a whole. But the pieces are closer to done than the brief assumed.** The
closest prior work (Tschudi 2026a,b) already corrects 2P/3P/P/2 aliases automatically in TESS
data, with deterministic rules, and applies its pipeline to all 461 M-dwarf TOI hosts.
The part that remains new is narrower: a *calibrated posterior over the alias family with an
abstain option*, a *stellar-density alias likelihood*, a *public alias-resolution benchmark*
with a confusion matrix by alias factor, and an *all-TOI period audit* (not restricted to M
dwarfs). The Gate 1 verdict is therefore **proceed, with repositioning** (Section 4), not pivot.
Approved by the project owner on 2026-09-30 (docs/decisions.md).

Search method: arXiv API title/abstract queries (logged in `data/cache/lit/arxiv_q*.txt`), web
search, Crossref, and full-text grep of the PDFs of the closest papers (cached in
`data/cache/lit/pdf/`, gitignored). No ADS token was available, so ADS full-text search was not
run. **That is a gap:** an ADS full-text query for "period alias" AND ("TCE" OR "TOI") should
be run before submission.

---

## 1. Closest prior work, with evidence

### Tschudi 2026a (arXiv:2603.10247, submitted to A&A): the closest match

A TLS search on 121 M3-M6 dwarfs with a "harmonic correction" step (their Sect. 3.2.1):

> "Harmonic aliasing occurs when TLS detects 2P or 3P instead of the true period, P. Three
> evidence modes address this event, each targeting a distinct aliasing signature"

- Mode A: per-epoch matched filter at phase 0.5, depth ratio > 0.4 → 2P alias.
- Mode B: odd/even > 3σ → P/2 alias of an EB.
- Mode C: equal-depth events at phases 1/3, 2/3 → 3P alias.
- Transit-count correlation between candidate periods; TLS re-run at P/2, P/3, P/4 with
  SDE-ratio thresholds (0.3 or 0.8 × SDE_orig).
- Code public on GitLab (`gitlab.com/yohanntschudi/`).

What it does **not** do: output probabilities; abstain; use stellar density to rank aliases;
handle P/k locks from missing-data coverage in a likelihood; report a benchmark of alias
resolution (validation is 16/16 known planets recovered, not an alias confusion matrix).

### Tschudi 2026b (arXiv:2607.23781): same pipeline applied to TOIs

Applied to all 461 active ExoFOP M-dwarf TOI hosts. Adds an "event-time-coherence test that
rejects signals with incoherent per-transit timing (window-function aliases of stellar
variability)" via a p-value against a no-clock null. Reports 221 recoveries "at the archive
period and four at a half-period alias". This is a partial TOI period audit, restricted to
M dwarfs, and its coherence test overlaps our planned feature 5 (ephemeris coherence).

### Kepler Robovetter DR24/DR25 (Coughlin et al. 2016; Thompson et al. 2018)

Has a `PERIOD_ALIAS` flag from model-shift phases. From Thompson et al. (2018), App. B:

> "a possible period alias is seen at a ratio of N:1, where N is an integer of 3 or greater ...
> This flag is currently informational only and not used to declare any TCE"

plus a `PLANET_PERIOD_IS_HALF` flag ("Planet scenario possible at half the DV period"). DR25
table: PERIOD_ALIAS fired on 5 (ALT) and 2 (DV) TCEs. Deterministic, informational, Kepler only.

### LEO-Vetter (Kunimoto et al. 2025, AJ 170, 280)

Odd/even differences are used to **fail** TCEs, i.e. EBs "identified at half the correct the
period" are rejected, not corrected. Also a Data Gap Test (fail if ≥ 50 % of transits are within
2 durations of a gap; targets 13.7 d scattered light) and an SNR-consistency (CHI) test on
per-transit SNRs. No alias correction or probability over periods.

### ExoMiner++ (Valizadegan et al. 2025, AJ 170, 287)

Explicitly names wrong TCE periods as a problem, not a solved one:

> "incorrect periods, such as those being twice or half the true period of a planet, can also
> lead to misclassification, thereby lowering recall values."

and gives examples (TIC 82308728: period twice TOI 1821.01 "because two out of four transits are
missing"; TOI 4635.01: 49.01 d vs correct 12 d). This is direct motivation for our work.

### Long-period alias work (Cooke et al. 2019, 2021; Hawthorn et al. 2024; MonoTools via Osborn et al. 2022; Bass et al. 2025)

Probabilistic alias ranking exists **for duotransits** (two transits years apart, ~38 aliases
each per Cooke et al. 2021), using stellar density and window functions (MonoTools). Bass et al.
(2025) rule out shorter aliases for > 100 d TOIs. Out of scope for us; cited as the long-period
counterpart.

### Methodological precedents we must credit

- RAVEN (Hadjigeorghiou et al. 2025): gradient-boosted trees + GP classifier trained on injections
  into real TESS light curves, output combined into posterior probabilities. Our learned
  combiner (brief 3b) follows the same recipe for a different target (alias class, not planet/FP).
- TARS (Boyle et al. 2026): a learned half-period alias classifier, for *stellar rotation*.
- TLS (Hippke & Heller 2019) already reports `empty_transit_count` / `distinct_transit_count`,
  the raw ingredient of our "missing transit" feature.
- Stellar density vs duration: Seager & Mallén-Ornelas 2003; Kipping 2010, 2014; Dawson &
  Johnson 2012.

## 2. Gap table

| Capability | Exists? | Where |
|---|---|---|
| Detect that period is 2P/3P/P/2 with rules | **Yes** | Tschudi 2026a; Robovetter flags; LEO-Vetter (reject only) |
| Correct the period automatically (short period, TESS) | **Yes, deterministic** | Tschudi 2026a,b |
| Per-transit timing coherence against aliases | **Yes** | Tschudi 2026b |
| Calibrated probability over the alias family {P0·r} | **Not found** | — |
| Abstain / "unresolvable" outcome with accuracy-coverage curve | **Not found** | — |
| Stellar-density likelihood per alias (short period) | **Not found** (exists for duotransits: MonoTools) | — |
| Public alias-resolution benchmark (injections + temporal holdout, confusion by alias factor, ECE) | **Not found** | — |
| TOI period audit across all TOIs with ≥ 2 sectors | **Partial** (M dwarfs only, Tschudi 2026b; long-period only, Bass 2025) | — |
| Robustness to inverted/scrambled light curves for an alias method | **Not found** (exists for vetters: Robovetter, LEO-Vetter) | — |

## 3. Risks to the novelty claim

1. Tschudi 2026a,b could be read as "alias resolution for TESS is done". The paper must engage
   with it directly and, ideally, **run it as a baseline in B2** (its code is public).
2. A reviewer may argue calibration is incremental. The answer has to be empirical: B1/B2 must
   show that probabilities + abstain give a better accuracy-coverage trade-off than deterministic
   rules on the same targets. If they do not, the contribution shrinks to the benchmark + audit.
3. ADS full-text search not yet done (Section 0).

## 4. Repositioning (approved 2026-09-30)

- Title can stay. Framing: "calibrated and benchmarked". The words "first" / "the first"
  are not used anywhere in the paper, including for the benchmark.
- Add Tschudi 2026a (rule-based harmonic correction) as an explicit baseline in B2, next to raw
  BLS/TLS peaks, SPOC TCE and QLP periods.
- The benchmark and the audit are the most defensible contributions; the resolver is the tool
  that makes them possible.
