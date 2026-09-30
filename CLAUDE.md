# Exo-Gargantua v2: working rules

Working paper: "Resolving Orbital Period Aliases in TESS Transit Signals: A Calibrated
Resolver, a Public Benchmark, and an Audit of the TOI Catalog".

The previous version (AAS80459) was rejected. Everything it produced lives in `legacy/`
and is **not trusted**: no legacy number, table, figure or claim is reused. See
`docs/legacy_inventory.md` for why.

## Hard rules (verbatim from the project brief)

1. **No number exists unless a committed script printed it from a committed results file.** Every number in the manuscript comes from `results/*.json`, written by a script, and is injected into LaTeX through an auto-generated `paper/generated_macros.tex` (`\newcommand{\aliasAccHoldout}{87.3}` etc). Never type a result number into the .tex by hand.
2. **`make paper` regenerates everything**: data pulls (cached), runs, tables, figures, macros, PDF. A test (`tests/test_paper_consistency.py`) parses the compiled table data and asserts every summary number is recomputable from the per-target table. CI fails if not.
3. **Sample selection is frozen before results are seen.** Every inclusion/exclusion goes into `data/sample_log.csv` with target, reason, timestamp, and git commit, written by the selection script. No manual adding or dropping of targets after running. Ever.
4. **No invented citations.** Every BibTeX entry must be pulled from NASA ADS (API) or arXiv with a DOI or bibcode, and `scripts/verify_bib.py` checks each resolves. If you cannot verify a paper exists, do not cite it.
5. **Every figure asserts its own labels.** If a legend says b = 0.93, the plotting script reads b from the fit result and asserts it matches. No hardcoded annotations.
6. **Every target disposition is checked against ExoFOP TFOPWG dispositions and the NASA Exoplanet Archive** before it appears in any table. A known/confirmed planet may never appear as "false positive" without an explicit flagged discussion.
7. **No "state of the art" or "outperforms" claims** unless the benchmark shows it with non-overlapping 95% confidence intervals or a significant paired test (McNemar). Otherwise say "comparable".
8. **Seeds, configs, and git commit hash are stored in every results file.**
9. **Stop and report** (do not improvise) if a phase gate fails.
10. **No heavy compute on the local MacBook Air.** Anything beyond quick tests on a handful of targets runs on Kaggle (see Section 1b). Locally: writing code, unit tests on tiny fixtures, small debugging runs (under ~5 minutes), LaTeX builds, and analysis of results files pulled back from Kaggle.

## Compute policy (brief Section 1b, summarised)

- Heavy = bulk light-curve downloads, full BLS/TLS searches, injection grid (B1), temporal
  holdout (B2), TOI audit (B3), runtime test (B4), scrambled/inverted false-alarm runs,
  training/calibrating the learned combiner. These run on Kaggle only.
- Each heavy job lives in `kaggle/jobs/<job_name>/` with `kernel-metadata.json`, installs this
  package from GitHub at a pinned commit, and writes outputs pulled back into
  `results/kaggle/<job_name>/<run_id>/`.
- Kaggle credentials stay in `~/.kaggle/`; never commit them.
- CPU sessions for downloads/TLS/per-target fitting; GPU only where profiling shows it helps.
- Smoke run on 20 targets first, report runtime and a full-run cost estimate, then wait for
  approval before the full run. Jobs checkpoint in resumable chunks.

## Things the paper must never say

- That the resolver corrects TTVs, does anything "Fourier", or replaces SPOC/QLP.
- "Perfectly", "definitively", "pristine", "brute force", "democratizes", "black-box",
  "heavy institutional".
- The thermal-hook / "convexity bias" claim, unless benchmark B5 establishes it statistically.
- Any "Rejected (FP)" vetting verdict: this paper is about periods, not vetting.

## Layout

```
src/exogargantua/   installable package (resolver, features, data access)
scripts/            data pulls, runs, figure/table/macro generation, verify_bib.py
kaggle/jobs/        heavy jobs (one directory per job)
data/raw/           dated catalog snapshots          data/cache/  downloaded LCs (gitignored)
data/sample_log.csv frozen sample selection log
results/            results JSON written by scripts (with seed, config, git commit)
paper/              AASTeX 7 source, refs.bib (auto-built), generated_macros.tex
tests/              pytest, incl. paper consistency and bib verification
docs/               inventory, related work, novelty check, success criteria
legacy/             the rejected v1 project, read only
```
