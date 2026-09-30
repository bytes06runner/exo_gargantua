# Success criteria (committed before any benchmark run)

Written down on 2026-09-30, before any Phase 4 result exists. These are not changed after
results are seen. If a criterion is missed, the paper reports it.

- **B2 (temporal holdout):** our alias accuracy beats raw BLS, TLS, and single-sector SPOC TCE
  periods with significant McNemar p < 0.01.
- **B1 (injection-recovery):** ECE < 0.05; accuracy at 90% coverage reported.
- **False alarms:** confident resolution on scrambled data < 5%.

Definitions used when these are evaluated (fixed now, per the brief):
- Alias accuracy: resolved period within 0.1% of the true period after refinement. P/2, 2P and
  other aliases count as **wrong**.
- 95% confidence intervals: Wilson (proportions) or bootstrap.
- "Confident" = the resolver's maximum alias probability ≥ its abstain threshold.

Added at Gate 1 (pending owner approval, see docs/novelty_check.md): the rule-based harmonic
correction of Tschudi (2026a) is included as an additional B2 baseline. The B2 criterion above
is unchanged; the comparison with Tschudi (2026a) is reported whatever its outcome.
