# Exo-Gargantua v2

Resolving orbital-period aliases in TESS transit signals: a calibrated resolver, a public
benchmark, and an audit of the TOI catalog.

**Status: under reconstruction (Phase 1 of 7).** There are no results in this branch yet, by
design. The v1 pipeline and paper (rejected, AAS80459) are kept read-only in [`legacy/`](legacy/);
no v1 number is reused. Why is documented in
[`docs/legacy_inventory.md`](docs/legacy_inventory.md).

## What this project is (and is not)

Given a periodic transit-like signal and its detected period P0 (from any search: BLS, TLS, SPOC
TPS, QLP), the resolver will return a probability for each member of the alias family
{P0 · r}, r ∈ {1/5, 1/4, 1/3, 1/2, 2/3, 1, 3/2, 2, 3, 4, 5}, including an explicit
"unresolvable" outcome, together with the evidence behind it.

It does not search for transits, does not vet planets versus false positives, does not correct
TTVs, and does not replace SPOC or QLP. It is meant to sit after them.

## Where things are

| Path | Contents |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | Hard rules for this repository (reproducibility, citations, claims) |
| [`docs/legacy_inventory.md`](docs/legacy_inventory.md) | Audit of v1 and root causes of the reviewer-flagged errors |
| [`docs/related_work.md`](docs/related_work.md), [`docs/related_work_matrix.csv`](docs/related_work_matrix.csv) | Literature review |
| [`docs/novelty_check.md`](docs/novelty_check.md) | What is and is not new, with quoted evidence |
| [`docs/success_criteria.md`](docs/success_criteria.md) | Benchmark pass/fail criteria, fixed before any run |
| `paper/bib_sources.csv` → `paper/refs.bib` | Bibliography, generated from DOIs/arXiv IDs only |

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
make test          # unit tests + bibliography verification (needs internet)
make legacy-audit  # re-runs the v1 validator on synthetic transits (Phase 0 evidence)
```

`make paper` will regenerate every number, table and figure once Phases 2-7 exist. Heavy runs
execute on Kaggle (see `CLAUDE.md`).

## License

MIT (see `LICENSE`).
