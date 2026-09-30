# `make paper` must regenerate everything from scratch (CLAUDE.md rule 2).
# Targets are filled in phase by phase; unimplemented steps fail loudly.
PY ?= python

.PHONY: bib verify-bib test legacy-audit paper

bib:            ## rebuild paper/refs.bib from paper/bib_sources.csv (network)
	$(PY) scripts/build_bib.py

verify-bib:     ## check every citation resolves (network)
	$(PY) scripts/verify_bib.py

test:
	$(PY) -m pytest -q

legacy-audit:   ## Phase 0 root-cause demonstration on synthetic data
	$(PY) scripts/legacy_audit/demo_validator_regression.py

paper:
	@echo "make paper: not implemented yet (Phases 2-7 pending). See docs/." && exit 1
