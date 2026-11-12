# `make paper` must regenerate everything from scratch (CLAUDE.md rule 2).
# Targets are filled in phase by phase; unimplemented steps fail loudly.
PY ?= python

.PHONY: bib verify-bib test ads-search legacy-audit analysis paper

bib:            ## rebuild paper/refs.bib from paper/bib_sources.csv (network)
	$(PY) scripts/build_bib.py

verify-bib:     ## check every citation resolves (network)
	$(PY) scripts/verify_bib.py

test:
	$(PY) -m pytest -q

ads-search:     ## Gate 1 ADS full-text novelty search (needs ADS_API_TOKEN)
	$(PY) scripts/ads_novelty_search.py

legacy-audit:   ## Phase 0 root-cause demonstration on synthetic data
	$(PY) scripts/legacy_audit/demo_validator_regression.py

analysis:       ## rescore every benchmark and post hoc analysis from the Kaggle outputs in results/kaggle/ (local, minutes)
	$(PY) scripts/b5_score.py
	$(PY) scripts/independent_period_check.py
	$(PY) scripts/b5_spoc_score.py
	$(PY) scripts/b5_audit_report.py
	$(PY) scripts/b5_audit_compare.py
	$(PY) scripts/b5_sample_accounting.py
	$(PY) scripts/b5_tls_floor.py
	$(PY) scripts/b5_gaia_check.py
	$(PY) scripts/b5_alias_rates.py
	$(PY) scripts/b5_error_analysis.py
	$(PY) scripts/b5_tschudi_score.py
	$(PY) scripts/fa_context_score.py
	$(PY) scripts/paper_disclosures.py
	$(PY) scripts/toi_period_revisions.py
	$(PY) scripts/b2_secondary_check.py
	$(PY) scripts/tolerance_sensitivity.py
	$(PY) scripts/feature_importance.py
	$(PY) scripts/audit_recompute.py

paper: analysis ## regenerate macros, tables, figures and released tables from results/; the PDF is built by .github/workflows/paper.yml (manual trigger)
	$(PY) scripts/clean_bib.py
	$(PY) scripts/make_paper_assets.py
	$(PY) scripts/make_lc_figure.py
	$(PY) scripts/make_reliability_figure.py
	$(PY) scripts/make_mast_doi_lists.py
	$(PY) scripts/make_paper_assets.py
	$(PY) -m pytest -q tests/test_paper_consistency.py
	cd paper && pdflatex -interaction=nonstopmode ms && bibtex ms && pdflatex -interaction=nonstopmode ms && pdflatex -interaction=nonstopmode ms
