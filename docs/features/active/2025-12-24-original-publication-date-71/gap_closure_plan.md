# 2025-12-24-original-publication-date - Gap Closure Plan

- Issue: #71
- Owner: drmoisan
- Last Updated: 2026-01-06

## Required References

- General Coding Standards: [`.github/instructions/general-code-change.instructions.md`](../../../../.github/instructions/general-code-change.instructions.md)
- General Unit Test Policy: [`.github/instructions/general-unit-test.instructions.md`](../../../../.github/instructions/general-unit-test.instructions.md)
- Python Coding Policy: [`.github/instructions/python-code-change.instructions.md`](../../../../.github/instructions/python-code-change.instructions.md)
- Python Unit Test Policy: [`.github/instructions/python-unit-test.instructions.md`](../../../../.github/instructions/python-unit-test.instructions.md)

## Overview

Scope is locked to Open Library only; Wikidata/LOC fallback is out of scope and tracked in the potential feature. `--batch-size` should be removed unless a real multi-row request strategy is introduced.

## Phased Atomic Plan

**Phase 0 — Context & Inputs**
 - [x] [P0-T1] Read .github/copilot-instructions.md and .github/instructions/general-code-change.instructions.md to confirm baseline rules
 - [x] [P0-T2] Read .github/instructions/general-unit-test.instructions.md and .github/instructions/python-unit-test.instructions.md to confirm testing standards
 - [x] [P0-T3] Read .github/instructions/python-code-change.instructions.md for Python-specific coding rules
 - [x] [P0-T4] Re-read feature docs (spec.md, user-story.md) and potential fallback doc to confirm scope (Open Library only, no fallback, remove batch-size)

**Phase 1 — Remove fallback knobs and wiring**
 - [x] [P1-T1] Remove CLI flags --enable-wikidata and --enable-loc from the enrichment CLI module (help text, argument parsing, defaults) so they are no longer exposed
 - [x] [P1-T2] Remove config fields enable_wikidata and enable_loc from the enrichment config dataclass and any schema/loader so configs cannot carry fallback toggles
 - [x] [P1-T3] Remove fallback protocol/noop fallback/LOC client modules if no longer referenced; ensure original_pub_source still sets openlibrary on successful matches
 - [x] [P1-T4] Remove fallback toggle usage from the pipeline orchestration code so only Open Library execution remains

**Phase 2 — Remove batch-size option**
 - [x] [P2-T1] Remove --batch-size from CLI flags, help text, and config dataclass so it is no longer user-facing
 - [x] [P2-T2] Delete internal batch-size usage for checkpoint cadence or loop control so processing is per-row without dead code
 - [x] [P2-T3] Verify codebase has no remaining references to batch_size (search/clean) and that checkpoint cadence still functions

**Phase 3 — Align default paths**
 - [x] [P3-T1] Update default output path constant to data/meta/gutenberg/gutenberg_books_enhanced.parquet and confirm CLI defaults match
 - [x] [P3-T2] Update default checkpoint path constant to data/meta/gutenberg/.original_pub_year.ckpt and confirm CLI defaults match
 - [x] [P3-T3] Verify docstrings/help text reflect the updated defaults and no other default path conflicts exist

**Phase 4 — Tests (scenario-specific)**
 - [x] [P4-T1] Update unit test for enrichment config parsing to drop fallback toggle expectations and assert Open Library-only config in tests/enrich_original_pub_year/test_enrich_original_pub_year.py
 - [x] [P4-T2] Update unit test for original_pub_source assignment when Open Library returns a match (no fallback path) in tests/enrich_original_pub_year/test_enrich_original_pub_year.py
 - [x] [P4-T3] Update integration-style test (mocked pd.read_parquet/DataFrame.to_parquet/Open Library client) to expect no fallback and to assert summary JSON is printed to stdout in tests/enrich_original_pub_year/test_enrich_original_pub_year.py
 - [x] [P4-T4] Update checkpoint resume test using the in-memory checkpoint store to confirm resume logic without batch-size or fallback flags in tests/enrich_original_pub_year/test_enrich_original_pub_year.py

**Phase 5 — Docs alignment**
 - [x] [P5-T1] Verify spec.md and user-story.md contain no fallback or batch-size references and reflect updated defaults
 - [x] [P5-T2] Verify docs/source-curation-guide.md command examples and “flags to adjust” match the Open Library-only scope and updated defaults
 - [x] [P5-T3] Verify potential feature doc docs/features/potential/2026-01-06-enhance-gutenberg-wiki-loc-fallback.md is the only place documenting fallback behavior

**Phase 6 — Toolchain loop**
 - [x] [P6-T1] Run `poetry run black .`
 - [x] [P6-T2] Run `poetry run ruff check`
 - [x] [P6-T3] Run `poetry run pyright`
 - [x] [P6-T4] Run `poetry run pytest`
