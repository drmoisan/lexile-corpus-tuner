# improper-match-short-circuit (Plan)

- Issue: #72
- Owner: improper-match-short-circuit
- Date: 2026-01-05
- Status: Completed

**Phase 0 — Context & Inputs**
- [x] [P0-T1] Link approved spec: [docs/features/active/2026-01-05-improper-match-short-circuit-72/spec.md](docs/features/active/2026-01-05-improper-match-short-circuit-72/spec.md)
- [x] [P0-T2] Record branch/commit baseline: feature/original-publication-date-71 @ 931d84fe3e6e47a562a52790d3ba632c9903ede0
- [x] [P0-T3] List required environment/fixtures/data: pytest command `poetry run pytest tests/src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/enrich_original_pub_year/test_match_utils.py`; inline `MatchCandidate` fixtures with duplicated Moby-Dick entries.

**Phase 1 — Preparation**
- [x] [P1-T1] Confirm scope is locked per spec (earliest-year selection fix only; no API/CLI changes).
- [x] [P1-T2] Sync workspace to feature/original-publication-date-71; ensure Poetry env available and deps installed.

**Phase 2 — Regression Test (must fail first)**
- [x] [P2-T1] Verify regression test exists at [tests/src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/enrich_original_pub_year/test_match_utils.py](tests/src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/enrich_original_pub_year/test_match_utils.py) and matches spec scenarios.
- [x] [P2-T2] Run `poetry run pytest tests/src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/enrich_original_pub_year/test_match_utils.py` to confirm failure reproduces order-bias (expects 1956, gets 2014).

**Phase 3 — Minimal Fix**
- [x] [P3-T1] Update [src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/enrich_original_pub_year/match_utils.py](src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/enrich_original_pub_year/match_utils.py) to select earliest qualifying year among exact matches while preserving fuzzy threshold handling; keep change minimal.

**Phase 4 — Verification Loop**
- [x] [P4-T1] Re-run targeted pytest: `poetry run pytest tests/src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/enrich_original_pub_year/test_match_utils.py`.
- [x] [P4-T2] Run formatter → linter → type checker → tests in order: `poetry run black .`, `poetry run ruff check`, `poetry run pyright`, `poetry run pytest` (full); restart loop if any step fails or edits files.

**Phase 5 — Documentation & Status**
- [x] [P5-T1] Update spec and [issue.md](docs/features/active/2026-01-05-improper-match-short-circuit-72/issue.md) with fix summary, validation results, and any scope deviations.

**Phase 6 — PR & Handoff**
- [x] [P6-T1] Prepare PR summary (problem, fix, risks, validation commands) and request review; include links to spec, issue #72, and key tests.

**Phase 7 — Rollout / Follow-up**
- [x] [P7-T1] Capture rollout/monitoring notes (none expected; confirm CI green).
- [x] [P7-T2] Record links for traceability: issue #72, PR, updated spec and plan.
