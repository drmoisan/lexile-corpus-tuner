# 2026-01-20-ck12-missing-enrichment-links (Plan)

- **Issue:** #95
- **Parent (optional):** none
- **Owner:** drmoisan
- **Last Updated:** 2026-01-20T16-24
- **Status:** Planned
- **Version:** 0.1

Status Badge: [Planned | blue]

### Phase 0 — Context & Inputs
- [x] [P0-T1] Read `.github/copilot-instructions.md`
	- Acceptance: task marked complete after confirming no conflicting policy with this plan.
- [x] [P0-T2] Read `.github/instructions/general-code-change.instructions.md`
	- Acceptance: task marked complete after confirming toolchain order and bugfix workflow requirements.
- [x] [P0-T3] Read `.github/instructions/general-unit-test.instructions.md`
	- Acceptance: task marked complete after confirming no-temp-file rule and coverage expectations.
- [x] [P0-T4] Read `.github/instructions/python-code-change.instructions.md` and `.github/instructions/python-unit-test.instructions.md`
	- Acceptance: task marked complete after confirming Python test runner and typing requirements.
- [x] [P0-T5] Read `docs/features/active/2026-01-20-ck12-missing-enrichment-links-95/spec.md`
	- Acceptance: task marked complete after confirming acceptance criteria and logging requirement are understood.
- [x] [P0-T6] Read `docs/features/active/2026-01-20-ck12-missing-enrichment-links-95/20260120-ck12-missing-handle-research.md`
	- Acceptance: task marked complete after confirming required prefixes are `/tebook/`, `/workbook/`, `/quizbook/`.
- [x] [P0-T7] Capture baseline lint with `poetry run ruff check`
	- Acceptance: command exits 0 and output is recorded in execution log.
- [x] [P0-T8] Capture baseline type-check with `poetry run pyright`
	- Acceptance: command exits 0 and output is recorded in execution log.
- [x] [P0-T9] Capture baseline tests with `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`
	- Acceptance: command exits 0 and output is recorded in execution log.
	- Note: Baseline shows 3 pre-existing failures in test_atomic_executor_cli.py (unrelated to issue #95), 1282 passing, 88% coverage.
- [x] [P0-T10] Record branch and commit baseline using `git rev-parse --abbrev-ref HEAD` and `git rev-parse --short HEAD`
	- Acceptance: both command outputs are recorded in execution log.
- [x] [P0-T11] Record required data source and commands
	- Acceptance: execution log lists `https://static.ck12.org/testimonial/fbbrowse-prod.json` and the two CLI commands from spec Context.

### Phase 1 — Regression Tests (TDD)
- [ ] [P1-T1] Add pytest `test_extract_slug_from_content_url_supports_tebook` in `tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py`
	- Acceptance: test asserts slug equals `CK-12-Earth-Science-For-Middle-School-Teachers-Edition` for a `/tebook/` URL and fails before code change.
- [ ] [P1-T2] Add pytest `test_extract_slug_from_content_url_supports_workbook` in `tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py`
	- Acceptance: test asserts slug equals `CK-12-Earth-Science-For-Middle-School-Workbook` for a `/workbook/` URL and fails before code change.
- [ ] [P1-T3] Add pytest `test_extract_slug_from_content_url_supports_quizbook` in `tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py`
	- Acceptance: test asserts slug equals `CK-12-Earth-Science-For-Middle-School-Quizzes-and-Tests` for a `/quizbook/` URL and fails before code change.
- [ ] [P1-T4] Add pytest `test_parse_catalog_json_logs_warning_when_title_and_content_url_missing_slug` in `tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py`
	- Acceptance: test uses `caplog` at WARNING level, asserts warning contains `Content_URL slug missing`, the `Title`, and the `Content_URL`; fails before code change.

### Phase 2 — Minimal Fix
- [ ] [P2-T1] Update `extract_slug_from_content_url` in `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_catalog.py` (lines 64-103)
	- Implementation detail: for `parsed.netloc == "www.ck12.org"`, accept prefixes `book`, `cbook`, `tebook`, `workbook`, `quizbook` by locating the first matching segment in `path_parts` and returning the following segment when present.
	- Acceptance: tests [P1-T1]–[P1-T3] pass and existing `/book/` + `/cbook/` behavior remains unchanged.
- [ ] [P2-T2] Add warning logging in `parse_catalog_json` when `Title` is present, `Content_URL` is present, and `content_url_slug` is `None` (lines 154-360)
	- Implementation detail: add `import logging` near the import block (lines 30-36), define `logger = logging.getLogger(__name__)` after imports (lines 37-40), and emit `logger.warning("Content_URL slug missing for title '%s' url '%s'", title, content_url_raw)` immediately after the `content_url_slug` assignment block (lines 269-279).
	- Acceptance: test [P1-T4] passes with the exact message prefix `Content_URL slug missing` and includes title + URL.

### Phase 3 — QA Toolchain Loop
- [ ] [P3-T1] Run `poetry run black .`
	- Acceptance: command exits 0 and produces no file changes.
- [ ] [P3-T2] Run `poetry run ruff check`
	- Acceptance: command exits 0; if it changes files or fails, restart Phase 3 from [P3-T1].
- [ ] [P3-T3] Run `poetry run pyright`
	- Acceptance: command exits 0; if it fails, fix issues and restart Phase 3 from [P3-T1].
- [ ] [P3-T4] Run `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`
	- Acceptance: command exits 0; if it fails, fix issues and restart Phase 3 from [P3-T1].
