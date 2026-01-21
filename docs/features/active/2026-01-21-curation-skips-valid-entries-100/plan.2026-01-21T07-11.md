---
id: 2026-01-21-curation-skips-valid-entries-100
status: Planned
status_color: blue
owner: drmoisan
last_updated: 2026-01-21
---

# 2026-01-21-curation-skips-valid-entries (Plan)

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

- **Issue:** #100
- **Spec (authoritative):** `docs/features/active/2026-01-21-curation-skips-valid-entries-100/spec.md`
- **Research (authoritative):** `artifacts/research/20260121-ck12-curation-skips-research.md`
- **Owner:** drmoisan
- **Plan file:** `docs/features/active/2026-01-21-curation-skips-valid-entries-100/plan.2026-01-21T07-11.md`

## Requirements Traceability (REQ-*)

| REQ ID | Source | Requirement (machine-verifiable) |
| --- | --- | --- |
| REQ-001 | spec.md Acceptance Criteria | `extract_slug_from_content_url` returns `(slug, artifact_type)` with artifact_type in `{cbook, tebook, workbook, quizbook, book}` for supported hosts. |
| REQ-002 | spec.md Acceptance Criteria | Unknown URL prefixes never default to `book`, emit warnings, and leave `artifact_type` unset. |
| REQ-003 | spec.md Test Strategy | Unit tests in `tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py` cover all supported prefixes plus unknown-prefix scenario. |
| REQ-004 | spec.md Acceptance Criteria | CK-12 pipeline (catalog → enrichment → curation `--require-json`) keeps `/cbook/`, `/tebook/`, `/workbook/`, `/quizbook/` entries with revision JSON candidates. |
| REQ-005 | spec.md Acceptance Criteria | Final QC loop (Black → Ruff → Pyright → Pytest + coverage) passes in one uninterrupted run. |
| REQ-006 | spec.md Acceptance Criteria | Documentation reflects artifact-type mapping behavior; no CLI contracts change. |

## Implementation plan (atomic tasks)

### Phase 0 — Context & Inputs

- [x] [P0-T1] Read `.github/copilot-instructions.md` and confirm policy precedence (Copilot instructions → general policies → language-specific policies → unit-test addenda)
	- Acceptance: Add a one-line note under this task summarizing the precedence order.
	- **Completed**: Policy precedence confirmed per line 18: General instructions first (general-code-change.instructions.md), then language-specific (python-code-change.instructions.md), then unit-test addenda (general-unit-test.instructions.md, python-unit-test.instructions.md), with developer-tooling.md and CI docs as operational guidance underneath.

- [x] [P0-T2] Read `.github/instructions/general-code-change.instructions.md` to reaffirm required QC loop (Black → Ruff → Pyright → Pytest)
	- Acceptance: Note added citing the exact loop order.
	- **Completed**: QC loop confirmed per lines 225-262: (1) Formatting (Black), (2) Linting (Ruff), (3) Type checking (Pyright), (4) Testing (Pytest). If any step fails or auto-fixes, restart from step 1. Only when all four steps complete without errors in a single pass is the change complete.

- [x] [P0-T3] Read `.github/instructions/general-unit-test.instructions.md`, `.github/instructions/python-code-change.instructions.md`, and `.github/instructions/python-unit-test.instructions.md`
	- Acceptance: Note confirms tests must be deterministic, isolated, Pytest-based, and use `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing` for coverage.
	- **Completed**: All three policy files read and confirmed:
		- **General unit test policy**: Tests must be independent, isolated, fast, deterministic, with >=80% repo coverage and >=90% for new code. No external dependencies, no temp files on filesystem (line 82-84).
		- **Python code change policy**: Black formatting, Ruff linting, Pyright type-checking required. Full type annotations mandatory. Suppressions require pre-authorization or explicit approval (lines 25-70).
		- **Python unit test policy**: Pytest framework mandatory. Coverage command: `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing` (line 62).

- [x] [P0-T4] Review `spec.md` and `artifacts/research/20260121-ck12-curation-skips-research.md`; map each acceptance criterion to REQ-001..REQ-006
	- Acceptance: Note lists each REQ with pointers to spec/research sections proving coverage.
	- **Completed**: All 6 acceptance criteria from spec.md (lines 181-194) mapped to REQ-001..REQ-006. Research file does not exist (evidence already in spec.md lines 66-71). Mapping documented with spec line references and verification methods.

- [x] [P0-T5] Capture baseline QC status before changes
	- Commands: `poetry run black .`, `poetry run ruff check`, `poetry run pyright`, `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`
	- Acceptance: PASS/FAIL for each command recorded for later comparison.
	- **Completed**: Baseline QC status captured successfully:
		- **Black**: PASS - 206 files would be left unchanged
		- **Ruff**: PASS - All checks passed
		- **Pyright**: PASS - 0 errors, 0 warnings, 0 informations
		- **Pytest**: PASS - 1302 passed, 11 warnings in 6.57s
		- **Coverage**: 88% overall (9260 lines, 1153 missed)

### Phase 1 — Regression tests for slug/type extraction (TDD Red)

- [x] [P1-T1] Add parameterized test `test_extract_slug_and_type_from_content_url` in `tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py`
	- Cases: `(url, expected_slug, expected_type)` for `/cbook/`, `/tebook/`, `/workbook/`, `/quizbook/`, `/book/` URLs.
	- Acceptance: Running `poetry run pytest tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py -k extract_slug_and_type_from_content_url` fails because implementation still returns `str | None`.
	- **Completed**: Added parameterized test with 5 cases covering all supported URL prefixes. Test properly fails with "Expected tuple, got <class 'str'>". Toolchain (Black, Ruff, Pyright) passes. Test suite shows 5 failed (new tests), 18 passed (existing tests). TDD Red phase confirmed.

- [x] [P1-T2] Add test `test_extract_slug_and_type_unknown_prefix_returns_none` in the same file verifying unknown prefixes return `None` and emit warnings via `caplog`
	- Acceptance: Test fails before implementation updates because warning/None behavior is absent.
	- **Completed**: Added test with unknown URL prefix pattern. Test properly fails with two assertions: (1) result is None (currently returns string), (2) warning is logged (currently no warning). Toolchain (Black, Ruff, Pyright) passes. TDD Red phase confirmed for both P1-T1 and P1-T2 - 6 tests failing as expected (5 from P1-T1, 1 from P1-T2), 18 existing tests still passing.

### Phase 2 — Implementation: CK-12 catalog parsing

- [x] [P2-T1] Update `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_catalog.py::extract_slug_from_content_url`
	- Actions:
		1. Change signature to `tuple[str, str] | None`.
		2. Parse path segments using `urlparse`; detect prefixes in order: `cbook` (flexbooks host), `book`, `tebook`, `workbook`, `quizbook` (www host).
		3. Return `(segment_after_prefix, prefix)` when matched; otherwise return `None`.
	- Acceptance: `poetry run pyright` recognizes signature change without errors.
	- **Completed**: Function signature changed from `str | None` to `tuple[str, str] | None`. Returns `(slug, artifact_type)` where artifact_type is one of {cbook, book, tebook, workbook, quizbook}. Flexbooks.ck12.org URLs with /cbook/ return ("slug", "cbook"). WWW.ck12.org URLs return ("slug", prefix) for detected prefixes. Unknown patterns return None. Updated parse_catalog_json to unpack tuples and use perma_type when available, falling back to host-based defaults. Updated existing tests to match new behavior. Full toolchain passes: Black ✓, Ruff ✓, Pyright ✓, Pytest 1308 passed ✓, Coverage 88% ✓.

- [x] [P2-T2] Modify `parse_catalog_json` to consume `(slug, artifact_type)`
	- Steps:
		1. Unpack tuple results; set `content_url_slug` and `perma_type` accordingly.
		2. Use `perma_type` for `artifact_type_value` when available; only fall back to previous host-based defaults when tuple is `None`.
		3. Preserve warning logging when tuple is `None` despite Title + URL.
	- Acceptance: `poetry run pytest tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py -k extract_slug` passes all new cases.
	- **Completed**: All modifications already implemented in P2-T1. Verified: (1) Tuple unpacking at lines 285-291 extracts slug and perma_type from extract_slug_from_content_url result. (2) Both static feed (lines 329-338) and Browse API (lines 363-372) paths check perma_type first and use it when available, falling back to host-based defaults only when None. (3) Warning logging preserved at lines 293-307 when slug_and_type is None despite Title + URL present. Acceptance test passes: 9 extract_slug tests all pass. Full toolchain passes: Black ✓, Ruff ✓, Pyright ✓, Pytest 1308 passed ✓, Coverage 88% ✓.

### Phase 3 — Integration validation (satisfy REQ-004)

- [x] [P3-T1] Regenerate CK-12 catalog and enrichment artifacts
	- Commands:
		1. `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_catalog --out-dir data/meta/catalogs`
		2. `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_enrichment --catalog-file data/meta/catalogs/ck12_catalog.jsonl --output data/meta/catalogs/ck12_enriched.jsonl`
	- Acceptance: Verify sample handles (`cbse-biology-class-10`, `CK-12-Biology-Teachers-Edition`, `CK-12-Biology-Workbook`, `CK-12-Biology-Quizzes-and-Tests`) now carry artifact types `cbook`, `tebook`, `workbook`, `quizbook` in `ck12_catalog.jsonl`.
	- ✅ Completed: Generated 165 catalog entries, all sample handles verified with correct artifact types. Full toolchain passed (Black, Ruff, Pyright, Pytest: 1308 tests, 88% coverage).

- [x] [P3-T2] Run curation with `--require-json` and validate drops decrease
	- Command: `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_curation --catalog-dir data/meta/catalogs --require-json --sources "ck12" --out-dir data/meta/catalogs`
	- Acceptance: Document before/after counts for `ck12_curated.jsonl` kept/skipped totals; sample handles above must not appear in `ck12_skips.jsonl` with `no json candidate`.
	- ✅ Completed: BEFORE: 84 curated, 81 skipped | AFTER: 155 curated (+71, 85% increase), 10 skipped (-71, 88% decrease). All four sample handles (cbse-biology-class-10, ck-12-biology-teachers-edition, ck-12-biology-workbook, ck-12-biology-quizzes-and-tests) are now in curated file, none in skips with "no json candidate". Full toolchain passed (Black, Ruff, Pyright, Pytest: 1308 tests, 88% coverage).

### Phase 4 — Documentation & housekeeping (REQ-006)

- [x] [P4-T1] Update CK-12 documentation references
	- Files: `docs/developer-tooling.md` (CK-12 workflow section) and `docs/source-curation-guide.md`.
	- Content: Explicitly state that artifact types derive from `Content_URL` path prefixes and list supported values.
	- Acceptance: `grep -n "artifact type" docs/developer-tooling.md` shows the new note; same for `docs/source-curation-guide.md`.
	- ✅ Completed: Added "OER Curation Workflows" section to developer-tooling.md (lines 240, 248) and artifact type note to source-curation-guide.md (lines 207, 214). Both documents now explicitly list all five supported artifact types (cbook, book, tebook, workbook, quizbook) and explain they derive from Content_URL path prefixes. Full toolchain passed (Black, Ruff, Pyright, Pytest: 1308 tests, 88% coverage).

### Phase 5 — Final QC loop (REQ-005)

- [x] [P5-T1] Run `poetry run black .` (record PASS/FAIL)
- [x] [P5-T2] Run `poetry run ruff check` (record PASS/FAIL)
- [x] [P5-T3] Run `poetry run pyright` (record PASS/FAIL)
- [x] [P5-T4] Run `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing` (record PASS/FAIL)
- [x] [P5-T5] If any command above fails or edits files, restart from [P5-T1] until all pass sequentially; log final PASS summary.
	- ✅ Completed: Ran all four Phase 5 commands sequentially - all PASSED on first run. Black: 206 files unchanged. Ruff: All checks passed. Pyright: 0 errors. Pytest: 1308 tests passed, 88% coverage. No failures, no files edited, no restart required.

### Phase 6 — Rollout & traceability

- [x] [P6-T1] Update issue #100 with summary, curation before/after metrics, and QC evidence
	- Acceptance: Issue comment lists commands run, PASS results, and artifact file paths.
	- ✅ Completed: Posted comprehensive comment to https://github.com/drmoisan/lexile-corpus-tuner/issues/100#issuecomment-3778311621 including: (1) Fix summary and root cause, (2) Before/after metrics (84→155 curated, 81→10 skipped), (3) QC evidence with all toolchain commands and PASS results, (4) Complete artifact file paths for code, docs, and generated catalogs. Full toolchain passed (Black, Ruff, Pyright, Pytest: 1308 tests, 88% coverage).

- [x] [P6-T2] Capture artifact links (issue, PR, regenerated catalog/enriched/curated files) within this plan for future traceability
	- Acceptance: Plan notes include bullet list of artifact file paths + PR URL once created.
	- ✅ Completed: Added traceability section below with all artifact links.

---

## Traceability Artifacts

**Issue:**
- GitHub Issue #100: https://github.com/drmoisan/lexile-corpus-tuner/issues/100
- Issue Comment (Implementation Summary): https://github.com/drmoisan/lexile-corpus-tuner/issues/100#issuecomment-3778311621

**Pull Request:**
- Branch: `fix/curation-skips-valid-entries-#100`
- PR URL: *(To be created - branch ready for PR submission)*

**Modified Source Code:**
- `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_catalog.py`
  - Updated `extract_slug_from_content_url` to return `(slug, artifact_type)` tuples
  - Modified `parse_catalog_json` to consume tuple results and use detected artifact types
- `tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py`
  - Updated 4 tests to expect tuple returns from `extract_slug_from_content_url`
  - Fixed assertions to expect correct artifact types (e.g., "cbook" instead of "flexbook")

**Updated Documentation:**
- `docs/developer-tooling.md`
  - Added "OER Curation Workflows" section with CK-12 artifact type documentation
- `docs/source-curation-guide.md`
  - Added artifact type note in CK-12 workflow section (step 1)

**Generated Data Artifacts:**
- `data/meta/catalogs/ck12_catalog.jsonl` (165 entries with correct artifact types)
- `data/meta/catalogs/ck12_enriched.jsonl` (165 entries with JSON candidates)
- `data/meta/catalogs/ck12_curated.jsonl` (155 curated entries)
- `data/meta/catalogs/ck12_skips.jsonl` (10 skipped entries)

**Implementation Plan:**
- `docs/features/active/2026-01-21-curation-skips-valid-entries-100/plan.2026-01-21T07-11.md` (this file)
- `docs/features/active/2026-01-21-curation-skips-valid-entries-100/spec.md`

**Metrics:**
- Before: 84 curated, 81 skipped (49% skip rate)
- After: 155 curated, 10 skipped (6% skip rate)
- Improvement: +71 curated entries (85% increase), -71 skipped entries (88% decrease)
