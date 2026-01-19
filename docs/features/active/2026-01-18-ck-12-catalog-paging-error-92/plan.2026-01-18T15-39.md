# 2026-01-18-ck-12-catalog-paging-error (Plan)

- **Issue:** #92
- **Owner:** drmoisan
- **Date:** 2026-01-18T15-39
- **Branch:** feature/populate-open-stax-ck-12-manifest-#73
- **Commit:** a701c36e9ef0cfdce0dcec69bd06c032fe161231
- **Status:** Planned
- **Outcome:** CK-12 catalog ingestion uses the static FlexBook browse feed, with deterministic parsing for `/cbook/`, `/user:<handle>/cbook/`, and `/book/` URL patterns plus unit tests that validate slug derivation, dedupe behavior, and fallback logic.
- **Root Cause:** The Browse API ignores `limit`/`offset` and always returns the first 10 items, so the current single-request implementation can never reach the full catalog.

Status Badge: ![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

## Overview
This plan updates the CK-12 catalog ingestion to use the static FlexBook browse feed and extends parsing logic to handle all observed URL patterns while keeping tests deterministic and 1-to-1 with the production module.

## Requirements Traceability
| REQ-ID | Description | Source | Planned Tasks |
| --- | --- | --- | --- |
| REQ-01 | Use static feed `https://static.ck12.org/testimonial/fbbrowse-prod.json` as default CK-12 catalog URL. | `spec.md` | P3-T1, P3-T2 |
| REQ-02 | Parse static feed `books` entries and derive slugs from `/cbook/`, `/user:<handle>/cbook/`, and `/book/` URL patterns. | `spec.md`, research | P3-T3, P3-T4, P3-T5 |
| REQ-03 | Preserve Browse API parsing for explicit `--catalog-url` usage. | `spec.md` | P3-T6 |
| REQ-04 | Enforce CatalogEntry invariants: non-empty identifier, `language` list, optional artifact fields. | research | P3-T4, P3-T5 |
| REQ-05 | Update `tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py` with deterministic static-feed fixtures and scenarios. | `spec.md` | P2-T1 to P2-T5 |

### Phase 0 — Context & Inputs
- [x] [P0-T1] Read `.github/copilot-instructions.md` and record compliance notes in this plan (TASK-CK12-P0-T1).
	- Acceptance: Plan section “Phase 0 — Context & Inputs” references the file by name and states “Reviewed: yes”.
- [x] [P0-T2] Read `.github/instructions/general-code-change.instructions.md` and record compliance notes in this plan (TASK-CK12-P0-T2).
	- Acceptance: Plan section “Phase 0 — Context & Inputs” references the file by name and states “Reviewed: yes”.
- [x] [P0-T3] Read `.github/instructions/general-unit-test.instructions.md` and record compliance notes in this plan (TASK-CK12-P0-T3).
	- Acceptance: Plan section “Phase 0 — Context & Inputs” references the file by name and states “Reviewed: yes”.
- [x] [P0-T4] Read `.github/instructions/python-code-change.instructions.md` and record compliance notes in this plan (TASK-CK12-P0-T4).
	- Acceptance: Plan section “Phase 0 — Context & Inputs” references the file by name and states “Reviewed: yes”.
- [x] [P0-T5] Read `.github/instructions/python-unit-test.instructions.md` and record compliance notes in this plan (TASK-CK12-P0-T5).
	- Acceptance: Plan section “Phase 0 — Context & Inputs” references the file by name and states “Reviewed: yes”.

- .github/copilot-instructions.md — Reviewed: yes
- .github/instructions/general-code-change.instructions.md — Reviewed: yes
- .github/instructions/general-unit-test.instructions.md — Reviewed: yes
- .github/instructions/python-code-change.instructions.md — Reviewed: yes
- .github/instructions/python-unit-test.instructions.md — Reviewed: yes

- [x] [P0-T6] Capture baseline format results with `poetry run black .` and save output to `artifacts/ck12_catalog_baseline_black.txt` (TASK-CK12-P0-T6).
	- Acceptance: File exists and contains the string `black` and the exit status summary.
- [x] [P0-T7] Capture baseline lint results with `poetry run ruff check` and save output to `artifacts/ck12_catalog_baseline_ruff.txt` (TASK-CK12-P0-T7).
	- Acceptance: File exists and contains the string `ruff` and the exit status summary.

- [x] [P0-T8] Capture baseline type-check results with `poetry run pyright` and save output to `artifacts/ck12_catalog_baseline_pyright.txt` (TASK-CK12-P0-T8).
	- Acceptance: File exists and contains the string `pyright` and the exit status summary.

- [x] [P0-T9] Capture baseline test results with `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing` and save output to `artifacts/ck12_catalog_baseline_pytest.txt` (TASK-CK12-P0-T9).
	- Acceptance: File exists and contains `tests collected` and `passed` or `failed`.

- [x] [P0-T10] Record current branch and commit SHA at the top of this plan (TASK-CK12-P0-T10).
	- Acceptance: The plan header includes a `Branch:` line and a `Commit:` line with concrete values.

### Phase 1 — Test Design (TDD)
- [x] [P1-T1] Define static-feed in-memory fixtures in `tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py` for URL patterns `/cbook/`, `/user:<handle>/cbook/`, and `/book/` (TASK-CK12-P1-T1).
	- Acceptance: The test file includes a fixture dict named `STATIC_FEED_FIXTURE` with three `books` entries covering each URL pattern.
- [x] [P1-T2] Define fallback fixtures in `test_ck12_catalog.py` for missing `Content_URL` and missing `handle` (TASK-CK12-P1-T2).
	- Acceptance: The test file includes a fixture dict named `STATIC_FEED_FALLBACK_FIXTURE` with one entry missing `Content_URL` and one entry missing `handle`.
- [x] [P1-T3] Define dedupe fixture in `test_ck12_catalog.py` with duplicate slugs and differing titles (TASK-CK12-P1-T3).
	- Acceptance: The test file includes a fixture dict named `STATIC_FEED_DEDUPE_FIXTURE` with two `books` entries sharing the same slug.

### Phase 2 — Regression Tests (must fail first)
- [x] [P2-T1] Add `test_parse_catalog_json_accepts_static_feed_books` in `tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py` using `STATIC_FEED_FIXTURE` (TASK-CK12-P2-T1).
	- Acceptance: The test asserts 3 entries, correct `artifact_type` mapping, and slug extraction from each URL pattern.
- [x] [P2-T2] Add `test_parse_catalog_json_static_feed_missing_content_url_falls_back` using `STATIC_FEED_FALLBACK_FIXTURE` (TASK-CK12-P2-T2).
	- Acceptance: The test asserts identifier from `handle`, then slugified `Title` when `handle` missing.
- [x] [P2-T3] Add `test_parse_catalog_json_static_feed_dedupes_by_identifier` using `STATIC_FEED_DEDUPE_FIXTURE` (TASK-CK12-P2-T3).
	- Acceptance: The test asserts a single entry and that the first entry’s title is preserved.
- [x] [P2-T4] Update `test_fetch_catalog_page_targets_browse_api_with_required_headers` to assert default URL equals `https://static.ck12.org/testimonial/fbbrowse-prod.json` (TASK-CK12-P2-T4).
	- Acceptance: The test checks `captured["url"]` equals the static feed URL.
- [x] [P2-T5] Run only the new/updated tests and confirm they fail before code changes (TASK-CK12-P2-T5).
	- Acceptance: `poetry run pytest tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py -k "static_feed or default"` exits non-zero.

### Phase 3 — Implementation (minimal fix)
- [x] [P3-T1] Update `DEFAULT_CK12_CATALOG_URL` in `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_catalog.py` to `https://static.ck12.org/testimonial/fbbrowse-prod.json` (TASK-CK12-P3-T1).
	- Acceptance: The constant equals the new URL and is referenced by `build_ck12_catalog` default option.
- [x] [P3-T2] Add a helper `extract_slug_from_content_url(url: str) -> str | None` in `ck12_catalog.py` to parse `/cbook/`, `/user:<handle>/cbook/`, and `/book/` patterns (TASK-CK12-P3-T2).
	- Acceptance: The helper returns the slug for the three URL patterns and returns `None` for unparseable URLs.
- [x] [P3-T3] Update `parse_catalog_json` to derive identifiers from `Content_URL` via `extract_slug_from_content_url` before falling back to `handle` and `Title` (TASK-CK12-P3-T3).
	- Acceptance: The parser uses the helper first, then `handle`, then slugified `Title` as described in `spec.md`.
- [x] [P3-T4] Update `parse_catalog_json` to map `artifact_type` by hostname (`flexbooks.ck12.org` => `flexbook`, `www.ck12.org` => `book`) and default to `flexbook` when no URL or artifact type is present (TASK-CK12-P3-T4).
	- Acceptance: The output `CatalogEntry.artifact_type` matches the mapping rules and is never empty when an entry is emitted.
- [x] [P3-T5] Update `parse_catalog_json` to allow `artifact_id=None` and to emit `language` as a list even when a single language string is provided (TASK-CK12-P3-T5).
	- Acceptance: The parser does not drop entries solely for missing IDs and always emits `language` as `list[str]`.
- [x] [P3-T6] Preserve Browse API parsing support for `response.flexbook` / `response.items` when `books` is absent (TASK-CK12-P3-T6).
	- Acceptance: Existing tests that parse Browse API shapes continue to pass.

### Phase 4 — Verification
- [x] [P4-T1] Re-run the updated tests for `test_ck12_catalog.py` and confirm they pass (TASK-CK12-P4-T1).
	- Acceptance: `poetry run pytest tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py` exits 0.

### Phase 5 — QA Toolchain Loop
- [x] [P5-T1] Run `poetry run black .` and record output in `artifacts/ck12_catalog_qc_black.txt` (TASK-CK12-P5-T1).
	- Acceptance: Output file exists and indicates no reformatting needed or shows completion without errors.
- [x] [P5-T2] Run `poetry run ruff check` and record output in `artifacts/ck12_catalog_qc_ruff.txt` (TASK-CK12-P5-T2).
	- Acceptance: Output file exists and contains no error entries.
- [x] [P5-T3] Run `poetry run pyright` and record output in `artifacts/ck12_catalog_qc_pyright.txt` (TASK-CK12-P5-T3).
	- Acceptance: Output file exists and contains `0 errors`.
- [x] [P5-T4] Run `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing` and record output in `artifacts/ck12_catalog_qc_pytest.txt` (TASK-CK12-P5-T4).
	- Acceptance: Output file exists and contains `passed` with exit code 0.
- [x] [P5-T5] If any step in P5-T1 to P5-T4 changes files or fails, restart the toolchain loop from P5-T1 until a clean pass completes (TASK-CK12-P5-T5).
	- Acceptance: The final loop completes with all four steps passing in a single run.

### Phase 6 — Documentation & Status
- [x] [P6-T1] Update `docs/features/active/2026-01-18-ck-12-catalog-paging-error-92/spec.md` with final outcomes and any deviations from the plan (TASK-CK12-P6-T1).
	- Acceptance: Spec `Status` is updated to reflect completion and any deviations are explicitly documented.
