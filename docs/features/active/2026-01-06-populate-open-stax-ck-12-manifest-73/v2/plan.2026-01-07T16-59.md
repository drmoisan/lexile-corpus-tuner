# 2026-01-06-populate-open-stax-ck-12-manifest-73 — CK-12 Follow-on Plan

- **Status:** Executed as Designed
- **Outcome:** Policy Audit Failure
- **Root Cause:** Spec Based on Incorrect Endpoint Assumption
- Issue: #73
- Owner: drmoisan
- Last Updated: 2026-01-07

## Required References

- General Coding Standards: `.github/instructions/general-code-change.instructions.md`
- General Unit Test Policy: `.github/instructions/general-unit-test.instructions.md`
- Python Code Standards: `.github/instructions/python-code-change.instructions.md`
- Python Unit Test Policy: `.github/instructions/python-unit-test.instructions.md`
- Spec (revised): `docs/features/active/2026-01-06-populate-open-stax-ck-12-manifest-73/spec.md`
- User Story: `docs/features/active/2026-01-06-populate-open-stax-ck-12-manifest-73/user-story.md`

**All work must comply with these policies; do not duplicate their content here.**

## Implementation Plan (Atomic Tasks)

### Phase 0 — Context & Inputs
- [x] [P0-T1] Read .github/copilot-instructions.md and general-code-change.instructions.md to confirm baseline policies
- [x] [P0-T2] Read general-unit-test.instructions.md and python-unit-test.instructions.md to confirm testing and scenario rules
- [x] [P0-T3] Read python-code-change.instructions.md and self-explanatory-code-commenting.instructions.md to confirm typing/docstring/comment guardrails
- [x] [P0-T4] Re-read docs/features/active/2026-01-06-populate-open-stax-ck-12-manifest-73/spec.md for CK-12 scope and acceptance criteria
- [x] [P0-T5] Re-read docs/features/active/2026-01-06-populate-open-stax-ck-12-manifest-73/user-story.md for user-facing outcomes and constraints

### Phase 1 — Dependencies & Design Pre-Flight
- [x] [P1-T1] Check pyproject.toml to confirm pdfplumber, beautifulsoup4, requests, types-beautifulsoup4, and types-requests are not already listed
- [x] [P1-T2] Add pdfplumber>=0.10.0 to pyproject.toml
- [x] [P1-T3] Add beautifulsoup4 to pyproject.toml
- [x] [P1-T4] Add requests to pyproject.toml
- [x] [P1-T5] Add types-beautifulsoup4 to pyproject.toml for typing support
- [x] [P1-T6] Add types-requests to pyproject.toml for typing support
- [x] [P1-T7] Create docs/features/active/2026-01-06-populate-open-stax-ck-12-manifest-73/ck12-design.md capturing PDF validation patterns and fallback strategy
- [x] [P1-T8] Research pdfplumber system requirements and record any OS/package prerequisites in ck12-design.md

### Phase 2 — CK-12 Catalog Scraper (ck12_catalog.py)
- [x] [P2-T1] Create src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_catalog.py with module docstring and typed stubs
- [x] [P2-T2] Implement `fetch_catalog_page(url: str) -> str` in ck12_catalog.py using requests with timeout and error handling
- [x] [P2-T3] Implement `parse_catalog_rows(html: str) -> list[CatalogEntry]` in ck12_catalog.py using BeautifulSoup and `oer_models.generate_stable_slug`
- [x] [P2-T4] Implement `write_catalog_jsonl(rows: list[CatalogEntry], path: Path)` in ck12_catalog.py with deterministic ordering and overwrite safety
- [x] [P2-T5] Add Typer CLI entrypoint `build_ck12_catalog` wiring fetch, parse, and write functions

### Phase 3 — CK-12 Enrichment (ck12_enrichment.py)
- [x] [P3-T1] Create src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_enrichment.py with module docstring and typed stubs
- [x] [P3-T2] Implement `fetch_flexbook_html(url: str) -> str` in ck12_enrichment.py with requests, timeout, and status validation
- [x] [P3-T3] Implement `parse_flexbook_metadata(html: str) -> dict[str, str | None]` in ck12_enrichment.py extracting author, grade, and language fields
- [x] [P3-T4] Implement `extract_pdf_url(html: str) -> str | None` in ck12_enrichment.py parsing FLEXBOOK HTML for PDF link
- [x] [P3-T5] Implement `enrich_entry_logic(entry: CatalogEntry, metadata: dict[str, str | None], pdf_url: str | None) -> CatalogEntry` in ck12_enrichment.py merging enrichment
- [x] [P3-T6] Add Typer CLI entrypoint `enrich_ck12_catalog` wiring fetch, parse, and enrich behaviors

### Phase 4 — PDF Extraction (extract_pdf_text.py)
- [x] [P4-T1] Create src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/extract_pdf_text.py with module docstring and typed stubs
- [x] [P4-T2] Implement `extract_text_from_pdf(pdf_path: Path) -> str` in extract_pdf_text.py using pdfplumber with timeout-safe file handling
- [x] [P4-T3] Implement `save_text_file(text: str, output_path: Path)` in extract_pdf_text.py ensuring parent creation and overwrite safety
- [x] [P4-T4] Implement `process_ck12_directory(input_dir: Path, output_dir: Path)` in extract_pdf_text.py iterating PDFs (consider parallelism where allowed) and invoking extract/save
- [x] [P4-T5] Implement error logging for failed extractions with actionable context
- [x] [P4-T6] Add Typer CLI entrypoint `extract_pdf_text` wiring directory processing

### Phase 5 — Pipeline Integration
- [x] [P5-T1] Add `has_pdf_candidate` helper to `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/oer_curation.py`
- [x] [P5-T2] Extend `curate_oer_catalog` CLI in `oer_curation.py` with `--require-pdf` filter logic
- [x] [P5-T3] Update `build_manifest_entry` in `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/oer_manifest.py` to set extension from content-type (PDF => .pdf, otherwise .txt)
- [x] [P5-T4] Update `validate_url` signature in `oer_manifest.py` to accept `allowed_content_types` list and honor it in HEAD checks
- [x] [P5-T5] Update `generate_manifest` in `oer_manifest.py` to pass `['application/pdf']` for CK-12 candidates

### Phase 6 — Tests (Scenarios)
- [x] [P6-T1] Add test for `fetch_catalog_page` mocked HTTP success in `tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py`
- [x] [P6-T2] Add test for `parse_catalog_rows` with valid HTML fixture in `test_ck12_catalog.py`
- [x] [P6-T3] Add test for `parse_catalog_rows` handling missing metadata gracefully in `test_ck12_catalog.py`
- [x] [P6-T4] Add test for slug generation idempotence in `test_ck12_catalog.py`
- [x] [P6-T5] Add test for `fetch_flexbook_html` mocked HTTP success in `tests/lexile_scoring_model/pipeline_scripts/test_ck12_enrichment.py`
- [x] [P6-T6] Add test for `parse_flexbook_metadata` extracting author/grade/language in `test_ck12_enrichment.py`
- [x] [P6-T7] Add test for `extract_pdf_url` parsing PDF link from FLEXBOOK HTML in `test_ck12_enrichment.py`
- [x] [P6-T8] Add test for `extract_text_from_pdf` successful extraction using fixture PDF in `tests/lexile_scoring_model/pipeline_scripts/test_extract_pdf_text.py`
- [x] [P6-T9] Add test for `extract_text_from_pdf` failure handling (logged/raised) in `test_extract_pdf_text.py`
- [x] [P6-T10] Add test for `curate_oer_catalog` `--require-pdf` filter behavior in `tests/lexile_scoring_model/pipeline_scripts/test_oer_curation.py`
- [x] [P6-T11] Add test for `build_manifest_entry` assigning .pdf when content-type is application/pdf in `tests/lexile_scoring_model/pipeline_scripts/test_oer_manifest.py`
- [x] [P6-T12] Add test for `validate_url` accepting application/pdf when `allowed_content_types` includes it in `test_oer_manifest.py`

### Phase 7 — Documentation
- [x] [P7-T1] Update `docs/features/active/2026-01-06-populate-open-stax-ck-12-manifest-73/spec.md` with any implementation deviations or decisions
- [x] [P7-T2] Update `docs/source-curation-guide.md` with CK-12 workflow steps (catalog, enrich, curate, manifest, extract)
- [x] [P7-T3] Update `README.md` with new CLI commands and flags for CK-12 catalog/enrich/extract

### Phase 8 — Validation & Release Gate
- [x] [P8-T1] Run `poetry lock --no-update` and confirm lockfile sync after dependency additions
- [x] [P8-T2] Run formatter (`poetry run black .`) and confirm no pending formatting changes
- [x] [P8-T3] Run linter (`poetry run ruff check`) and resolve findings
- [x] [P8-T4] Run type checker (`poetry run pyright`) and resolve diagnostics
- [x] [P8-T5] Run tests (`poetry run pytest`) and ensure all new scenarios pass
- [x] [P8-T6] Execute manual integration flow Catalog -> Enrich -> Curate -> Manifest -> Extract on a small CK-12 sample and record outcomes
- [x] [P8-T7] Benchmark PDF extraction time on a representative CK-12 sample and note results in `ck12-design.md`
- [x] [P8-T8] Manually review extracted text quality (formatting/table retention) on sample PDFs and capture observations in `ck12-design.md`

## Notes
- Keep tasks scoped to CK-12 follow-on only; OpenStax remains stable and unchanged.
- Respect 500-line file limit and strict typing/docstring/commenting policies.
- Do not introduce temporary files in tests; use in-memory fixtures/mocking for HTML/PDF content.
