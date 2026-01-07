# 2026-01-06-populate-open-stax-ck-12-manifest-73 — CK-12 Follow-on Plan

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
- [ ] [P0-T1] Re-read spec.md focusing on CK-12 native scraping/PDF steps and manifest schema updates
- [ ] [P0-T2] Re-read .github/copilot-instructions.md, general-code-change, and python-code-change to confirm guardrails
- [ ] [P0-T3] Re-read general-unit-test and python-unit-test policies to confirm test expectations and scenario rules

### Phase 1 — Dependencies & Design
- [ ] [P1-T1] Add pdfplumber>=0.10.0 to pyproject.toml
- [ ] [P1-T2] Add beautifulsoup4 to pyproject.toml
- [ ] [P1-T3] Add requests to pyproject.toml
- [ ] [P1-T4] Add types-beautifulsoup4 to pyproject.toml (for Pyright)
- [ ] [P1-T5] Add types-requests to pyproject.toml (for Pyright)
- [ ] [P1-T6] Create design note `ck12-design.md` defining PDF validation patterns and fallback strategy
- [ ] [P1-T7] Research if pdfplumber requires system binaries and document findings in `ck12-design.md`

### Phase 2 — CK-12 Catalog Scraper (ck12_catalog.py)
- [ ] [P2-T1] Create module `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_catalog.py`
- [ ] [P2-T2] Implement `fetch_catalog_page(url: str) -> str` in ck12_catalog.py using requests
- [ ] [P2-T3] Implement `parse_catalog_rows(html: str) -> list[CatalogEntry]` in ck12_catalog.py using BeautifulSoup and `oer_models.generate_stable_slug`
- [ ] [P2-T4] Implement `write_catalog_jsonl(rows: list, path: Path)` in ck12_catalog.py
- [ ] [P2-T5] Implement CLI entrypoint `build_ck12_catalog` in ck12_catalog.py using Typer

### Phase 3 — CK-12 Enrichment (ck12_enrichment.py)
- [ ] [P3-T1] Create module `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_enrichment.py`
- [ ] [P3-T2] Implement `fetch_flexbook_html(url: str) -> str` in ck12_enrichment.py
- [ ] [P3-T3] Implement `parse_flexbook_metadata(html: str) -> dict` in ck12_enrichment.py (extracts author, grade, language)
- [ ] [P3-T4] Implement `extract_pdf_url(html: str) -> str | None` in ck12_enrichment.py
- [ ] [P3-T5] Implement `enrich_entry_logic(entry: CatalogEntry, metadata: dict, pdf_url: str | None) -> CatalogEntry` in ck12_enrichment.py
- [ ] [P3-T6] Implement CLI entrypoint `enrich_ck12_catalog` in ck12_enrichment.py using Typer

### Phase 4 — PDF Extraction (extract_pdf_text.py)
- [ ] [P4-T1] Create module `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/extract_pdf_text.py`
- [ ] [P4-T2] Implement `extract_text_from_pdf(pdf_path: Path) -> str` in extract_pdf_text.py using pdfplumber
- [ ] [P4-T3] Implement `save_text_file(text: str, output_path: Path)` in extract_pdf_text.py
- [ ] [P4-T4] Implement `process_ck12_directory(input_dir: Path, output_dir: Path)` in extract_pdf_text.py (iterates PDFs, extracts text parallel)
- [ ] [P4-T5] Implement error logging logic for failed extractions in extract_pdf_text.py
- [ ] [P4-T6] Implement CLI entrypoint `extract_pdf_text` in extract_pdf_text.py using Typer

### Phase 5 — Pipeline Integration
- [ ] [P5-T1] Update `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/oer_curation.py`: Add `has_pdf_candidate` function
- [ ] [P5-T2] Update `oer_curation.py`: Add `--require-pdf` argument filter logic to `curate_oer_catalog` CLI
- [ ] [P5-T3] Update `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/oer_manifest.py`: specific update to `build_manifest_entry` to determine extension from candidate format (if application/pdf use .pdf, else .txt)
- [ ] [P5-T4] Update `oer_manifest.py`: Update `validate_url` signature to accept `allowed_content_types` list and use it in the HEAD check
- [ ] [P5-T5] Update `oer_manifest.py`: Update `generate_manifest` to pass `['application/pdf']` when validating CK-12 candidates

### Phase 6 — Tests (Scenarios)
- [ ] [P6-T1] Add test for fetching catalog (mocked) in `tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py`
- [ ] [P6-T2] Add test for parsing catalog rows (valid HTML fixture) in `test_ck12_catalog.py`
- [ ] [P6-T3] Add test for parsing catalog rows (missing metadata) in `test_ck12_catalog.py`
- [ ] [P6-T4] Add test for slug generation idempotence in `test_ck12_catalog.py`
- [ ] [P6-T5] Add test for fetching enrichment (mocked) in `tests/lexile_scoring_model/pipeline_scripts/test_ck12_enrichment.py`
- [ ] [P6-T6] Add test for parsing flexbook metadata (fixture) in `test_ck12_enrichment.py`
- [ ] [P6-T7] Add test for parsing PDF URL from FLEXBOOK HTML in `test_ck12_enrichment.py`
- [ ] [P6-T8] Add test for successful PDF extraction (fixture) in `tests/lexile_scoring_model/pipeline_scripts/test_extract_pdf_text.py`
- [ ] [P6-T9] Add test for PDF extraction failure handling in `test_extract_pdf_text.py`
- [ ] [P6-T10] Add test for oer_curation filter logic (pdf required) in `tests/lexile_scoring_model/pipeline_scripts/test_oer_curation.py`
- [ ] [P6-T11] Add test for `build_manifest_entry` setting .pdf extension for PDF candidates in `tests/lexile_scoring_model/pipeline_scripts/test_oer_manifest.py`
- [ ] [P6-T12] Add test for `validate_url` accepting application/pdf when configured in `test_oer_manifest.py`

### Phase 7 — Documentation
- [ ] [P7-T1] Update `docs/features/active/2026-01-06-populate-open-stax-ck-12-manifest-73/spec.md` with any implementation deviations
- [ ] [P7-T2] Update `docs/source-curation-guide.md` with new CK-12 workflow steps
- [ ] [P7-T3] Update `README.md` with new CLI commands documentation

### Phase 8 — Validation
- [ ] [P8-T1] Run `poetry lock --no-update` to confirm dependencies
- [ ] [P8-T2] Run full toolchain (format, lint, type-check, test)
- [ ] [P8-T3] Run integration verification manually (Catalog -> Enrich -> Curate -> Manifest -> Extract)
- [ ] [P8-T4] Benchmark PDF extraction speed on sample
- [ ] [P8-T5] Verify PDF extraction quality manually (checking formatting/table loss)

## Notes
- Keep tasks scoped to CK-12 follow-on only; OpenStax remains stable and unchanged.
- Respect 500-line file limit and strict typing/docstring/commenting policies.
- Do not introduce temporary files in tests; use in-memory fixtures/mocking for HTML/PDF content.