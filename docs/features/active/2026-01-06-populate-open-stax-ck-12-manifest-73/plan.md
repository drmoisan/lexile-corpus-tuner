# 2026-01-06-populate-open-stax-ck-12-manifest-73 - Plan

- Issue: #73
- Owner: 2026-01-06-populate-open-stax-ck-12-manifest-73
- Last Updated: 2026-01-06

## Required References

- General Coding Standards: [`.github/instructions/general-code-change.instructions.md`](../../../../.github/instructions/general-code-change.instructions.md)
- General Unit Test Policy: [`.github/instructions/general-unit-test.instructions.md`](../../../../.github/instructions/general-unit-test.instructions.md)
- Python Code Standards: [`.github/instructions/python-code-change.instructions.md`](../../../../.github/instructions/python-code-change.instructions.md)
- Python Unit Test Policy: [`.github/instructions/python-unit-test.instructions.md`](../../../../.github/instructions/python-unit-test.instructions.md)

**All work must comply with these policies; do not duplicate their content here.**

## Implementation Plan (Atomic Tasks)

### Phase 0 — Context & Inputs
- [] [P0-T1] Read .github/instructions/general-code-change.instructions.md and confirm plan format requirements
- [] [P0-T2] Read .github/instructions/general-unit-test.instructions.md and confirm testing requirements
- [] [P0-T3] Read .github/instructions/python-code-change.instructions.md and confirm Python-specific rules
- [] [P0-T4] Read .github/instructions/python-unit-test.instructions.md and confirm Pytest requirements
- [] [P0-T5] Read docs/features/active/2026-01-06-populate-open-stax-ck-12-manifest-73/spec.md and list all required CLI commands
- [] [P0-T6] Read docs/features/active/2026-01-06-populate-open-stax-ck-12-manifest-73/user-story.md and extract acceptance criteria
- [] [P0-T7] Verify plan.md front matter has no placeholders

### Phase 1 — Requirements Enumeration
- [] [P1-T1] Document required catalog fields from spec: source_id, identifier, title, creator, year, language, license_url, download_candidates
- [] [P1-T2] Document required manifest fields from spec: source_id, id (slug), url, filename (.txt only)
- [] [P1-T3] Document slug derivation rule from spec: use immutable IA identifier, enforce lowercase hyphen format, guarantee idempotence
- [] [P1-T4] Document filtering constraints from spec: OpenStax/CK-12 collections only, text/plain format only (_djvu.txt preferred), skip non-text with logged reason
- [] [P1-T5] Document validation rule from spec: all manifest URLs must return HTTP 200 and Content-Type text/*
- [] [P1-T6] List 6 required CLI tools from spec: build_oer_catalog, enrich_oer_catalog, curate_oer_manifest, generate_oer_manifest, curate_oer_ui, plus existing download/normalize

### Phase 2 — Module Structure Design
- [] [P2-T1] Create module `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/oer_catalog.py` for catalog builder
- [] [P2-T2] Create module `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/oer_enrichment.py` for metadata API enrichment
- [] [P2-T3] Create module `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/oer_curation.py` for filtering logic
- [] [P2-T4] Create module `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/oer_manifest.py` for manifest generation
- [] [P2-T5] Create module `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/oer_ui.py` for Tkinter UI
- [] [P2-T6] Create shared module `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/oer_models.py` for dataclasses (CatalogEntry, ManifestEntry, DownloadCandidate)

### Phase 3 — Implement Core Utilities
- [] [P3-T1] Implement function `generate_stable_slug(ia_identifier: str) -> str` in oer_models.py that converts IA identifier to lowercase-hyphen slug
- [] [P3-T2] Implement dataclass `CatalogEntry` in oer_models.py with fields: source_id, identifier, title, creator, year, language, license_url, download_candidates (list)
- [] [P3-T3] Implement dataclass `ManifestEntry` in oer_models.py with fields: source_id, id (slug), url, filename
- [] [P3-T4] Implement dataclass `DownloadCandidate` in oer_models.py with fields: format, url, size

### Phase 4 — Implement Catalog Builder
- [] [P4-T1] Implement function `build_ia_query(source: str) -> str` in oer_catalog.py that returns IA Advanced Search query for OpenStax or CK-12
- [] [P4-T2] Implement function `fetch_ia_search_results(query: str) -> list[dict]` in oer_catalog.py that calls IA search API and returns raw JSON results
- [] [P4-T3] Implement function `parse_catalog_entry(raw: dict, source_id: str) -> CatalogEntry` in oer_catalog.py that extracts required fields from IA search result
- [] [P4-T4] Implement function `write_catalog_jsonl(entries: list[CatalogEntry], output_path: Path) -> None` in oer_catalog.py that writes entries to JSONL file
- [] [P4-T5] Implement CLI entrypoint `build_oer_catalog` in oer_catalog.py with args: --sources (csv), --out-dir (default: data/meta/catalogs)

### Phase 5 — Implement Enrichment
- [] [P5-T1] Implement function `fetch_ia_metadata(identifier: str) -> dict` in oer_enrichment.py that calls IA Metadata API and returns files list
- [] [P5-T2] Implement function `extract_text_candidates(files: list[dict]) -> list[DownloadCandidate]` in oer_enrichment.py that filters for _djvu.txt and text/plain formats
- [] [P5-T3] Implement function `enrich_catalog_entry(entry: CatalogEntry) -> CatalogEntry` in oer_enrichment.py that appends download_candidates via metadata API
- [] [P5-T4] Implement CLI entrypoint `enrich_oer_catalog` in oer_enrichment.py with args: --catalog-file, --output (overwrites or creates new enriched catalog)

### Phase 6 — Implement Curation Filter
- [] [P6-T1] Implement function `has_text_candidate(entry: CatalogEntry) -> bool` in oer_curation.py that returns True if download_candidates contains text/plain format
- [] [P6-T2] Implement function `filter_by_collection(entry: CatalogEntry, allowed: list[str]) -> bool` in oer_curation.py that checks source_id against allowed list
- [] [P6-T3] Implement function `curate_entries(entries: list[CatalogEntry], require_text: bool, allowed_sources: list[str]) -> tuple[list[CatalogEntry], list[tuple[str, str]]]` in oer_curation.py that returns (included, skipped_with_reason)
- [] [P6-T4] Implement CLI entrypoint `curate_oer_catalog` in oer_curation.py with args: --catalog-dir, --require-text, --sources (default: openstax,ck12), --out-dir

### Phase 7 — Implement Manifest Generator
- [] [P7-T1] Implement function `build_manifest_entry(catalog_entry: CatalogEntry, candidate: DownloadCandidate) -> ManifestEntry` in oer_manifest.py that creates ManifestEntry with stable slug
- [] [P7-T2] Implement function `validate_url(url: str) -> tuple[bool, int, str]` in oer_manifest.py that performs HTTP HEAD request and returns (success, status_code, content_type)
- [] [P7-T3] Implement function `generate_manifest(curated_entries: list[CatalogEntry], validate_urls: bool) -> list[ManifestEntry]` in oer_manifest.py that builds and optionally validates manifest entries
- [] [P7-T4] Implement function `write_manifest_json(entries: list[ManifestEntry], output_path: Path) -> None` in oer_manifest.py that writes to oer_sources.json format
- [] [P7-T5] Implement CLI entrypoint `generate_oer_manifest` in oer_manifest.py with args: --catalog-dir, --out (default: data/meta/oer_sources.json), --validate-urls

### Phase 8 — Implement Visual Curation UI
- [] [P8-T1] Implement function `load_catalog_files(catalog_dir: Path) -> list[CatalogEntry]` in oer_ui.py that reads all JSONL files from catalog directory
- [] [P8-T2] Implement class `CatalogViewModel` in oer_ui.py with methods: get_entries(), toggle_selection(index), get_selected_entries()
- [] [P8-T3] Implement function `create_catalog_table(parent: tk.Widget, viewmodel: CatalogViewModel) -> tk.Frame` in oer_ui.py that renders catalog entries with checkboxes
- [] [P8-T4] Implement function `create_filter_panel(parent: tk.Widget, on_filter: Callable) -> tk.Frame` in oer_ui.py that provides subject/grade/language filter controls
- [] [P8-T5] Implement function `export_manifest(entries: list[CatalogEntry], output_path: Path) -> None` in oer_ui.py that calls manifest generation and saves file
- [] [P8-T6] Implement CLI entrypoint `curate_oer_ui` in oer_ui.py with no args (launches Tkinter window)

### Phase 9 — Integration & Documentation
- [] [P9-T1] Update README.md CLI reference section to document `build_oer_catalog` command with example
- [] [P9-T2] Update README.md CLI reference section to document `enrich_oer_catalog` command with example
- [] [P9-T3] Update README.md CLI reference section to document `curate_oer_catalog` command with example
- [] [P9-T4] Update README.md CLI reference section to document `generate_oer_manifest` command with example
- [] [P9-T5] Update README.md CLI reference section to document `curate_oer_ui` command with example
- [] [P9-T6] Update docs/source-curation-guide.md to add OpenStax/CK-12 workflow section referencing new CLI tools

### Phase 10 — Unit Tests (Utilities)
- [] [P10-T1] Create test file `tests/lexile_scoring_model/pipeline_scripts/test_oer_models.py`
- [] [P10-T2] Add unit test `test_generate_stable_slug_converts_to_lowercase_hyphens` in test_oer_models.py verifying "OpenStax_Book" → "openstax-book"
- [] [P10-T3] Add unit test `test_generate_stable_slug_is_idempotent` in test_oer_models.py verifying repeated calls return same result
- [] [P10-T4] Add unit test `test_catalog_entry_dataclass_validates_required_fields` in test_oer_models.py verifying TypeError on missing fields
- [] [P10-T5] Add unit test `test_manifest_entry_dataclass_enforces_txt_extension` in test_oer_models.py verifying filename ends with .txt

### Phase 11 — Unit Tests (Catalog Builder)
- [] [P11-T1] Create test file `tests/lexile_scoring_model/pipeline_scripts/test_oer_catalog.py`
- [] [P11-T2] Add unit test `test_build_ia_query_for_openstax_includes_collection_filter` in test_oer_catalog.py verifying query contains "collection:openstaxcnx"
- [] [P11-T3] Add unit test `test_build_ia_query_for_ck12_includes_collection_filter` in test_oer_catalog.py verifying query contains "collection:ck12"
- [] [P11-T4] Add unit test `test_parse_catalog_entry_extracts_all_required_fields` in test_oer_catalog.py with mocked IA result verifying CatalogEntry fields populated
- [] [P11-T5] Add unit test `test_parse_catalog_entry_handles_missing_optional_fields` in test_oer_catalog.py verifying defaults for year/creator when absent
- [] [P11-T6] Add unit test `test_write_catalog_jsonl_creates_valid_jsonl_format` in test_oer_catalog.py verifying each line is valid JSON

### Phase 12 — Unit Tests (Enrichment)
- [] [P12-T1] Create test file `tests/lexile_scoring_model/pipeline_scripts/test_oer_enrichment.py`
- [] [P12-T2] Add unit test `test_extract_text_candidates_filters_djvu_txt_files` in test_oer_enrichment.py verifying _djvu.txt files are included
- [] [P12-T3] Add unit test `test_extract_text_candidates_excludes_pdf_files` in test_oer_enrichment.py verifying PDF files are excluded
- [] [P12-T4] Add unit test `test_extract_text_candidates_includes_text_plain_format` in test_oer_enrichment.py verifying format=text/plain files are included
- [] [P12-T5] Add unit test `test_enrich_catalog_entry_appends_download_candidates` in test_oer_enrichment.py with mocked metadata API verifying candidates list updated

### Phase 13 — Unit Tests (Curation)
- [] [P13-T1] Create test file `tests/lexile_scoring_model/pipeline_scripts/test_oer_curation.py`
- [] [P13-T2] Add unit test `test_has_text_candidate_returns_true_when_text_present` in test_oer_curation.py verifying entry with text/plain candidate passes
- [] [P13-T3] Add unit test `test_has_text_candidate_returns_false_when_only_pdf` in test_oer_curation.py verifying entry with only PDF fails
- [] [P13-T4] Add unit test `test_filter_by_collection_accepts_openstax` in test_oer_curation.py verifying source_id=openstax passes filter
- [] [P13-T5] Add unit test `test_filter_by_collection_rejects_gutenberg` in test_oer_curation.py verifying source_id=gutenberg fails filter
- [] [P13-T6] Add unit test `test_curate_entries_returns_included_and_skipped_lists` in test_oer_curation.py verifying tuple structure
- [] [P13-T7] Add unit test `test_curate_entries_logs_skip_reason_for_missing_text` in test_oer_curation.py verifying skipped list contains ("identifier", "no text candidate")

### Phase 14 — Unit Tests (Manifest)
- [] [P14-T1] Create test file `tests/lexile_scoring_model/pipeline_scripts/test_oer_manifest.py`
- [] [P14-T2] Add unit test `test_build_manifest_entry_uses_stable_slug_from_identifier` in test_oer_manifest.py verifying id field uses generate_stable_slug result
- [] [P14-T3] Add unit test `test_build_manifest_entry_sets_filename_to_txt_extension` in test_oer_manifest.py verifying filename ends with .txt
- [] [P14-T4] Add unit test `test_validate_url_returns_true_for_http_200_text_content_type` in test_oer_manifest.py with mocked HTTP HEAD returning 200 and text/plain
- [] [P14-T5] Add unit test `test_validate_url_returns_false_for_http_404` in test_oer_manifest.py with mocked HTTP HEAD returning 404
- [] [P14-T6] Add unit test `test_validate_url_returns_false_for_application_pdf_content_type` in test_oer_manifest.py with mocked HTTP HEAD returning 200 but application/pdf
- [] [P14-T7] Add unit test `test_write_manifest_json_creates_valid_json_structure` in test_oer_manifest.py verifying output file is valid JSON array

### Phase 15 — Unit Tests (UI Logic)
- [] [P15-T1] Create test file `tests/lexile_scoring_model/pipeline_scripts/test_oer_ui.py`
- [] [P15-T2] Add unit test `test_load_catalog_files_reads_all_jsonl_in_directory` in test_oer_ui.py with temp directory containing 2 JSONL files
- [] [P15-T3] Add unit test `test_catalog_viewmodel_toggle_selection_updates_state` in test_oer_ui.py verifying toggle changes selection
- [] [P15-T4] Add unit test `test_catalog_viewmodel_get_selected_entries_returns_only_selected` in test_oer_ui.py verifying filtering logic
- [] [P15-T5] Add unit test `test_export_manifest_calls_manifest_generation` in test_oer_ui.py with mocked generate_manifest verifying function called

### Phase 16 — Integration Tests
- [] [P16-T1] Create test file `tests/lexile_scoring_model/pipeline_scripts/test_oer_integration.py`
- [] [P16-T2] Add integration test `test_end_to_end_catalog_to_manifest_with_mocked_ia` in test_oer_integration.py that runs catalog→enrich→curate→manifest with mocked IA API
- [] [P16-T3] Add integration test `test_manifest_entries_validate_against_schema` in test_oer_integration.py verifying all required fields present
- [] [P16-T4] Add integration test `test_download_normalize_consumes_manifest_without_errors` in test_oer_integration.py that verifies existing corpus pipeline can parse manifest

### Phase 17 — Validation & Toolchain
- [] [P17-T1] Run `poetry run black .` on all modified files
- [] [P17-T2] Run `poetry run ruff check` and fix all reported issues
- [] [P17-T3] Run `poetry run pyright` and fix all type errors
- [] [P17-T4] Run `poetry run pytest tests/lexile_scoring_model/pipeline_scripts/test_oer_models.py -v` and verify all pass
- [] [P17-T5] Run `poetry run pytest tests/lexile_scoring_model/pipeline_scripts/test_oer_catalog.py -v` and verify all pass
- [] [P17-T6] Run `poetry run pytest tests/lexile_scoring_model/pipeline_scripts/test_oer_enrichment.py -v` and verify all pass
- [] [P17-T7] Run `poetry run pytest tests/lexile_scoring_model/pipeline_scripts/test_oer_curation.py -v` and verify all pass
- [] [P17-T8] Run `poetry run pytest tests/lexile_scoring_model/pipeline_scripts/test_oer_manifest.py -v` and verify all pass
- [] [P17-T9] Run `poetry run pytest tests/lexile_scoring_model/pipeline_scripts/test_oer_ui.py -v` and verify all pass
- [] [P17-T10] Run `poetry run pytest tests/lexile_scoring_model/pipeline_scripts/test_oer_integration.py -v` and verify all pass
- [] [P17-T11] Run `poetry run pytest --cov=src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts --cov-report=term-missing` and verify ≥90% coverage for new modules
- [] [P17-T12] Update plan.md Last Updated date to current date

## Test Plan

**Unit Tests (94 total)**
- Utilities (5): slug conversion, idempotence, dataclass validation, filename extension enforcement
- Catalog Builder (5): query building (OpenStax/CK-12), entry parsing with all fields, missing optional fields, JSONL format
- Enrichment (4): _djvu.txt filtering, PDF exclusion, text/plain inclusion, candidate appending
- Curation (6): text candidate detection (positive/negative), collection filtering (accept/reject), skip reason logging
- Manifest (6): stable slug usage, .txt filename enforcement, URL validation (200/text, 404, PDF rejection), JSON structure
- UI Logic (4): catalog loading from directory, viewmodel selection toggle/filtering, manifest export
- Integration (3): end-to-end mocked IA flow, schema validation, download/normalize compatibility

**Manual Verification**
- Run each CLI command (build_oer_catalog, enrich_oer_catalog, curate_oer_catalog, generate_oer_manifest, curate_oer_ui) with sample data
- Verify catalog JSONL files are created with correct fields
- Verify manifest JSON matches schema and all URLs are .txt
- Verify Tkinter UI loads, filters, and exports without errors
- Verify existing corpus download/normalize commands consume manifest successfully

## Open Questions / Notes

- ...
