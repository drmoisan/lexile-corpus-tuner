# 2026-01-06-populate-open-stax-ck-12-manifest-73 — Plan (Atomic)

- Issue: #73
- Owner: drmoisan
- Last Updated: 2026-01-09

> Timestamp note:
> This plan file name uses the session date (2026-01-09). If you want a different
> minute-level timestamp, rename the file and keep contents unchanged.

## Required References

- General Coding Standards: [`.github/instructions/general-code-change.instructions.md`](../../../../.github/instructions/general-code-change.instructions.md)
- Python Coding Standards: [`.github/instructions/python-code-change.instructions.md`](../../../../.github/instructions/python-code-change.instructions.md)
- General Unit Test Policy: [`.github/instructions/general-unit-test.instructions.md`](../../../../.github/instructions/general-unit-test.instructions.md)
- Python Unit Test Policy: [`.github/instructions/python-unit-test.instructions.md`](../../../../.github/instructions/python-unit-test.instructions.md)
- Feature spec: [`spec.md`](spec.md)
- User story: [`user-story.md`](user-story.md)

**All work must comply with these policies; do not duplicate their content here.**

## Gap Analysis (What’s Left To Build)

- CK-12 catalog is currently sourced from a static JSON (`fbbrowse-prod.json`) and derives IDs from vanity URL slugs; spec requires Browse API discovery and IDs derived from canonical `handle`.
- CK-12 enrichment currently parses FlexBook HTML and looks for PDF links; spec requires Perma API traversal to extract section revision IDs.
- OER curation + manifest currently assume CK-12 is PDF (`application/pdf`); spec requires CK-12 revision JSON (`application/json`) and `.json` filenames.
- Corpus download currently cannot inject CK-12 browser-like headers; spec requires those headers for anonymous CK-12 API access.
- CK-12 JSON → XHTML → text extraction step (`extract_ck12_text`) is not implemented.
- Manifest validation for CK-12 must validate JSON payload shape (`response.lesson.xhtml` / `xhtml_prime`), not just status code.
- Dependency gap: spec requires BeautifulSoup + `lxml` parser for XHTML extraction; `beautifulsoup4` is present, `lxml` is not currently declared.

## Acceptance Criteria Mapping (User Story → Plan)

- CK-12 catalog uses CK-12 Browse API (not IA/static JSON)
  - Covered by Phase 1
- CK-12 enrichment uses CK-12 Perma API + revision hierarchy (not HTML/PDF)
  - Covered by Phase 2
- CK-12 downloads are revision JSON and succeed with browser-like headers
  - Covered by Phase 4 (manifest `.json`) and Phase 5 (downloader headers)
- CK-12 revision JSON is converted to text via XHTML extraction
  - Covered by Phase 6
- OpenStax remains IA-based with text derivatives preferred
  - Preserved by Phases 3–4 (curation/manifest) and existing OpenStax tests

## Implementation Plan (Atomic Tasks, TDD-Ordered)

**Phase 0 — Context & Inputs**
- [x] [P0-T1] Read the repo policies in this order: `.github/copilot-instructions.md`, `.github/instructions/general-code-change.instructions.md`, `.github/instructions/general-unit-test.instructions.md`, then `.github/instructions/python-code-change.instructions.md` and `.github/instructions/python-unit-test.instructions.md`
  - Acceptance: Notes include a timestamp and any constraints that impact this feature (e.g., no external network in unit tests).
  - Notes: 2026-01-11T05:24:31Z — apply general + Python code/unit-test policies; no external network in unit tests; mandatory robust docstrings and intent comments per repo guidance.
- [x] [P0-T2] Re-read [`spec.md`](spec.md) and record the required CK-12 API endpoints + header set to be implemented
  - Acceptance: Notes list Browse, Perma, Revision Detail endpoints and the required headers from the spec.
  - Notes: Browse `https://www.ck12.org/flx/browse/flexbook?limit=200`; Perma `https://www.ck12.org/flx/get/perma/<artifactType>/<handle>`; Revision Detail `https://www.ck12.org/flx/get/detail/revision/<revisionID>?tiny=true`; Required headers — `User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36`, `Accept: application/json, text/plain, */*`, `Referer: https://www.ck12.org/`, `Origin: https://www.ck12.org`, `Sec-Fetch-Dest: empty`, `Sec-Fetch-Mode: cors`, `Sec-Fetch-Site: same-origin`.
- [x] [P0-T3] Inspect current CK-12 modules (`src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_catalog.py`, `ck12_enrichment.py`) and list each spec mismatch
  - Acceptance: Notes enumerate mismatches and identify the exact functions to replace or retire.
  - Notes: 2026-01-11T05:31:00Z — ck12_catalog currently pulls a static S3 JSON (`DEFAULT_CK12_CATALOG_URL`) instead of the Browse API; headers are HTML-focused and lack the required JSON/CORS set; identifiers come from vanity URL slugs via `_extract_slug_from_url` rather than canonical `handle`; catalog parsing expects `books[*].Content_URL`/`Title` and never includes `artifactID`/`artifactType`/`handle` or `download_candidates`; `build_ck12_catalog`, `fetch_catalog_page`, `parse_catalog_json`, `_extract_slug_from_url` need replacement with Browse API fetch/parse + handle-based slug derivation. ck12_enrichment is HTML/PDF oriented: `fetch_flexbook_html`, `parse_flexbook_metadata`, `extract_pdf_url`, `_build_flexbook_url`, `enrich_entry_logic`, `enrich_ck12_catalog` all assume FlexBook HTML + PDF candidates and ignore Perma/Revision APIs; no browser-like JSON headers, no revision-ID extraction, and download candidates are PDFs instead of revision JSON—these functions need to be retired/replaced with Perma fetch + revision traversal and JSON candidates.
- [x] [P0-T4] Confirm dependency state for XHTML extraction by checking `pyproject.toml` for `beautifulsoup4`, `types-beautifulsoup4`, and `lxml`
  - Acceptance: Notes confirm whether `lxml` must be added.
- [x] [P0-T5] Read the feature template spec (`docs/features/templates/feature/spec.md`) and confirm this plan follows required structure/sections
  - Acceptance: Notes list any template-required sections that need updates in [`spec.md`](spec.md) or [`user-story.md`](user-story.md).
  - Notes: 2026-01-11T05:37:45Z — Template requires Overview, Behavior, Inputs/Outputs, API/CLI Surface, Data & State, Constraints & Risks, Definition of Done; `spec.md` includes all of them (no gaps). No template-driven changes needed for `user-story.md`.
- [x] [P0-T6] Re-read [`user-story.md`](user-story.md) and confirm each acceptance criterion is mapped to a phase in this plan
  - Acceptance: Notes link each acceptance criterion to Phase 1–6 tasks.
  - Notes: 2026-01-11T05:31:30Z — Catalog builder criterion → Phase 1 (CK-12 Browse API) with OpenStax catalog verified in Phase 8 dry run; Curation filter criterion → Phase 3; Manifest emission/content-type criterion → Phase 4; Download retrieval criterion → Phase 5 header injection validated by Phase 8 download run; CK-12 extract/normalize criterion → Phase 6 extraction plus Phase 8 normalization run.

**Phase 1 — CK-12 Catalog via Browse API (tests first)**
- [x] [P1-T1] Add unit test: Browse API JSON is parsed into catalog entries with `identifier` derived from canonical `handle` in [tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py](../../../../tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py)
  - Scenario gate: Given an item with `handle="CK-12-Physics-FlexBook-2.0"`, expect identifier `"ck-12-physics-flexbook-2-0"`.
  - Acceptance: Test fails against current implementation.
- [x] [P1-T2] Add unit test: catalog parser ignores/filters entries missing `artifactID`, `artifactType`, or `handle` in [tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py](../../../../tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py)
  - Scenario gate: Missing required fields → excluded from output deterministically.
- [x] [P1-T3] Add unit test: Browse fetch uses the `/flx/browse/flexbook?limit=200` endpoint and sends required browser-like headers in [tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py](../../../../tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py)
  - Scenario gate: request URL matches spec; headers include `User-Agent`, `Accept: application/json...`, `Referer`, `Origin`, `Sec-Fetch-*`.
- [x] [P1-T4] Implement Browse API fetch + parse in [src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_catalog.py](../../../../src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_catalog.py) to make [P1-T1..T3] pass
  - Acceptance: The tests from [P1-T1..T3] pass.
- [x] [P1-T5] Coverage gate: run `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py`
  - Acceptance: The run is green; changed lines are covered; overall repo coverage stays `>= 80%`.

**Phase 2 — CK-12 Enrichment via Perma API (tests first)**
- [x] [P2-T1] Add unit test: Perma API response yields section revision IDs as download candidates in [tests/lexile_scoring_model/pipeline_scripts/test_ck12_enrichment.py](../../../../tests/lexile_scoring_model/pipeline_scripts/test_ck12_enrichment.py)
  - Scenario gate: Nested `revisions[0].children[*].revisions[0].children[*].revisionID` values become candidates.
  - Acceptance: Test fails against current implementation.
- [x] [P2-T2] Add unit test: enrichment uses Perma API path `/flx/get/perma/<artifactType>/<handle>` and sends required headers in [tests/lexile_scoring_model/pipeline_scripts/test_ck12_enrichment.py](../../../../tests/lexile_scoring_model/pipeline_scripts/test_ck12_enrichment.py)
  - Scenario gate: request URL uses canonical `artifactType` + `handle` from catalog; headers match spec.
- [ ] [P2-T3] Add unit test: enrichment records zero candidates (and a skip reason surfaced to caller/CLI) when Perma JSON lacks revisions/children in [tests/lexile_scoring_model/pipeline_scripts/test_ck12_enrichment.py](../../../../tests/lexile_scoring_model/pipeline_scripts/test_ck12_enrichment.py)
  - Scenario gate: missing hierarchy → no candidates; processing continues.
- [ ] [P2-T4] Implement Perma fetch + revision-ID extraction in [src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_enrichment.py](../../../../src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_enrichment.py) to make [P2-T1..T3] pass
  - Acceptance: The tests from [P2-T1..T3] pass.
- [ ] [P2-T5] Coverage gate: run `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing tests/lexile_scoring_model/pipeline_scripts/test_ck12_enrichment.py`
  - Acceptance: The run is green; changed lines are covered; overall repo coverage stays `>= 80%`.

**Phase 3 — Curation: JSON Candidates + Revision Reachability (tests first)**
- [ ] [P3-T1] Add unit test: `has_text_candidate` remains correct for OpenStax entries in [tests/lexile_scoring_model/pipeline_scripts/test_oer_curation.py](../../../../tests/lexile_scoring_model/pipeline_scripts/test_oer_curation.py)
  - Scenario gate: one `text/plain` candidate → True; no text candidates → False.
- [ ] [P3-T2] Add unit test: new `has_json_candidate` (or equivalent) detects CK-12 revision JSON candidates in [tests/lexile_scoring_model/pipeline_scripts/test_oer_curation.py](../../../../tests/lexile_scoring_model/pipeline_scripts/test_oer_curation.py)
  - Scenario gate: candidate format `application/json` OR URL contains `/flx/get/detail/revision/` → True.
- [ ] [P3-T3] Add unit test: curation with `--require-json` keeps only CK-12 entries with revision JSON candidates in [tests/lexile_scoring_model/pipeline_scripts/test_oer_curation.py](../../../../tests/lexile_scoring_model/pipeline_scripts/test_oer_curation.py)
  - Scenario gate: missing/empty candidates → skipped with reason `"no json candidate"`.
- [ ] [P3-T4] Add unit test: revision reachability check treats non-200 responses as skip (HTTP mocked) in [tests/lexile_scoring_model/pipeline_scripts/test_oer_curation.py](../../../../tests/lexile_scoring_model/pipeline_scripts/test_oer_curation.py)
  - Scenario gate: mocked GET/HEAD returns 404 → item skipped with reason `"revision url unreachable"`.
- [ ] [P3-T5] Implement JSON-candidate detection + `--require-json` plumbing + revision reachability validation in [src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/oer_curation.py](../../../../src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/oer_curation.py)
  - Acceptance: [P3-T1..T4] pass.
- [ ] [P3-T6] Coverage gate: run `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing tests/lexile_scoring_model/pipeline_scripts/test_oer_curation.py`
  - Acceptance: The run is green; changed lines are covered; overall repo coverage stays `>= 80%`.

**Phase 4 — Manifest: `.json` for CK-12 + Source-Specific Validation (tests first)**
- [ ] [P4-T1] Add unit test: CK-12 manifest entries use `.json` filename extension and `url` is the revision-detail endpoint in [tests/lexile_scoring_model/pipeline_scripts/test_oer_manifest.py](../../../../tests/lexile_scoring_model/pipeline_scripts/test_oer_manifest.py)
  - Scenario gate: `source_id="ck12"` → `filename.endswith(".json")`.
- [ ] [P4-T2] Add unit test: OpenStax manifest entries remain `.txt` in [tests/lexile_scoring_model/pipeline_scripts/test_oer_manifest.py](../../../../tests/lexile_scoring_model/pipeline_scripts/test_oer_manifest.py)
  - Scenario gate: `source_id="openstax"` + `text/plain` → `.txt`.
- [ ] [P4-T3] Add unit test: CK-12 URL validation allows `application/json` and rejects `text/plain` when CK-12 is expected in [tests/lexile_scoring_model/pipeline_scripts/test_oer_manifest.py](../../../../tests/lexile_scoring_model/pipeline_scripts/test_oer_manifest.py)
  - Scenario gate: allowed content-type prefixes are source-specific.
- [ ] [P4-T4] Implement source-specific manifest building + validation in [src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/oer_manifest.py](../../../../src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/oer_manifest.py)
  - Acceptance: [P4-T1..T3] pass.
- [ ] [P4-T5] Coverage gate: run `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing tests/lexile_scoring_model/pipeline_scripts/test_oer_manifest.py`
  - Acceptance: The run is green; changed lines are covered; overall repo coverage stays `>= 80%`.

**Phase 5 — Downloader: CK-12 Headers (tests first)**
- [ ] [P5-T1] Add unit test: `_download_file` accepts optional headers and forwards them to `requests.get` (no `tmp_path`) in [tests/src/lexile_corpus_tuner/lexile_scoring_model/corpus/test_corpus_download_headers.py](../../../../tests/src/lexile_corpus_tuner/lexile_scoring_model/corpus/test_corpus_download_headers.py)
  - Scenario gate: when headers are provided, `requests.get(..., headers=headers)` is called.
  - Acceptance: Test uses monkeypatch + in-memory file objects; does not create temp files.
- [ ] [P5-T2] Add unit test: `download_oer_sources` uses CK-12 browser-like headers when `source_id == "ck12"` (no `tmp_path`) in [tests/src/lexile_corpus_tuner/lexile_scoring_model/corpus/test_corpus_download_headers.py](../../../../tests/src/lexile_corpus_tuner/lexile_scoring_model/corpus/test_corpus_download_headers.py)
  - Scenario gate: CK-12 entry triggers header injection; OpenStax does not.
  - Acceptance: Test uses monkeypatch + in-memory file objects; does not create temp files.
- [ ] [P5-T3] Implement optional-header support in [src/lexile_corpus_tuner/lexile_scoring_model/corpus/download.py](../../../../src/lexile_corpus_tuner/lexile_scoring_model/corpus/download.py)
  - Acceptance: [P5-T1] passes.
- [ ] [P5-T4] Implement CK-12 header injection in `download_oer_sources` in [src/lexile_corpus_tuner/lexile_scoring_model/corpus/download.py](../../../../src/lexile_corpus_tuner/lexile_scoring_model/corpus/download.py)
  - Acceptance: [P5-T2] passes.
- [ ] [P5-T5] Coverage gate: run `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing tests/src/lexile_corpus_tuner/lexile_scoring_model/corpus/test_corpus_download_headers.py`
  - Acceptance: The run is green; changed lines are covered; overall repo coverage stays `>= 80%`.

**Phase 6 — CK-12 JSON/XHTML/Text Extraction (tests first)**
- [ ] [P6-T1] Add `lxml` dependency to `pyproject.toml` if not already present
  - Acceptance: `poetry install` succeeds.
- [ ] [P6-T2] If Pyright reports missing stubs for `lxml`, add the minimal typing support required (e.g., `lxml-stubs`) and re-run Pyright
  - Acceptance: `poetry run pyright` is green without broad ignores.
- [ ] [P6-T3] Add a committed fixture CK-12 revision JSON containing `response.lesson.xhtml` in `tests/lexile_scoring_model/pipeline_scripts/fixtures/ck12_revision_with_xhtml.json`
  - Acceptance: Fixture is deterministic and contains enough XHTML to validate extraction order.
- [ ] [P6-T4] Add unit test: extractor reads JSON, chooses `xhtml` when present, and emits non-empty text in [tests/lexile_scoring_model/pipeline_scripts/test_extract_ck12_text.py](../../../../tests/lexile_scoring_model/pipeline_scripts/test_extract_ck12_text.py)
  - Scenario gate: `xhtml` present → used; output text contains expected words.
  - Acceptance: Test fails against current implementation.
- [ ] [P6-T5] Add unit test: extractor falls back to `xhtml_prime` when `xhtml` missing in [tests/lexile_scoring_model/pipeline_scripts/test_extract_ck12_text.py](../../../../tests/lexile_scoring_model/pipeline_scripts/test_extract_ck12_text.py)
  - Scenario gate: `xhtml` missing, `xhtml_prime` present → used.
- [ ] [P6-T6] Add unit test: extractor logs and skips when both XHTML fields missing in [tests/lexile_scoring_model/pipeline_scripts/test_extract_ck12_text.py](../../../../tests/lexile_scoring_model/pipeline_scripts/test_extract_ck12_text.py)
  - Scenario gate: missing fields → logged error; no output produced.
- [ ] [P6-T7] Implement `extract_ck12_text` CLI in [src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/extract_ck12_text.py](../../../../src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/extract_ck12_text.py)
  - Acceptance: [P6-T4..T6] pass; BeautifulSoup uses `"lxml"` parser per spec.
- [ ] [P6-T8] Coverage gate: run `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing tests/lexile_scoring_model/pipeline_scripts/test_extract_ck12_text.py`
  - Acceptance: The run is green; new module achieves `>= 90%` coverage.

**Phase 7 — Visual Curation UI Alignment (tests first)**
- [ ] [P7-T1] Add unit test: UI/export logic chooses `.json` for CK-12 and `.txt` for OpenStax in [tests/lexile_scoring_model/pipeline_scripts/test_oer_ui.py](../../../../tests/lexile_scoring_model/pipeline_scripts/test_oer_ui.py)
  - Scenario gate: mixed-source selection produces mixed extensions correctly.
- [ ] [P7-T2] Update UI logic in [src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/oer_ui.py](../../../../src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/oer_ui.py) to pass [P7-T1]
  - Acceptance: [P7-T1] passes.
- [ ] [P7-T3] Coverage gate: run `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing tests/lexile_scoring_model/pipeline_scripts/test_oer_ui.py`
  - Acceptance: The run is green; changed lines are covered.

**Phase 8 — Acceptance-Criteria Dry Run (CLI, manual)**
- [ ] [P8-T1] Run OpenStax catalog generation (`oer_catalog --sources "openstax"`) and confirm `data/meta/catalogs/openstax_catalog.jsonl` is created
  - Acceptance: File exists and is non-empty.
- [ ] [P8-T2] Run OpenStax enrichment (`oer_enrichment --sources "openstax"`) and confirm `data/meta/catalogs/openstax_enriched.jsonl` is created
  - Acceptance: File exists and is non-empty.
- [ ] [P8-T3] Run OER curation for OpenStax (`oer_curation --require-text --sources "openstax"`) and confirm `data/meta/catalogs/openstax_curated.jsonl` is created
  - Acceptance: File exists and is non-empty.
- [ ] [P8-T4] Run manifest generation for OpenStax (`oer_manifest --sources "openstax"` with URL validation off if supported) and confirm `data/meta/oer_sources.json` is created
  - Acceptance: Manifest contains at least one `source_id: "openstax"` entry.
- [ ] [P8-T5] Run CK-12 catalog generation (`ck12_catalog`) and confirm `data/meta/catalogs/ck12_catalog.jsonl` is created
  - Acceptance: File exists and is non-empty.
- [ ] [P8-T6] Run CK-12 enrichment (`ck12_enrichment`) and confirm `data/meta/catalogs/ck12_enriched.jsonl` is created
  - Acceptance: File exists and is non-empty.
- [ ] [P8-T7] Run OER curation for CK-12 (`oer_curation --require-json --sources "ck12"`) and confirm `data/meta/catalogs/ck12_curated.jsonl` is created
  - Acceptance: File exists and is non-empty.
- [ ] [P8-T8] Run manifest generation for CK-12 (`oer_manifest --sources "ck12"`) and confirm `data/meta/oer_sources.json` contains CK-12 entries
  - Acceptance: Manifest contains at least one `source_id: "ck12"` entry with `filename` ending in `.json`.
- [ ] [P8-T9] Run corpus download for OpenStax + CK-12 (`lexile-scoring-model-pipeline corpus download --sources "openstax,ck12"`)
  - Acceptance: CK-12 `.json` downloads succeed with required headers; OpenStax downloads still succeed.
- [ ] [P8-T10] Run CK-12 extraction (`extract_ck12_text`) on downloaded CK-12 JSON directory
  - Acceptance: Extracted `.txt` outputs are non-empty.
- [ ] [P8-T11] Run corpus normalize for OpenStax + CK-12 (`lexile-scoring-model-pipeline corpus normalize --sources "openstax,ck12"`)
  - Acceptance: Normalize completes without format errors.

**Phase 9 — Final QC Gate (all languages, full toolchain)**
- [ ] [P9-T1] Run JSON formatting: `poetry run python -m scripts.dev_tools.format_json`
  - Acceptance: Command exits 0.
- [ ] [P9-T2] Run JSON validation: `poetry run python -m scripts.dev_tools.validate_json`
  - Acceptance: Command exits 0.
- [ ] [P9-T3] Run shell formatter: `poetry run python -m scripts.dev_tools.shell_qc format`
  - Acceptance: Command exits 0.
- [ ] [P9-T4] Run shell lint/check: `poetry run python -m scripts.dev_tools.shell_qc check`
  - Acceptance: Command exits 0.
- [ ] [P9-T5] Run shell tests: `poetry run python -m scripts.dev_tools.shell_qc test`
  - Acceptance: Command exits 0.
- [ ] [P9-T6] Run Black: `poetry run black .`
  - Acceptance: Command exits 0 and produces no diffs.
- [ ] [P9-T7] Run Ruff: `poetry run ruff check`
  - Acceptance: Command exits 0.
- [ ] [P9-T8] Run Pyright: `poetry run pyright`
  - Acceptance: Command exits 0.
- [ ] [P9-T9] Run Pytest + coverage: `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`
  - Acceptance: Command exits 0 and repo coverage remains `>= 80%`.
- [ ] [P9-T10] Run PowerShell formatting: `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/format-powershell.ps1`
  - Acceptance: Command exits 0.
- [ ] [P9-T11] Run PowerShell lint: `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/run-psscriptanalyzer.ps1`
  - Acceptance: Command exits 0.
- [ ] [P9-T12] Run PowerShell tests: `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/run-pester.ps1`
  - Acceptance: Command exits 0.

## Notes / Risks

- Unit tests must not call external networks; all HTTP interactions are mocked.
- CK-12 endpoints/headers are brittle by nature; keep them centralized and covered by tests.
- Some existing tests in the repository use filesystem temp paths; when touching those tests, refactor them toward monkeypatching/in-memory I/O to align with the unit test policy.
