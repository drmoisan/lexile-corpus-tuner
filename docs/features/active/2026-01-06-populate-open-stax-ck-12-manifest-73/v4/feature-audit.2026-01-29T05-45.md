# Feature Audit — populate-open-stax-ck-12-manifest-73 (v4)

## Scope and Baseline

- **Base branch:** `origin/development`
- **Feature branch:** `feature/populate-open-stax-ck-12-manifest-#73`
- **Evidence sources:**
  - `artifacts/pr_context.summary.txt`
  - `artifacts/pr_context.appendix.txt`
- **Feature folder:** `docs/features/active/2026-01-06-populate-open-stax-ck-12-manifest-73/v4`

## Acceptance Criteria Inventory (v4)

From `v4/spec.md` and `v4/user-story.md`:
1. Catalog builder writes IA search results for OpenStax and CK-12 native catalog rows to `data/meta/catalogs/*.jsonl` with fields: `source_id`, identifiers/slugs, `title`, `creator`, `year/last_modified` (if available), `language`, `license_url`, and `download_candidates` that include `_djvu.txt` (OpenStax) or reader HTML/JSON endpoints (CK-12).
2. Curation step filters catalog rows to entries with at least one text/HTML candidate; rows without retrievable content are skipped with a recorded reason.
3. Manifest generator emits `data/meta/oer_sources.json` entries containing `source_id`, stable slug `id`, direct `url`, and appropriate `filename` extensions (`.txt` for OpenStax text, `.html`/`.json` for CK-12 reader payloads); all URLs return HTTP 200 and the expected content-type (text/* for OpenStax, text/html or application/json for CK-12).
4. `lexile-scoring-model-pipeline corpus download --sources "openstax,ck12"` retrieves the manifest entries into the correct raw subfolders with the expected filenames.
5. A CK-12 extractor renders downloaded reader payloads to `.txt`; `lexile-scoring-model-pipeline corpus normalize --sources "openstax,ck12"` ingests the resulting `.txt` files without empty outputs or format errors.

## Acceptance Criteria Evaluation

| Criterion | Status | Evidence | Verification Command(s) | Notes |
|---|---|---|---|---|
| 1. Catalog builder outputs required fields and download candidates | PARTIAL | Unit tests executed: `tests/lexile_scoring_model/pipeline_scripts/test_oer_catalog.py`, `test_ck12_catalog.py` (pytest run). | `poetry run pytest tests/lexile_scoring_model/pipeline_scripts/test_oer_catalog.py tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py` | No end-to-end run against live IA/CK-12 data in this audit. |
| 2. Curation filters to entries with text/HTML candidates and records skips | PARTIAL | Unit test suite executed: `tests/lexile_scoring_model/pipeline_scripts/test_oer_curation.py`. | `poetry run pytest tests/lexile_scoring_model/pipeline_scripts/test_oer_curation.py` | Live curation output not validated in this audit. |
| 3. Manifest entries emit correct fields; URLs return HTTP 200 with expected content-type | UNVERIFIED | Unit tests run for manifest formatting: `test_oer_manifest.py` (pytest). | `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_manifest --catalog-dir data/meta/catalogs --out data/meta/oer_sources.json --validate-urls` | Requires network + catalog data; not run here. |
| 4. Corpus download retrieves manifest entries into correct raw subfolders | UNVERIFIED | CLI exists per spec; not executed during audit. | `poetry run lexile-scoring-model-pipeline corpus download --sources "openstax,ck12"` | Requires populated manifest and network access. |
| 5. CK-12 extractor renders to `.txt` and normalization ingests without errors | PARTIAL | Unit tests executed: `test_extract_ck12_text.py`, `test_oer_integration.py`. | `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.extract_ck12_text --source ck12 --input-dir data/corpus/raw/ck12 --output-dir data/corpus/raw/ck12` + `poetry run lexile-scoring-model-pipeline corpus normalize --sources "openstax,ck12"` | End-to-end pipeline not run in this audit. |

## Summary

**Overall Feature Readiness:** **NEEDS REVISION**

Key gaps preventing PASS:
- Shell test failures block the toolchain (see policy audit).
- Acceptance criteria that require live HTTP checks and end-to-end pipeline runs remain unverified.
- New module coverage below 90% for several core pipeline scripts (policy violation).

**Recommended follow-up verification:**
1. Generate catalogs for OpenStax and CK-12, run curation/manifest with `--validate-urls`.
2. Run corpus download and normalize for `openstax,ck12` with a controlled dataset.
3. Capture logs and artifacts to confirm HTTP status/content-type and output integrity.
