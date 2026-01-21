# 2026-01-21-curation-skips-valid-entries (Spec)

- **Issue:** #100
- **Parent (optional):** none
- **Owner:** drmoisan
- **Last Updated:** 2026-01-21T07-11
- **Status:** Draft
- **Version:** 0.1

## Context
CK-12 curation with `--require-json` is skipping valid entries because the catalog assigns artifact types (`flexbook`/`book`) that do not match Perma API types. This results in empty revision JSON candidates and `no json candidate` skips for entries that do have valid Perma endpoints.

Overview:
This spec aligns CK-12 catalog artifact type mapping with Perma API expectations so enrichment can emit valid revision candidates and curation keeps eligible entries. The fix is limited to catalog parsing and does not change CLI contracts or downstream curation rules.

Behavior:
Catalog parsing derives the Perma artifact type from `Content_URL` prefixes, enrichment uses that type to form Perma requests, and curation retains entries that have at least one JSON candidate when `--require-json` is supplied.

Environment:
- OS/version: Linux (dev container; Debian GNU/Linux 12)
- Python version: Not captured in session output.
- Command/flags used: `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_curation --catalog-dir data/meta/catalogs --require-json --sources "ck12" --out-dir data/meta/catalogs`
- Data source or fixture: `data/meta/catalogs/ck12_enriched.jsonl` generated from CK-12 Browse feed (`https://static.ck12.org/testimonial/fbbrowse-prod.json`) via `ck12_catalog` + `ck12_enrichment`.

Impact / Severity:
- [ ] Blocker
- [ ] High
- [x] Medium
- [ ] Low


## Repro & Evidence
Steps to Reproduce:
1. Run CK-12 catalog build: `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_catalog --out-dir data/meta/catalogs`.
2. Run CK-12 enrichment: `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_enrichment --catalog-file data/meta/catalogs/ck12_catalog.jsonl --output data/meta/catalogs/ck12_enriched.jsonl`.
3. Curate CK-12 with JSON requirement: `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_curation --catalog-dir data/meta/catalogs --require-json --sources "ck12" --out-dir data/meta/catalogs`.

Expected:
Valid CK-12 entries should retain revision JSON candidates and be kept during curation when `--require-json` is used.

Actual:
Nearly half of CK-12 entries are skipped with `no json candidate`. The run output is:
`Curated ck12_enriched.jsonl: 84 kept, 81 skipped -> data/meta/catalogs/ck12_curated.jsonl`.
The skip file `data/meta/catalogs/ck12_skips.jsonl` contains 81 entries, all with reason `no json candidate`.

Logs / Screenshots:
- [x] Attached minimal logs or screenshot
- Snippet:
	- `Curated ck12_enriched.jsonl: 84 kept, 81 skipped -> data/meta/catalogs/ck12_curated.jsonl`
	- `ck12_skips.jsonl` shows `{"identifier": "...", "reason": "no json candidate"}` for all 81 rows.


## Scope & Non-Goals
- In scope:
	- Map CK-12 URL prefixes to Perma artifact types when building the catalog (`cbook`, `tebook`, `workbook`, `quizbook`, `book`).
	- Keep existing catalog/enrichment/curation CLI contracts unchanged.
	- Reduce `no json candidate` skips by ensuring valid Perma revisions are discoverable.
- Out of scope / non-goals:
	- Changing CK-12 downloader, revision extraction, or curation business rules.
	- Retrying Perma requests or adding new backoff logic.
	- Altering OpenStax or Gutenberg pipelines.
- Explicitly excluded systems, integrations, or datasets:
	- Internet Archive content discovery or downloads.
	- Any non-CK-12 sources.

## Root Cause Analysis
- `ck12_catalog.py` assigns `artifact_type` as `flexbook` when `content_url_host == "flexbooks.ck12.org"` and `book` when `content_url_host == "www.ck12.org"`. The Perma API does not accept `flexbook` and expects `cbook`, `tebook`, `workbook`, or `quizbook` depending on the URL prefix.
- Perma validation examples:
	- `https://www.ck12.org/flx/get/perma/flexbook/cbse-biology-class-10` -> `Unknown artifact type[flexbook]`.
	- `https://www.ck12.org/flx/get/perma/cbook/cbse-biology-class-10` -> valid response with `revisions`.
	- `https://www.ck12.org/flx/get/perma/quizbook/CK-12-Biology-Quizzes-and-Tests` -> valid response with `revisions`.


## Proposed Fix

### Design summary (what changes where):
Update `ck12_catalog.py` to correctly identify the `artifact_type` of each entry by parsing its `Content_URL` path segments.
1. Return type of `extract_slug_from_content_url` changes from `str | None` to `tuple[str, str] | None` (returning `(slug, artifact_type)`).
2. `parse_catalog_json` updates to unpack this tuple and use the extracted `artifact_type` instead of deriving it loosely from the hostname.

### Boundaries and invariants to preserve:
- `Content_URL` remains the source of truth.
- Slugs validation remains unchanged (must be non-empty strings).
- If the URL prefix is unknown, the entry is skipped/warned as before (no default to "book" or "flexbook" if the URL pattern is unrecognized).
- Warning logs for unparsable URLs are preserved.

### Dependencies or blocked work:
- None.

### Implementation strategy (what changes, not sequencing):
	
#### Files/modules to change:
- `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_catalog.py`

#### Functions/classes/CLI commands impacted:
- **`extract_slug_from_content_url(url: str) -> tuple[str, str] | None`**:
    - **Current Logic**: Returns slug string or None.
    - **New Logic**: 
        - Parse URL. Split path by `/`.
        - If host is `flexbooks.ck12.org` and path contains `cbook`: Return `attained_slug, "cbook"`.
        - If host is `www.ck12.org` and path contains `book`, `tebook`, `workbook`, or `quizbook`: Return `attained_slug, found_prefix`.
        - Otherwise return `None`.
- **`parse_catalog_json`**:
    - Update call site: `result = extract_slug_from_content_url(...)`.
    - If result is not None: `slug, perma_type = result`.
    - priority logic for `artifact_type_value`: Use `perma_type` if available. Fallback to `flexbook`/`book` based on host only if URL parsing failed (though if URL parsing fails, we likely don't have a slug either, so this effectively replaces the logic).

#### Data flow and validation changes:
- `Content_URL` -> `extract_slug...` -> `(slug, artifact_type)` -> `CatalogEntry.artifact_type`.
- This ensures `cbook`, `tebook`, etc. flow into the JSONL file.

#### Error handling and logging updates:
- Existing warning: `logger.warning("Content_URL slug missing...")` logic remains, but checks if the *tuple* is None.

#### Rollback/feature-flag considerations (if applicable):
- None; safe to roll forward.

### Technical specifications (interfaces/contracts):

#### Inputs/outputs and formats:
- `extract_slug_from_content_url` input: `https://www.ck12.org/tebook/Biology-Teachers-Edition/`
- `extract_slug_from_content_url` output: `("Biology-Teachers-Edition", "tebook")`

#### Required configuration keys and defaults:
- None.

#### Backward-compatibility expectations:
- Internal function signature change (`extract_slug_from_content_url`). 
- CLI output format (JSONL) is compatible, just with new string values in `artifact_type` field.

#### Performance constraints (latency/throughput/memory):
- Negligible.

## Assumptions, Constraints, Dependencies
- Assumptions (environment, data, access):
	- CK-12 Perma API remains accessible from CI/dev environments.
	- CK-12 Browse feed continues to include `Content_URL` with stable prefixes.
- Constraints (budget, performance, compatibility):
	- No new dependencies; keep request volume unchanged.
	- Preserve existing CLI flags and output file names.
- External dependencies (services, libraries, releases):
	- CK-12 Perma API (`/flx/get/perma/<type>/<handle>`).

## Data / API / Config Impact
- User-facing or API changes:
	- None (internal mapping only).
- Inputs:
	- CK-12 browse feed JSON (`https://static.ck12.org/testimonial/fbbrowse-prod.json`) ingested by `ck12_catalog`.
	- `ck12_catalog.jsonl` consumed by `ck12_enrichment`.
	- `ck12_enriched.jsonl` consumed by `oer_curation` when `--require-json` is set.
- Outputs:
	- `ck12_catalog.jsonl`, `ck12_enriched.jsonl`, `ck12_curated.jsonl`, `ck12_skips.jsonl` (unchanged file names and locations).
- API / CLI surface:
	- No new flags; existing commands remain `ck12_catalog`, `ck12_enrichment`, and `oer_curation`.
- Data or migration considerations:
	- Newly generated `ck12_catalog.jsonl` and `ck12_enriched.jsonl` will contain updated `artifact_type` values.
- Data & state:
	- Mapping is derived from `Content_URL` at catalog build time; no new persistent state or caching is introduced.
- Logging/telemetry updates (if any):
	- None required beyond existing warnings.
- Compatibility notes (CLI flags, config schemas, versioning):
	- No CLI changes; same commands and outputs.

## Test Strategy
- [x] Unit coverage areas
	- Add/extend tests in `tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py` to validate the mapping rules:
		- `test_extract_slug_from_content_url_supports_cbook_artifact_type`
		- `test_extract_slug_from_content_url_supports_tebook_artifact_type`
		- `test_extract_slug_from_content_url_supports_workbook_artifact_type`
		- `test_extract_slug_from_content_url_supports_quizbook_artifact_type`
		- `test_extract_slug_from_content_url_supports_book_artifact_type`
		- `test_extract_slug_from_content_url_unknown_prefix_returns_none_type`
- [x] Integration scenario to retest
	- Re-run CK-12 catalog + enrichment + curation with `--require-json`; validate that `no json candidate` drops materially and newly mapped entries are kept.
- [x] Manual verification notes
	- Verified Perma success for a flexbook-derived entry by mapping to `cbook`: `cbse-biology-class-10` returns a valid Perma payload with `revisions`.
	- Verified `tebook`, `workbook`, and `quizbook` endpoints return valid payloads for representative handles.


## Acceptance Criteria
- [ ] Repro steps now keep CK-12 entries with `/cbook/`, `/tebook/`, `/workbook/`, and `/quizbook/` URLs when `--require-json` is used.
- [ ] Regression tests added and passing in `tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py`:
	- `test_extract_slug_from_content_url_supports_cbook_artifact_type`
	- `test_extract_slug_from_content_url_supports_tebook_artifact_type`
	- `test_extract_slug_from_content_url_supports_workbook_artifact_type`
	- `test_extract_slug_from_content_url_supports_quizbook_artifact_type`
	- `test_extract_slug_from_content_url_supports_book_artifact_type`
	- `test_extract_slug_from_content_url_unknown_prefix_returns_none_type`
- [ ] Entries with unsupported or unknown URL prefixes are skipped with the existing warning and are not misclassified as `book`.
- [ ] No unintended behavior changes outside CK-12 catalog/enrichment/curation.
- [ ] Performance is unchanged (same number of Perma calls).
- [ ] Full toolchain pass completed (format → lint → type-check → test).
- [ ] Docs/config references remain accurate (no CLI changes).
- [ ] Definition of Done: `ck12_enriched.jsonl` contains at least one revision candidate for entries mapped from `/cbook/`, `/tebook/`, `/workbook/`, and `/quizbook/` URLs in the same run used for curation.


## Risks & Mitigations
- Technical or operational risks:
	- CK-12 API may change artifact types or URL patterns, causing future mismatches.
	- Perma access may be blocked from certain environments (403s).
- Mitigations and rollbacks:
	- Keep mapping isolated to catalog parsing so rollback is localized.
	- Preserve warnings when URL parsing fails to surface future feed changes.

## Rollout & Follow-up
- Release/rollout steps:
	- Regenerate CK-12 catalog, enriched, and curated artifacts using existing commands.
- Post-fix monitoring or clean-up tasks:
	- Inspect `ck12_skips.jsonl` to confirm `no json candidate` drops and validate any remaining skips.
- Links: issue, PRs, related docs
	- Issue #100: https://github.com/drmoisan/lexile-corpus-tuner/issues/100
	- Research: `artifacts/research/20260121-ck12-curation-skips-research.md`