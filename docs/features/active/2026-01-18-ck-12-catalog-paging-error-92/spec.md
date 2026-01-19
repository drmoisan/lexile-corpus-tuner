# 2026-01-18-ck-12-catalog-paging-error (Spec)

- **Issue:** #92
- **Owner:** drmoisan
- **Date:** 2026-01-18
- **Status:** Completed
- **Outcome:** CK-12 catalog ingestion now uses the static FlexBook browse feed by default, derives deterministic identifiers across `/cbook/`, `/user:<handle>/cbook/`, and `/book/` URL patterns, and ships with tests that cover static-feed parsing, dedupe behavior, and fallback logic.
- **Root Cause:** The public Browse API endpoint ignores `limit`/`offset` inputs and always returns the first 10 items, while the implementation relies on that endpoint without any alternative source.


## Context
The CK-12 catalog download only returns 10 items even though the catalog contains 200+ entries. The resulting catalog JSON is incomplete because paging is not followed or not supported by the current fetch logic.

Environment:
- OS/version: Linux (Debian 12 devcontainer)
- Python version: Unknown (likely 3.10+; exact version not captured)
- Command/flags used: `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_catalog --out-dir data/meta/catalogs`
- Data source or fixture: CK-12 public catalog endpoint (JSON)

Impact / Severity:
- [ ] Blocker
- [ ] High
- [x] Medium
- [ ] Low


## Repro & Evidence
Steps to Reproduce:
1. Run the CK-12 catalog downloader to produce a JSONL catalog file.
2. Inspect the output count (e.g., number of JSONL lines).
3. Compare against the CK-12 catalog total (200+ items) visible on the source site.

Expected:
The downloader retrieves all CK-12 catalog items across all pages, producing a JSONL file with 200+ entries.

Actual:
Only the first page is returned, resulting in exactly 10 items in the output JSONL. No error is raised, but the catalog is incomplete.

Logs / Screenshots:
- [ ] Attached minimal logs or screenshot
- Snippet: Output JSONL contains 10 items; no pagination indicators are processed.


## Scope & Non-Goals
- In scope:
	- Switch the default CK-12 catalog source to the FlexBook browse static JSON feed.
	- Extend parsing logic to accept the static feed schema and derive stable identifiers from `Content_URL`.
	- Support community-contributed `Content_URL` patterns that include `/user:<handle>/cbook/<slug>/...`.
	- Preserve the existing `catalog_url` override for manual or future endpoints.
	- Update unit tests for `ck12_catalog.py` to cover the static feed and the new default URL.
- Out of scope / non-goals:
	- Scraping CK-12 HTML pages or section content.
	- Bypassing access restrictions or simulating authenticated paging.
	- Downloading or transforming CK-12 book content.
	- Any AI/ML training usage of CK-12 materials (explicitly prohibited by CK-12 terms).

## Root Cause Analysis
The CK-12 Browse API returns a `response` payload that advertises paging fields, but observed requests ignore the provided `limit` and `offset` values and always return 10 items. The current downloader only hits that endpoint once, so even with correct pagination logic the endpoint itself blocks access to the full catalog.


-## Proposed Fix
- [x] Update `DEFAULT_CK12_CATALOG_URL` to the static FlexBook browse feed (`https://static.ck12.org/testimonial/fbbrowse-prod.json`).
- [x] Extend `parse_catalog_json` to support the static feed schema with explicit mapping rules:
	- Accept `books` list objects that include `Title`, `Content_URL`, `Language`, `Language_Code`, `Subject`, `General`, `Grade`, `Flexbook_2dot0`, `Community_Contributed`, `FB`, `Standard`, `Thumb_URL`, `Enable`, and `Remarks`.
	- Derive `identifier` from `Content_URL` when it matches either:
		- `https://flexbooks.ck12.org/cbook/<slug>/...` -> `<slug>`
		- `https://flexbooks.ck12.org/user:<handle>/cbook/<slug>/...` -> `<slug>`
		- `https://www.ck12.org/book/<slug>/...` -> `<slug>`
	- If `Content_URL` is missing or unparseable, fall back to `handle`, then to slugified `Title` (in that order).
	- Set `artifact_type` by `Content_URL` host (`flexbooks.ck12.org` => `flexbook`, `www.ck12.org` => `book`). If no URL, preserve existing `artifactType`/`artifact_type` when present, otherwise default to `"flexbook"`.
	- Allow `artifact_id` to be `None` for static feed entries without IDs; do not drop rows solely for missing IDs.
	- Populate `language` from `Language` (string) or `Language_Code` to keep backwards compatibility.
	- Deduplicate entries by `identifier` while preserving first-seen ordering from the feed.
	- Treat non-list `books` values as a `ValueError` to keep parser behavior explicit.
- [x] Keep Browse API parsing support intact for explicit `--catalog-url` usage; do not remove `response.flexbook` / `response.items` parsing.
- [x] Update unit tests in `tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py` to cover static-feed parsing, dedupe, and fallback logic.


## Assumptions, Constraints, Dependencies
- Assumptions (environment, data, access):
	- The static FlexBook browse feed remains publicly accessible without authentication.
	- CK-12 allows metadata retrieval for catalog enumeration under existing usage permissions.
- Constraints (budget, performance, compatibility):
	- No new dependencies; continue using `requests` and the current CLI interface.
	- Unit tests must avoid network calls and filesystem temp files.
	- Catalog ingestion must respect CK-12 Terms of Use and licensing restrictions.
- Data invariants (CatalogEntry contract):
	- `identifier` must be a non-empty slug (validated by `generate_stable_slug`).
	- `language` must be a list of strings (even when a single language is present).
	- `artifact_type`, `handle`, and `artifact_id` may be `None` without violating the model contract.
- External dependencies (services, libraries, releases):
	- CK-12 static JSON feed (`static.ck12.org/testimonial/fbbrowse-prod.json`).

## Data / API / Config Impact
- User-facing or API changes:
	- Default catalog URL changes to the static feed; `--catalog-url` remains as an override for alternate endpoints.
- Data or migration considerations:
	- Output `ck12_catalog.jsonl` may include more entries and no longer rely on Browse API fields like `artifactID`.
- Logging/telemetry updates (if any):
	- None required; existing CLI output should remain unchanged.

## Test Strategy
- [x] Unit coverage areas (single test file for `ck12_catalog.py`):
	- Update `test_fetch_catalog_page_targets_browse_api_with_required_headers` to expect the new default static feed URL and headers, while still validating browser-like headers.
	- Add `test_parse_catalog_json_accepts_static_feed_books` with a static-feed `books` list containing two entries:
		- one `flexbooks.ck12.org` URL (expects `artifact_type="flexbook"`, identifier from `/cbook/<slug>/`)
		- one `www.ck12.org/book/` URL (expects `artifact_type="book"`, identifier from `/book/<slug>/`)
	- Add `test_parse_catalog_json_accepts_static_feed_user_cbook` with a `Content_URL` that includes `/user:<handle>/cbook/<slug>/...`; expect the identifier to use `<slug>` and `artifact_type="flexbook"`.
	- Add `test_parse_catalog_json_static_feed_dedupes_by_identifier` with duplicate `Content_URL` slugs across multiple `books` entries; assert length is 1 and first entry preserved.
	- Add `test_parse_catalog_json_static_feed_missing_content_url_falls_back` with no `Content_URL` and a `handle` field; assert identifier derives from `handle` and `artifact_type` falls back to existing `artifactType` or default.
	- Add `test_parse_catalog_json_static_feed_title_slug_fallback` with no `Content_URL` or `handle`; assert identifier is slugified `Title`.
	- Keep network calls mocked via `monkeypatch`; avoid filesystem writes and temporary files.
- [ ] Integration scenario to retest: run `ck12_catalog` in a controlled environment and verify output count is greater than 10 when the static feed includes more than 10 unique entries.
- [ ] Manual verification notes: spot-check a few CK-12 titles that appear beyond the first 10 in the static feed and confirm identifiers map to the expected slugs.

## Final Outcomes and Deviations
- Implementation matches the proposed fixes: static feed is the default catalog source, slug extraction covers `/cbook/`, `/user:<handle>/cbook/`, and `/book/`, artifact type mapping is enforced, and unit tests cover static-feed parsing, dedupe, and fallbacks.
- No deviations from the planned scope were introduced; integration run and manual spot-checks remain outstanding and tracked as follow-up items.


## Acceptance Criteria
- `DEFAULT_CK12_CATALOG_URL` equals `https://static.ck12.org/testimonial/fbbrowse-prod.json` and is used by `build_ck12_catalog` when `--catalog-url` is not supplied.
- Given a static-feed payload with two `books` entries (one `flexbooks.ck12.org/cbook/<slug>/...`, one `www.ck12.org/book/<slug>/...`), `parse_catalog_json` returns exactly two `CatalogEntry` objects with:
	- identifiers `<slug>` extracted from the URL path,
	- `artifact_type` values `flexbook` and `book` respectively,
	- `language` populated from `Language` or `Language_Code` when present.
- Given a static-feed payload with a `Content_URL` in the form `https://flexbooks.ck12.org/user:<handle>/cbook/<slug>/...`, the identifier is `<slug>` and `artifact_type` is `flexbook`.
- Given a static-feed payload with duplicate slugs (same `Content_URL` slug), `parse_catalog_json` returns a single entry and preserves the first entry’s `title`.
- Given a static-feed payload missing `Content_URL`, the parser falls back to `handle`; if `handle` is missing, it falls back to slugified `Title`.
- Unit tests for `ck12_catalog.py` validate the static-feed parsing, dedupe, and fallback behaviors using only in-memory fixtures and mocked HTTP calls.

## Risks & Mitigations
- Technical or operational risks:
	- CK-12 may change or remove the static feed URL.
	- Licensing/terms restrict automated access even for metadata.
- Mitigations and rollbacks:
	- Keep `--catalog-url` override to allow switching to a different source without code changes.
	- Document usage constraints and require confirmation of permitted access before running in production workflows.

## Rollout & Follow-up
- Release/rollout steps:
	- Merge fix and rerun the CK-12 catalog build in the standard pipeline flow.
- Post-fix monitoring or clean-up tasks:
	- Verify output entry counts and spot-check identifiers against known CK-12 titles.
	- Track CK-12 feed stability and update the catalog URL if the feed changes.
- Links: issue #92, research notes `20260117-ck12-catalog-limit-10-research.md`
