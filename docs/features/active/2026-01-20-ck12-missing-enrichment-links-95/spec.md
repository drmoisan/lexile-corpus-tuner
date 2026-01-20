# 2026-01-20-ck12-missing-enrichment-links (Spec)

- **Issue:** #95
- **Parent (optional):** none
- **Owner:** drmoisan
- **Last Updated:** 2026-01-20T16-24
- **Status:** Draft
- **Version:** 0.1

## Context
When running the ck12 pipeline, 165 catalog entries are created. However, the enrichment process initially revealed 0 enriched entries. This was because there was no fallback for the content_url_slug in the case where it was NOT a static feed entry, there was NO raw handle, AND slug has a real value. This was fixed and now 133 catalog entries are created. There are 32 missing entries, likely due to improper identification of a content_url_slug in these cases.

Environment:
- OS/version: Debian GNU/Linux 12 (bookworm) (devcontainer)
- Python version: Not captured in logs (use `python --version` to confirm)
- Command/flags used:
	- `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_catalog --out-dir data/meta/catalogs`
	- `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_enrichment --catalog-file data/meta/catalogs/ck12_catalog.jsonl --output data/meta/catalogs/ck12_enriched.jsonl`
- Data source or fixture:
	- `https://static.ck12.org/testimonial/fbbrowse-prod.json`

Impact / Severity:
- [ ] Blocker
- [ ] High
- [x] Medium
- [ ] Low


## Repro & Evidence
Steps to Reproduce:
1. Run the CK-12 catalog step to generate `data/meta/catalogs/ck12_catalog.jsonl`.
2. Run the CK-12 enrichment step on that catalog output.
3. Observe enrichment logs and output count; 32 entries are skipped with `missing handle`.

Expected:
Enrichment should produce entries for all 165 catalog rows when `Content_URL` is present, with minimal `missing handle` skips. If the Perma API rejects a subset, those should be explicitly logged with non-handle-related reasons.

Actual:
Enrichment produced 133 entries and skipped 32 rows due to missing handles. Logs show repeated lines such as:

`Skipping <identifier> due to missing handle`

Logs / Screenshots:
- [x] Attached minimal logs or screenshot
- Snippet:
	- `Skipping ck-12-biology-workbook due to missing handle`
	- `Skipping ck-12-earth-science-for-high-school-teacher-s-edition due to missing handle`


## Scope & Non-Goals
- In scope:
	- Extend `extract_slug_from_content_url` to recognize CK-12 URL path prefixes observed in the browse feed so `content_url_slug` is populated.
	- When `Title` is present, ensure the catalog entry has a resolvable handle derived from `Content_URL` so enrichment can proceed.
	- Add unit tests for the new URL path patterns and missing/invalid inputs.
	- Add logging when a `Content_URL` is present but slug extraction still fails.
- Out of scope / non-goals:
	- Changing enrichment behavior beyond handle availability checks.
	- Adding new data sources outside the CK-12 browse feed.
	- Altering identifier logic beyond the slug extraction scope.
- Explicitly excluded systems, integrations, or datasets:
	- Perma API behavior changes.
	- OpenStax/OER catalog pipelines.

## Root Cause Analysis
- `extract_slug_from_content_url` only recognizes `/cbook/` and `/book/` paths, but the CK-12 feed includes `/tebook/`, `/workbook/`, and `/quizbook/` URLs. This causes `content_url_slug` to be `None`, and `entry.handle` remains `None` for those rows.
- See:
	- `ck12_catalog.py`: `extract_slug_from_content_url` (URL parsing logic)
	- `ck12_catalog.py`: `entry_handle` assignment (handle fallback logic)
	- `ck12_enrichment.py`: `if handle is None: ... continue`


## Proposed Fix
- Design summary (what changes where):
	- Expand `extract_slug_from_content_url` in `ck12_catalog.py` to recognize CK-12 `Content_URL` prefixes observed in the feed (`/tebook/`, `/workbook/`, `/quizbook/`) in addition to existing `/book/` and `/cbook/`.
	- Keep `entry_handle` selection unchanged except for consuming the expanded slug extraction, so handle derivation remains: `handle` → `handle_raw` → `content_url_slug` → `None`.
	- Add warning logging when `Content_URL` is present and slug extraction still fails (to surface future feed changes).
- Boundaries and invariants to preserve:
	- Do not change identifier derivation beyond slug extraction scope.
	- Do not modify enrichment logic, Perma API calls, or skip conditions.
	- Preserve existing behavior for non-CK-12 URLs and missing `Content_URL` values.
- Dependencies or blocked work:
	- None; relies only on existing standard-library URL parsing.

- Implementation strategy (what changes, not sequencing):
	- Files/modules to change:
		- `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_catalog.py`
		- `tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py` (or existing CK-12 test module)
	- Functions/classes/CLI commands impacted:
		- `extract_slug_from_content_url` (extend prefix support)
		- `parse_catalog_json` (logging when `Content_URL` present but slug missing)
	- Data flow and validation changes:
		- When `Content_URL` matches CK-12 host, extract slug for `/tebook/`, `/workbook/`, `/quizbook/` paths using the same normalization as `/book/` and `/cbook/`.
		- If `Title` is present but slug extraction returns `None`, emit a warning including `Identifier` or `Title` + URL for traceability.
	- Error handling and logging updates:
		- Add `logger.warning` (or project-standard logger) to record `Content_URL` parsing failures when `Title` exists; do not raise exceptions.
	- Rollback/feature-flag considerations (if applicable):
		- None; this is a safe, additive parsing change.

- Technical specifications (interfaces/contracts):
	- Inputs/outputs and formats:
		- Input: CK-12 `Content_URL` string from browse feed JSON.
		- Output: slug string (lowercase, hyphenated per existing normalization) or `None` when unsupported.
	- Required configuration keys and defaults:
		- None; no config changes.
	- Backward-compatibility expectations:
		- Existing `/book/` and `/cbook/` slugs remain unchanged.
	- Performance constraints (latency/throughput/memory):
		- Parsing remains O(1) per entry with no added network calls.


## Assumptions, Constraints, Dependencies
- Assumptions (environment, data, access):
	- CK-12 browse feed remains accessible at `https://static.ck12.org/testimonial/fbbrowse-prod.json`.
	- `Content_URL` continues to use `www.ck12.org` with path prefixes documented in the research note.
- Constraints (budget, performance, compatibility):
	- No network calls in unit tests (URL parsing only).
	- Keep parsing logic fast and deterministic; avoid external dependencies.
	- Maintain backward compatibility for existing `/book/` and `/cbook/` paths.
- External dependencies (services, libraries, releases):
	- CK-12 browse feed content.
	- Perma API used by enrichment (unchanged).

## Data / API / Config Impact
- User-facing or API changes:
	- None (internal catalog slug parsing only).
- Data or migration considerations:
	- Existing `ck12_catalog.jsonl` should be regenerated to populate missing handles.
- Logging/telemetry updates (if any):
	- Add a warning log when `Content_URL` is present but slug extraction fails so skipped rows can be audited.
- Compatibility notes (CLI flags, config schemas, versioning):
	- No CLI or config changes; backward compatible.

## Test Strategy
- [ ] Unit coverage areas
	- Add tests for `extract_slug_from_content_url` to cover `/tebook/`, `/workbook/`, and `/quizbook/` paths.
	- Add tests for invalid/missing `Content_URL` values to ensure `None` and warning logging when `Title` is present.
- [x] Integration scenario to retest
	- Rerun `ck12_catalog` → `ck12_enrichment` on the live CK-12 feed and verify enriched count reaches 165.
- [x] Manual verification notes
	- Confirm that previously skipped identifiers now have `handle` values in `ck12_catalog.jsonl`.
	- Validate that `missing handle` log lines approach zero; any residual skips must log a non-handle reason.


## Acceptance Criteria
- [ ] `extract_slug_from_content_url` returns a non-`None` slug for CK-12 `Content_URL` paths with `/tebook/`, `/workbook/`, and `/quizbook/` prefixes.
- [ ] When `Title` and `Content_URL` are present but slug extraction fails, a warning log is emitted with enough context to trace the source row.
- [ ] Rerunning `ck12_catalog` followed by `ck12_enrichment` yields exactly 165 enriched entries.
- [ ] Unit tests added and passing for the new CK-12 URL prefixes and missing/invalid `Content_URL` cases (list file path + test names).
- [ ] Existing `/book/` and `/cbook/` URL handling remains unchanged (coverage demonstrates no regression).
- [ ] No changes to enrichment skip conditions beyond handle availability checks.
- [ ] Full toolchain pass completed (format → lint → type-check → test).

## Risks & Mitigations
- Technical or operational risks:
	- CK-12 feed introduces new URL prefixes not covered by the parser, reintroducing missing handles.
	- Title present but URL malformed, leading to false expectations of handle availability.
- Mitigations and rollbacks:
	- Add warning logging for slug extraction failures and document required prefixes in the spec.
	- If new prefixes appear, update the parser and add tests; rerun catalog/enrichment.

## Rollout & Follow-up
- Release/rollout steps:
	- Run `ck12_catalog` to regenerate `ck12_catalog.jsonl`.
	- Run `ck12_enrichment` and verify enriched count is 165.
- Post-fix monitoring or clean-up tasks:
	- Review logs for any remaining `missing handle` warnings and capture example URLs for future parser updates.
- Links: issue, PRs, related docs
	- Issue: https://github.com/drmoisan/lexile-corpus-tuner/issues/95
	- Research: `docs/features/active/2026-01-20-ck12-missing-enrichment-links-95/20260120-ck12-missing-handle-research.md`
