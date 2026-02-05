---
title: "2026-02-04-dropped-links - Spec"
issue: "115"
parent: "73"
owner: "drmoisan"
last_updated: "2026-02-04T15-18"
status: "Active"
status_color: "blue"
version: "1.0"
---

# 2026-02-04-dropped-links (Spec)

- **Issue:** #115
- **Parent (optional):** none
- **Owner:** drmoisan
- **Last Updated:** 2026-02-04T15-18
- **Status:** Active
- **Version:** 1.0

## Context
CK-12 pipeline step 4 (`oer_manifest --validate-urls`) fails URL validation for all CK-12 entries. The manifest generator writes zero entries because every HEAD request is blocked by the CK-12 CDN.

Environment:
- OS/version: Linux (Debian GNU/Linux 12, dev container)
- Python version: 3.13.9
- Command/flags used: `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_manifest --catalog-dir data/meta/catalogs --out data/meta/oer_sources.json --validate-urls`
- Data source or fixture: `data/meta/catalogs/ck12_curated.jsonl`

Impact / Severity:
- [ ] Blocker
- [x] High
- [ ] Medium
- [ ] Low


## Repro & Evidence
Steps to Reproduce:
1. Run the CK-12 catalog/enrichment/curation steps to generate `ck12_curated.jsonl`.
2. Run `oer_manifest` with `--validate-urls` pointing at `data/meta/catalogs`.
3. Observe the validation failure messages and the resulting manifest size.

Expected:
CK-12 entries validate successfully and the manifest contains the expected CK-12 items with JSON URLs.

Actual:
All 155 CK-12 entries fail validation with `status=None, content_type=None`, causing a zero-entry manifest. Validation requests return HTTP 403 from the CK-12 CDN.

Logs / Screenshots:
- [ ] Attached minimal logs or screenshot
- Snippet: `validation failed (status=None, content_type=None)`


## Scope & Non-Goals
- In scope:
- Add a browser-like User-Agent header to CK-12 URL validation requests.
- Preserve existing CK-12 candidate selection and content-type validation rules.
- Validate CK-12 revision endpoints without adding new dependencies.
- Out of scope / non-goals:
- Changing CK-12 catalog/enrichment/curation logic or data sources.
- Switching to GET requests or adding third-party HTTP clients.
- Explicitly excluded systems, integrations, or datasets:
- Non-CK-12 OER sources and downstream corpus normalization steps.

## Root Cause Analysis
The CK-12 CloudFront CDN blocks requests without a browser-like User-Agent. `validate_url()` in `oer_manifest.py` sends HEAD requests without a User-Agent header, so all requests are rejected and surfaced as `(False, None, None)`.


## Proposed Fix

### Design summary (what changes where):
Add a browser-like User-Agent header to `urllib.request.Request` in `validate_url()` within `oer_manifest.py` so CK-12 CloudFront accepts HEAD requests.

### Boundaries and invariants to preserve:
- Keep the HEAD-based validation flow and current content-type prefix checks.
- Maintain CK-12 candidate selection rules and filename conventions.

### Dependencies or blocked work:
- None. Use existing standard library `urllib.request`.

### Implementation strategy (what changes, not sequencing):
	
#### Files/modules to change:
- `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/oer_manifest.py`

#### Functions/classes/CLI commands impacted:
- `validate_url()`
- `generate_manifest()` (behavioral impact when `validate_urls=True`)
- CLI command: `python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_manifest`

#### Data flow and validation changes:
- Add a User-Agent header to the HEAD request; keep existing HTTP status and Content-Type validation logic unchanged.

#### Error handling and logging updates:
- No new logging. Continue emitting validation failures via `typer.echo` in `generate_manifest()`.

#### Rollback/feature-flag considerations (if applicable):
- Not applicable; change is a header addition only.

### Technical specifications (interfaces/contracts):

#### Inputs/outputs and formats:
- Inputs: CK-12 revision URLs (`https://www.ck12.org/flx/get/detail/revision/{id}?tiny=true`).
- Outputs: Manifest entries with `.json` filenames and `application/json` content-type validation.

#### Required configuration keys and defaults:
- None.

#### Backward-compatibility expectations:
- No changes to CLI flags or manifest schema. Non-CK-12 sources continue to validate as before.

#### Performance constraints (latency/throughput/memory):
- No change in request count or behavior; negligible overhead for header addition.

## Assumptions, Constraints, Dependencies
- Assumptions (environment, data, access):
- CK-12 revision endpoints are reachable and return JSON when a browser-like User-Agent is provided.
- Constraints (budget, performance, compatibility):
- Must remain compatible with existing `urllib.request` usage and HEAD-based validation.
- External dependencies (services, libraries, releases):
- CK-12 CloudFront CDN availability and behavior.

## Data / API / Config Impact
- User-facing or API changes:
- None; behavior changes only in CK-12 URL validation with User-Agent.
- Data or migration considerations:
- None.
- Logging/telemetry updates (if any):
- None.
- Compatibility notes (CLI flags, config schemas, versioning):
- No changes to CLI or schema.

## Test Strategy
Seeded from issue:

- [x] Unit coverage areas: `tests/lexile_scoring_model/pipeline_scripts/test_oer_manifest.py` (validate_url + CK-12 acceptance).
- [x] Integration scenario to retest: rerun `oer_manifest --validate-urls` and confirm CK-12 entries are written.
- [x] Manual verification notes: add User-Agent header in `validate_url()` and verify CK-12 URLs return HTTP 200 + `application/json`.

- Regression tests to add or update:
- Consider adding a unit test that asserts the `User-Agent` header is set on the request when validating CK-12 URLs (monkeypatch `urllib.request.Request`).
- Unit tests (pytest) for the fixed behavior and boundaries:
- Update `tests/lexile_scoring_model/pipeline_scripts/test_oer_manifest.py` to validate request headers when calling `validate_url()`.
- The HEAD-method check is an invariant; validate it via a standard passing unit test (`test_validate_url_still_uses_head_method`).
- Edge cases and negative scenarios (invalid inputs, missing data, boundary values):
- Ensure non-200 responses still fail validation and content-type mismatches are rejected.
- Error handling and logging verification:
- Confirm validation failures still emit `typer.echo` messages with status/content-type details.
- Coverage impact and targets for changed lines/modules:
- Maintain existing coverage; new test should cover header inclusion.
- Toolchain commands to run (format → lint → type-check → test):
- `poetry run black .`
- `poetry run ruff check`
- `poetry run pyright`
- `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`
- Manual validation steps (if required):
- Re-run `oer_manifest --validate-urls` and verify non-zero CK-12 entries.


## Acceptance Criteria
- [x] Repro steps now produce the expected behavior in all documented environments.
- [x] Regression test(s) added and passing (list file path and test name).
- [x] Edge cases and invalid inputs are handled with correct errors or fallbacks.
- [x] No unintended behavior changes outside the defined scope.
- [ ] Required logs/telemetry updated and validated (if applicable).
- [ ] Performance constraints met or explicitly waived with rationale.
- [x] Full toolchain pass completed (format → lint → type-check → test).
- [ ] Docs/config references updated to match the new behavior.

## Risks & Mitigations
- Technical or operational risks:
- CK-12 CDN could alter User-Agent requirements or block HEAD requests.
- Mitigations and rollbacks:
- If validation fails, allow temporarily disabling `--validate-urls` or revisit header strategy.

## Rollout & Follow-up
- Release/rollout steps:
- Update the manifest via `oer_manifest --validate-urls` after deployment.
- Post-fix monitoring or clean-up tasks:
- Spot-check CK-12 entries in `data/meta/oer_sources.json` for expected JSON URLs.
- Links: issue, PRs, related docs
