# 2026-01-08-ck-12-enrichment-pdf (Spec)

- Issue: #79
- Owner: 2026-01-08-ck-12-enrichment-pdf
- Date: 2026-01-09
- Status: Draft

## Context
CK-12 enrichment produces zero PDF download candidates because the FlexBook pages are client-rendered; the static HTML fetched by our scraper contains no PDF links, so enrichment never attaches PDFs.

Environment:
- OS/version: Debian 12 (devcontainer)
- Python version: 3.13.9 (poetry venv)
- Command/flags used: `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_enrichment --catalog-file data/meta/catalogs/ck12_catalog.jsonl --output data/meta/catalogs/ck12_enriched.jsonl`
- Data source or fixture: Live CK-12 FlexBook pages (e.g., ck-12-middle-school-math-concepts-grade-7)

Impact / Severity:
- [ ] Blocker
- [x] High
- [ ] Medium
- [ ] Low


## Repro & Evidence
Steps to Reproduce:
1. Ensure a CK-12 catalog exists (or reuse the default produced by the catalog step).
2. Run the enrichment CLI above against the catalog.
3. Inspect the enriched JSONL (or stdout) and note that no entries gain PDF download candidates.
4. Optional: Fetch a sample FlexBook page with browser headers; the returned HTML lacks any `.pdf` or `/pdf/` links.

Expected:
Enrichment should find a PDF download URL per FlexBook and append an `application/pdf` download candidate to each catalog entry so downstream normalization can pull the textbook.

Actual:
Enrichment completes without errors but adds zero PDF links. Manual HTTP fetch shows the page is a React shell with no static PDF anchors, so `extract_pdf_url` returns `None` for every entry.

Logs / Screenshots:
- [ ] Attached minimal logs or screenshot
- Snippet:

	- `requests.get("https://flexbooks.ck12.org/cbook/ck-12-middle-school-math-concepts-grade-7/", headers=REQUEST_HEADERS)` -> `Status 200 length 10729; pdf present? False; download/print/export matches: 0` (static HTML contains no PDF markers).
	- Without browser headers the same URL returns `403 ERROR` (CloudFront), so the scraper currently relies on headers but still cannot see the PDF link.
	- CK-12 Help article “Downloading PDFs for Your 2.0 FlexBooks” shows the UI flow: user clicks Choose → Download, fills a popup form, and receives a PDF link by email. This suggests PDF generation is asynchronous and likely gated behind an authenticated session.


## Scope & Non-Goals
- In scope:
	- Identify and implement a reliable way to resolve CK-12 FlexBook PDF download URLs.
	- Integrate the resolved PDF URL into enrichment so `DownloadCandidate` entries include `application/pdf` links.
	- Validate enrichment end-to-end against at least one known FlexBook slug.
	- Document the network/API contract used to obtain PDFs.
- Out of scope / non-goals:
	- Broader CK-12 site automation beyond PDF acquisition.
	- Styling or UX changes to downstream consumers.
	- Rewriting non-CK-12 pipelines.

## Root Cause Analysis
- CK-12 FlexBook pages are React-rendered; the static HTML fetched by `fetch_flexbook_html` contains no PDF anchors.
- The PDF download link is likely produced client-side via a JS API call (e.g., print/export endpoint) that our HTML-only parser never sees; `extract_pdf_url` only scans static anchors/buttons.
- Missing browser headers cause 403s, but even with headers the static DOM is insufficient to locate PDFs. Likely need a headless browser or direct CK-12 export API.
- Help article indicates PDF export may be an authenticated, asynchronous job (request via popup, fulfillment via emailed link). If so, anonymous scraping will never see a direct PDF URL.
- Relevant code: [src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_enrichment.py](src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_enrichment.py).


## Proposed Fix
- [ ] Unit coverage areas
- [ ] Integration scenario to retest
- [ ] Manual verification notes

- Preferred: Capture and call the CK-12 “Download/Print” HTTP endpoint directly (no DOM rendering) if it yields a stable PDF URL per FlexBook. Fallback: use Playwright/Selenium to click the download control and intercept the network request.
- Add a resolver module that, given a FlexBook slug, returns an absolute PDF URL (with headers/cookies if required). Keep the interface mockable for tests.
- Update enrichment to invoke the resolver and append an `application/pdf` `DownloadCandidate` when available.
- Persist any required auth/cookie headers in config (or perform an unauthenticated call if supported); avoid hardcoding credentials.
- Add logging at `info` when a PDF is resolved and `warning` when missing.
- Research tasks to unblock implementation:
	- Use Playwright/DevTools to record the network traffic when clicking “Choose → Download” on a FlexBook while logged in; identify the API endpoint, method, payload (slug/id), required cookies/CSRF, and response shape (immediate vs. async job id vs. email link).
	- Test whether the export endpoint works unauthenticated; if not, document login requirements and how to supply session cookies or tokens via config.
	- Determine whether the export returns a direct PDF URL synchronously or requires polling/job completion, and how long links remain valid.


## Assumptions, Constraints, Dependencies
- Assumptions (environment, data, access):
	- CK-12 provides an HTTP endpoint for PDF export; it may require authentication and may return an async job id with an emailed link.
	- FlexBook slugs remain stable and map 1:1 to export endpoints.
- Constraints (budget, performance, compatibility):
	- Avoid heavy browser automation unless necessary; prefer a single HTTP call per book to limit runtime.
	- Must remain compatible with existing Poetry/CI stack; new dependencies need approval.
- External dependencies (services, libraries, releases):
	- Potential new dependency on Playwright/Selenium if direct HTTP export is unavailable or requires an authenticated browser context.

## Data / API / Config Impact
- User-facing or API changes:
	- Enriched catalog entries will start including `application/pdf` download candidates for CK-12 books.
- Data or migration considerations:
	- Regenerate `ck12_enriched.jsonl` and downstream curated manifests to populate PDFs; handle any auth config needed for export.
- Logging/telemetry updates (if any):
	- Add resolver success/failure counts to enrichment logs for observability.

## Test Strategy
- [ ] Unit coverage areas
- [ ] Integration scenario to retest
- [ ] Manual verification notes

- Unit: resolver module validates URL construction and absolute HTTPS requirement; enrichment merges `DownloadCandidate` idempotently.
- Integration: run enrichment against one known FlexBook slug and assert a PDF candidate is present in the enriched JSONL output.
- Regression: re-run full CK-12 enrichment and confirm non-zero PDFs appear in curated outputs; optionally spot-check that normalization can download one PDF.


## Acceptance Criteria
- At least one known CK-12 FlexBook slug resolves to an `application/pdf` download candidate via enrichment.
- Enrichment logs a warning when a PDF cannot be resolved, without aborting other entries.
- The resolver returns absolute HTTPS URLs and rejects malformed/relative links.
- Curated CK-12 outputs show non-zero PDF download candidates after a full enrichment run.

## Risks & Mitigations
- Technical or operational risks:
	- CK-12 may throttle or block scripted access; endpoints may require dynamic tokens.
	- Playwright/Selenium increases CI runtime and dependency surface.
	- Export endpoint shape may change without notice.
- Mitigations and rollbacks:
	- Prefer direct HTTP export with stable headers; rate-limit requests and backoff on failures.
	- Guard browser-based resolver behind a feature flag/config to disable quickly.
	- Validate schema of responses and log anomalies for fast rollback.

## Rollout & Follow-up
- Release/rollout steps:
	- Implement resolver and enrichment wiring; land behind config if using browser automation.
	- Regenerate enriched catalog and curated manifests; verify PDFs appear.
	- Run full QC toolchain (Black, Ruff, Pyright, Pytest) and pipeline smoke test.
- Post-fix monitoring or clean-up tasks:
	- Monitor enrichment logs for PDF resolution failures; capture counts.
	- Periodically re-validate a small sample of FlexBooks to detect endpoint changes.
- Links: issue #79; related docs in this spec.
