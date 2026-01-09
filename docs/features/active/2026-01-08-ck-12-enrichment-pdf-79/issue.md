# ck-12-enrichment-pdf (Issue #79)

- Date captured: 2026-01-08
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/ck-12-enrichment-pdf/ (Issue #79)

> Automation note: Keep the section headings below unchanged; the promotion tooling maps each of them into the GitHub bug issue template.

- Issue: #79
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/79
- Last Updated: 2026-01-09
## Summary

CK-12 enrichment produces zero PDF download candidates because the FlexBook pages are client-rendered; the static HTML fetched by our scraper contains no PDF links, so enrichment never attaches PDFs.

## Environment

- OS/version: Debian 12 (devcontainer)
- Python version: 3.13.9 (poetry venv)
- Command/flags used: `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_enrichment --catalog-file data/meta/catalogs/ck12_catalog.jsonl --output data/meta/catalogs/ck12_enriched.jsonl`
- Data source or fixture: Live CK-12 FlexBook pages (e.g., ck-12-middle-school-math-concepts-grade-7)

## Steps to Reproduce

1. Ensure a CK-12 catalog exists (or reuse the default produced by the catalog step).
2. Run the enrichment CLI above against the catalog.
3. Inspect the enriched JSONL (or stdout) and note that no entries gain PDF download candidates.
4. Optional: Fetch a sample FlexBook page with browser headers; the returned HTML lacks any `.pdf` or `/pdf/` links.

## Expected Behavior

Enrichment should find a PDF download URL per FlexBook and append an `application/pdf` download candidate to each catalog entry so downstream normalization can pull the textbook.

## Actual Behavior

Enrichment completes without errors but adds zero PDF links. Manual HTTP fetch shows the page is a React shell with no static PDF anchors, so `extract_pdf_url` returns `None` for every entry.

## Logs / Screenshots

- [ ] Attached minimal logs or screenshot
- Snippet:

	- `requests.get("https://flexbooks.ck12.org/cbook/ck-12-middle-school-math-concepts-grade-7/", headers=REQUEST_HEADERS)` -> `Status 200 length 10729; pdf present? False; download/print/export matches: 0` (static HTML contains no PDF markers).
	- Without browser headers the same URL returns `403 ERROR` (CloudFront), so the scraper currently relies on headers but still cannot see the PDF link.

## Impact / Severity

- [ ] Blocker
- [x] High
- [ ] Medium
- [ ] Low

## Suspected Cause / Notes

- CK-12 FlexBook pages are React-rendered; the static HTML fetched by `fetch_flexbook_html` contains no PDF anchors.
- The PDF download link is likely produced client-side via a JS API call (e.g., print/export endpoint) that our HTML-only parser never sees; `extract_pdf_url` only scans static anchors/buttons.
- Missing browser headers cause 403s, but even with headers the static DOM is insufficient to locate PDFs. Likely need a headless browser or direct CK-12 export API.
- Relevant code: [src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_enrichment.py](src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_enrichment.py).

## Proposed Fix / Validation Ideas

- [ ] Unit coverage areas
- [ ] Integration scenario to retest
- [ ] Manual verification notes

- Add a headless browser (Playwright/Selenium) or a direct CK-12 export API call to resolve the PDF URL server-side; avoid brittle DOM scraping.
- Capture and document the network call used by the site’s “Download PDF” action; prefer a reproducible HTTP endpoint over full-page rendering when possible.
- Update enrichment to attach the resolved PDF URL to `DownloadCandidate` and add an integration test that validates at least one known FlexBook slug yields a PDF candidate.
- Regression check: re-run enrichment and confirm the curated manifest shows non-zero CK-12 PDFs; ensure downstream normalization can fetch the file.

## Next Step

- [ ] Promote to GitHub issue (bug-report template)
- [ ] Move to active fix folder / branch