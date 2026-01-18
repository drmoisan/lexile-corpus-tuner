# ck-12-catalog-paging-error (Issue #92)

- Date captured: 2026-01-18
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/ck-12-catalog-paging-error/ (Issue #92)

> Automation note: Keep the section headings below unchanged; the promotion tooling maps each of them into the GitHub bug issue template.

- Issue: #92
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/92
- Last Updated: 2026-01-18
## Summary

The CK-12 catalog download only returns 10 items even though the catalog contains 200+ entries. The resulting catalog JSON is incomplete because paging is not followed or not supported by the current fetch logic.

## Environment

- OS/version: Linux (Debian 12 devcontainer)
- Python version: Unknown (likely 3.10+; exact version not captured)
- Command/flags used: `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_catalog --out-dir data/meta/catalogs`
- Data source or fixture: CK-12 public catalog endpoint (JSON)

## Steps to Reproduce

1. Run the CK-12 catalog downloader to produce a JSONL catalog file.
2. Inspect the output count (e.g., number of JSONL lines).
3. Compare against the CK-12 catalog total (200+ items) visible on the source site.

## Expected Behavior

The downloader retrieves all CK-12 catalog items across all pages, producing a JSONL file with 200+ entries.

## Actual Behavior

Only the first page is returned, resulting in exactly 10 items in the output JSONL. No error is raised, but the catalog is incomplete.

## Logs / Screenshots

- [ ] Attached minimal logs or screenshot
- Snippet: Output JSONL contains 10 items; no pagination indicators are processed.

## Impact / Severity

- [ ] Blocker
- [ ] High
- [x] Medium
- [ ] Low

## Suspected Cause / Notes

The CK-12 catalog endpoint appears to be paged with a default page size of 10. The downloader likely fetches only the first page and does not follow a `next` link or offset/limit parameter. Investigate `lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_catalog` for missing pagination or query params.

## Proposed Fix / Validation Ideas

- [ ] Unit coverage areas: add a unit test that simulates a paged response and verifies aggregation across pages.
- [ ] Integration scenario to retest: run `ck12_catalog` and confirm output count matches the source catalog size.
- [ ] Manual verification notes: spot-check a few known CK-12 titles that appear after page 1.

## Next Step

- [x] Promote to GitHub issue (bug-report template)
- [ ] Move to active fix folder / branch