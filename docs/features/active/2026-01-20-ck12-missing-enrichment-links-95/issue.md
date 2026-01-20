# ck12-missing-enrichment-links (Issue #95)

- Date captured: 2026-01-20
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/ck12-missing-enrichment-links/ (Issue #95)


- Issue: #95
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/95
- Last Updated: 2026-01-20
## Summary

When running the ck12 pipeline, 165 catalog entries are created. However, the enrichment process initially revealed 0 enriched entries. This was because there was no fallback for the content_url_slug in the case where it was NOT a static feed entry, there was NO raw handle, AND slug has a real value. This was fixed and now 133 catalog entries are created. There are 32 missing entries, likely due to improper identification of a content_url_slug in these cases. 

## Environment

- OS/version: Debian GNU/Linux 12 (bookworm) (devcontainer)
- Python version: Not captured in logs (use `python --version` to confirm)
- Command/flags used:
	- `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_catalog --out-dir data/meta/catalogs`
	- `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_enrichment --catalog-file data/meta/catalogs/ck12_catalog.jsonl --output data/meta/catalogs/ck12_enriched.jsonl`
- Data source or fixture:
	- `https://static.ck12.org/testimonial/fbbrowse-prod.json`

## Steps to Reproduce

1. Run the CK-12 catalog step to generate `data/meta/catalogs/ck12_catalog.jsonl`.
2. Run the CK-12 enrichment step on that catalog output.
3. Observe enrichment logs and output count; 32 entries are skipped with `missing handle`.

## Expected Behavior

Enrichment should produce entries for all 165 catalog rows when `Content_URL` is present, with minimal `missing handle` skips. If the Perma API rejects a subset, those should be explicitly logged with non-handle-related reasons.

## Actual Behavior

Enrichment produced 133 entries and skipped 32 rows due to missing handles. Logs show repeated lines such as:

`Skipping <identifier> due to missing handle`

## Logs / Screenshots

- [x] Attached minimal logs or screenshot
- Snippet:
	- `Skipping ck-12-biology-workbook due to missing handle`
	- `Skipping ck-12-earth-science-for-high-school-teacher-s-edition due to missing handle`

## Impact / Severity

- [ ] Blocker
- [ ] High
- [x] Medium
- [ ] Low

## Suspected Cause / Notes

- `extract_slug_from_content_url` only recognizes `/cbook/` and `/book/` paths, but the CK-12 feed includes `/tebook/`, `/workbook/`, and `/quizbook/` URLs. This causes `content_url_slug` to be `None`, and `entry.handle` remains `None` for those rows.
- See:
	- `ck12_catalog.py`: `extract_slug_from_content_url` (URL parsing logic)
	- `ck12_catalog.py`: `entry_handle` assignment (handle fallback logic)
	- `ck12_enrichment.py`: `if handle is None: ... continue`

## Proposed Fix / Validation Ideas

- [ ] Unit coverage areas
	- Add tests for `extract_slug_from_content_url` to cover `/tebook/`, `/workbook/`, and `/quizbook/` paths.
- [x] Integration scenario to retest
	- Rerun `ck12_catalog` → `ck12_enrichment` on the live CK-12 feed and verify enriched count increases from 133 toward 165.
- [x] Manual verification notes
	- Confirm that previously skipped identifiers now have `handle` values in `ck12_catalog.jsonl`.
	- Validate that `missing handle` log lines approach zero.

## Next Step

- [ ] Promote to GitHub issue (bug-report template)
- [ ] Move to active fix folder / branch