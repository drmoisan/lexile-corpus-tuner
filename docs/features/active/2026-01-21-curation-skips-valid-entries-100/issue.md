# curation-skips-valid-entries (Issue #100)

- Date captured: 2026-01-21
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/curation-skips-valid-entries/ (Issue #100)

> Automation note: Keep the section headings below unchanged; the promotion tooling maps each of them into the GitHub bug issue template.

- Issue: #100
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/100
- Last Updated: 2026-01-21
## Summary

CK-12 curation with `--require-json` is skipping valid entries because the catalog assigns artifact types (`flexbook`/`book`) that do not match Perma API types. This results in empty revision JSON candidates and `no json candidate` skips for entries that do have valid Perma endpoints.

## Environment

- OS/version: Linux (dev container; Debian GNU/Linux 12)
- Python version: Not captured in session output.
- Command/flags used: `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_curation --catalog-dir data/meta/catalogs --require-json --sources "ck12" --out-dir data/meta/catalogs`
- Data source or fixture: `data/meta/catalogs/ck12_enriched.jsonl` generated from CK-12 Browse feed (`https://static.ck12.org/testimonial/fbbrowse-prod.json`) via `ck12_catalog` + `ck12_enrichment`.

## Steps to Reproduce

1. Run CK-12 catalog build: `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_catalog --out-dir data/meta/catalogs`.
2. Run CK-12 enrichment: `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_enrichment --catalog-file data/meta/catalogs/ck12_catalog.jsonl --output data/meta/catalogs/ck12_enriched.jsonl`.
3. Curate CK-12 with JSON requirement: `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_curation --catalog-dir data/meta/catalogs --require-json --sources "ck12" --out-dir data/meta/catalogs`.

## Expected Behavior

Valid CK-12 entries should retain revision JSON candidates and be kept during curation when `--require-json` is used.

## Actual Behavior

Nearly half of CK-12 entries are skipped with `no json candidate`. The run output is:
`Curated ck12_enriched.jsonl: 84 kept, 81 skipped -> data/meta/catalogs/ck12_curated.jsonl`.
The skip file `data/meta/catalogs/ck12_skips.jsonl` contains 81 entries, all with reason `no json candidate`.

## Logs / Screenshots

- [x] Attached minimal logs or screenshot
- Snippet:
	- `Curated ck12_enriched.jsonl: 84 kept, 81 skipped -> data/meta/catalogs/ck12_curated.jsonl`
	- `ck12_skips.jsonl` shows `{"identifier": "...", "reason": "no json candidate"}` for all 81 rows.

## Impact / Severity

- [ ] Blocker
- [ ] High
- [x] Medium
- [ ] Low

## Suspected Cause / Notes

- `ck12_catalog.py` assigns `artifact_type` as `flexbook` when `content_url_host == "flexbooks.ck12.org"` and `book` when `content_url_host == "www.ck12.org"`. The Perma API does not accept `flexbook` and expects `cbook`, `tebook`, `workbook`, or `quizbook` depending on the URL prefix.
- Perma validation examples:
	- `https://www.ck12.org/flx/get/perma/flexbook/cbse-biology-class-10` -> `Unknown artifact type[flexbook]`.
	- `https://www.ck12.org/flx/get/perma/cbook/cbse-biology-class-10` -> valid response with `revisions`.
	- `https://www.ck12.org/flx/get/perma/quizbook/CK-12-Biology-Quizzes-and-Tests` -> valid response with `revisions`.

## Proposed Fix / Validation Ideas

- [x] Unit coverage areas
	- Add/extend tests to map CK-12 URL prefixes (`/cbook/`, `/tebook/`, `/workbook/`, `/quizbook/`, `/book/`) to Perma artifact types.
- [x] Integration scenario to retest
	- Re-run CK-12 catalog + enrichment + curation with `--require-json`; validate that `no json candidate` drops materially and newly mapped entries are kept.
- [x] Manual verification notes
	- Verified Perma success for a flexbook-derived entry by mapping to `cbook`: `cbse-biology-class-10` returns a valid Perma payload with `revisions`.
	- Verified `tebook`, `workbook`, and `quizbook` endpoints return valid payloads for representative handles.

## Next Step

- [ ] Promote to GitHub issue (bug-report template)
- [ ] Move to active fix folder / branch