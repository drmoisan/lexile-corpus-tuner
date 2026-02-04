# dropped-links (Issue #115)

- Date captured: 2026-02-04
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/dropped-links/ (Issue #115)

> Automation note: Keep the section headings below unchanged; the promotion tooling maps each of them into the GitHub bug issue template.

- Issue: #115
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/115
- Last Updated: 2026-02-04
## Summary

CK-12 pipeline step 4 (`oer_manifest --validate-urls`) fails URL validation for all CK-12 entries. The manifest generator writes zero entries because every HEAD request is blocked by the CK-12 CDN.

## Environment

- OS/version: Linux (Debian GNU/Linux 12, dev container)
- Python version: 3.13.9
- Command/flags used: `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_manifest --catalog-dir data/meta/catalogs --out data/meta/oer_sources.json --validate-urls`
- Data source or fixture: `data/meta/catalogs/ck12_curated.jsonl`

## Steps to Reproduce

1. Run the CK-12 catalog/enrichment/curation steps to generate `ck12_curated.jsonl`.
2. Run `oer_manifest` with `--validate-urls` pointing at `data/meta/catalogs`.
3. Observe the validation failure messages and the resulting manifest size.

## Expected Behavior

CK-12 entries validate successfully and the manifest contains the expected CK-12 items with JSON URLs.

## Actual Behavior

All 155 CK-12 entries fail validation with `status=None, content_type=None`, causing a zero-entry manifest. Validation requests return HTTP 403 from the CK-12 CDN.

## Logs / Screenshots

- [ ] Attached minimal logs or screenshot
- Snippet: `validation failed (status=None, content_type=None)`

## Impact / Severity

- [ ] Blocker
- [x] High
- [ ] Medium
- [ ] Low

## Suspected Cause / Notes

The CK-12 CloudFront CDN blocks requests without a browser-like User-Agent. `validate_url()` in `oer_manifest.py` sends HEAD requests without a User-Agent header, so all requests are rejected and surfaced as `(False, None, None)`.

## Proposed Fix / Validation Ideas

- [ ] Unit coverage areas: `tests/lexile_scoring_model/pipeline_scripts/test_oer_manifest.py` (validate_url + CK-12 acceptance).
- [ ] Integration scenario to retest: rerun `oer_manifest --validate-urls` and confirm 155 entries are written.
- [ ] Manual verification notes: add User-Agent header in `validate_url()` and verify CK-12 URLs return HTTP 200 + `application/json`.

## Next Step

- [ ] Promote to GitHub issue (bug-report template)
- [ ] Move to active fix folder / branch