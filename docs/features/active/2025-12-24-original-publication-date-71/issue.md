# original-publication-date (Issue #71)

- Date captured: 2025-12-24
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/original-publication-date/ (Issue #71)

- Issue: #71
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/71
- Last Updated: 2025-12-24
## Problem / Why

The Gutenberg books list does not have a reliable field for first publication date by which I can filter and downweight older titles that may use older english.

## Proposed Behavior
Enrich the Gutenberg parquet (`gutenberg_books.parquet`) with an additional field that captures the original print publication year (distinct from PG “Issued” release date) so downstream pipelines can filter or weight by publication era.

Planned approach (practical and scalable):
- Primary enrichment: call the **Open Library Search API** to retrieve `first_publish_year` for each Gutenberg ID using title + author matching; store as `original_pub_year` (integer) and keep the existing PG `issued_date` unchanged.
- Fallback coverage (optional, if enabled):
	- If Open Library has no match, attempt Wikidata/LOC lookups when an identifier (LCCN/ISBN) is present.
	- If an Open Library match exists but lacks year, keep `original_pub_year` null.
- Preserve partial coverage: do not overwrite records lacking a confident match; instead, set `original_pub_year` to null and retain a `pub_year_confidence` flag (e.g., `high` when exact title+author match, `low` when fuzzy).
- Keep pipeline batchable over >60k titles with request throttling, retry, and checkpointing similar to existing Gutendex fetch.

## Acceptance Criteria (early draft)
- [ ] New enrichment script/CLI populates `original_pub_year` for Gutenberg rows where a confident external match exists; leaves null otherwise.
- [ ] `issued_date` (PG release) remains unchanged and is not repurposed for original publication year.
- [ ] Adds `pub_year_confidence` (e.g., high/low/none) to indicate match quality.
- [ ] Runs end-to-end over the full Gutenberg parquet (>60k rows) with resumable checkpointing and rate-limit compliance.
- [ ] Unit tests cover matching logic (exact vs fuzzy), confidence labeling, and null-handling.
- [ ] Integration test (with mocked APIs) covers batch processing, throttling, and parquet update.

## Constraints & Risks

List notable constraints (performance, compatibility, scope) or risks.

- Open Library and other bibliographic APIs enforce rate limits; bulk runs must throttle and checkpoint.
- Title/author strings can be noisy; fuzzy matching may yield false positives—must gate by confidence and prefer exact matches when available.
- Coverage is partial: many older PG titles may have no external publication year; data should remain nullable.
- Network/API dependencies introduce fragility; implement retries with backoff and allow cached results.
- Keep parquet schema stable for existing consumers; add new columns rather than repurposing `issued_date`.

## Test Conditions to Consider

- [ ] Unit coverage areas
- [ ] Integration scenarios
- [ ] CLI/API examples

Recommended specifics:
- Unit: title/author normalization, exact vs fuzzy match selection, confidence scoring, null/default behaviors.
- Integration (mocked APIs): batch run over a small fixture parquet, with throttling and checkpoint resume.
- CLI example: run enrichment over `data/meta/gutenberg/gutenberg_books.parquet` and emit summary counts (total rows, matched, unmatched, null years).

## Next Step

- [ ] Promote to GitHub issue (feature request template)
- [ ] Create `docs/features/active/original-publication-date/` folder from the template
