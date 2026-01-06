# Original Publication Year Enrichment — Design Notes

## Column Schema and Defaults
- `original_pub_year`: int32, nullable; original publication year when confidently matched; null when unknown.
- `pub_year_confidence`: enum {`high`, `low`, `none`}.
  - `high`: exact title+author match (case/whitespace/punctuation normalized) with year present.
  - `low`: best fuzzy candidate above threshold with year present.
  - `none`: no acceptable candidate or no year available.
- `original_pub_source`: string; `openlibrary` when primary match, `openlibrary_error` on provider failure, null when no match.
- Default output parquet: `data/meta/gutenberg/gutenberg_books_enhanced.parquet` (never overwrite `issued_date`).

## Matching Thresholds and Rules
- Normalization: casefold, strip punctuation, collapse whitespace.
- Exact path: match on normalized title + at least one author token; year present -> `high`.
- Fuzzy path: use similarity score; accept when score >= 0.90 -> `low`; otherwise `none`.
- Tie-breakers: prefer candidate with year present; if multiple, prefer highest score; if still tied, prefer earliest `first_publish_year`.
## Operational Controls
 - Rate limit: default 5 req/sec (applied to Open Library).
- Retries: max 5; exponential backoff starting 0.5s, capped at 8s; retry on 5xx/429/timeouts.
- Checkpoint cadence: every 500 rows processed (writes last offset + stats); resume continues from next unprocessed row.
- Cache: optional on-disk cache at `data/cache/original_pub_year` keyed by PG ID + normalized title/author; cache hits skip network.
- Timeouts: HTTP timeout 10s per request; circuit-break on consecutive failures by respecting backoff.
