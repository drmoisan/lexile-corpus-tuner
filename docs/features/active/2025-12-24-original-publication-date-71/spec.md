# 2025-12-24-original-publication-date — Spec

- Issue: #71
- Owner: drmoisan
- Last Updated: 2025-12-24

## Overview

The Gutenberg books list does not have a reliable field for first publication date by which I can filter and downweight older titles that may use older english.


## Behavior

Enrich the Gutenberg parquet (`gutenberg_books.parquet`) with an additional field that captures the original print publication year (distinct from PG “Issued” release date) so downstream pipelines can filter or weight by publication era.

Planned approach (practical and scalable):
- Primary enrichment: call the **Open Library Search API** to retrieve `first_publish_year` for each Gutenberg ID using title + author matching; store as `original_pub_year` (integer) and keep the existing PG `issued_date` unchanged.
- Write the selected `first_publish_year` into the output parquet (default: `data/meta/gutenberg/gutenberg_books_enhanced.parquet`) as `original_pub_year`, leaving `issued_date` untouched.
- Preserve partial coverage: do not overwrite records lacking a confident match; instead, set `original_pub_year` to null and retain a `pub_year_confidence` flag (e.g., `high` when exact title+author match, `low` when fuzzy).
- Keep pipeline batchable over >60k titles with request throttling, retry, and checkpointing similar to existing Gutendex fetch.

Out of scope:
- Wikidata / Library of Congress fallbacks are tracked as a separate potential feature (see `docs/features/potential/2026-01-06-enhance-gutenberg-wiki-loc-fallback.md`).


## Inputs / Outputs

- Inputs (CLI flags, files, env vars)
	- Input parquet: `data/meta/gutenberg/gutenberg_books.parquet` (or user-specified path).
	- API base: Open Library Search (`https://openlibrary.org/search.json`).
	- CLI flags: input parquet path, output parquet path (can be in-place), rate-limit (req/sec), max retries/backoff, checkpoint path, optional fuzzy-match toggle/threshold.
	- Env vars/config: HTTP timeout, user agent, cache directory (optional local JSON cache per PG ID).
- Outputs (artifacts, logs, telemetry)
	- Updated parquet (default output path: `data/meta/gutenberg/gutenberg_books_enhanced.parquet`; in-place allowed when explicitly set) with added columns: `original_pub_year` (int or null, sourced from Open Library `first_publish_year`), `pub_year_confidence` (enum: high/low/none), and `original_pub_source` (e.g., `openlibrary`, `openlibrary_error` when calls fail).
	- Checkpoint file capturing last processed offset/batch.
	- Run summary (counts: total rows processed, matched high/low, unmatched, API errors, nulls) emitted to stdout/log.
	- Optional cache file storing raw API responses keyed by Gutenberg ID or normalized title+author.

## API / CLI Surface

- CLI command (draft):
	- `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.enrich_original_pub_year --input data/meta/gutenberg/gutenberg_books.parquet --output data/meta/gutenberg/gutenberg_books_enhanced.parquet --checkpoint data/meta/gutenberg/.original_pub_year.ckpt --rate-limit 5 --max-retries 5 --fuzzy-threshold 0.9`
- Flags:
	- `--input`: source parquet path (required)
	- `--output`: destination parquet path (defaults to input for in-place)
	- `--checkpoint`: path for resumable progress (required for long runs)
	- `--rate-limit`: requests per second (default conservative to respect Open Library)
	- `--max-retries`, `--initial-backoff`: network resilience
	- `--fuzzy-threshold`: minimum similarity for non-exact matches; `--disable-fuzzy` to require exact match
	- `--cache-dir`: optional on-disk cache for API responses
- Request shape (Open Library): `GET /search.json?title={title}&author={author}&limit=5`
- Match strategy:
	- Prefer exact title+author (case/whitespace-insensitive); if exact year present, mark `pub_year_confidence=high`.
	- Otherwise consider fuzzy candidate (above threshold) with `pub_year_confidence=low`.
	- If no acceptable candidate, leave year null and `pub_year_confidence=none`.

## Data & State

- Data flow:
	1) Read parquet rows; extract `id`, `title`, `authors`.
	2) Normalize title/author (casefold, strip punctuation) for matching.
	3) Query Open Library with throttling and retries.
	4) Select best candidate per row; derive `original_pub_year` and `pub_year_confidence`.
	5) Write updated parquet; persist checkpoint and optional cache.
- State changes: add columns to the Gutenberg parquet; checkpoint files to support resume; optional cached API responses for reuse.

## Constraints & Risks

List notable constraints (performance, compatibility, scope) or risks.

- Open Library enforces rate limits; bulk runs must throttle and checkpoint.
- Title/author strings can be noisy; fuzzy matching may yield false positives—must gate by confidence and prefer exact matches when available.
- Coverage is partial: many older PG titles may have no external publication year; data should remain nullable.
- Network/API dependencies introduce fragility; implement retries with backoff and allow cached results.
- Keep parquet schema stable for existing consumers; add new columns rather than repurposing `issued_date`.


## Definition of Done

- [ ] Behavior matches acceptance criteria
- [ ] Tests updated/added
- [ ] Docs updated (README, docs/features/active/... links)
- [ ] Telemetry/logging (if applicable)

## Seeded Test Conditions (from potential)
- [ ] Unit coverage areas
- [ ] Integration scenarios
- [ ] CLI/API examples
- [ ] Recommended specifics:
- Unit: title/author normalization, exact vs fuzzy match selection, confidence scoring, null/default behaviors.
- Integration (mocked APIs): batch run over a small fixture parquet, with throttling and checkpoint resume.
- CLI example: run enrichment over `data/meta/gutenberg/gutenberg_books.parquet` and emit summary counts (total rows, matched, unmatched, null years).
