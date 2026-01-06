# `2025-12-24-original-publication-date` — User Story

- Issue: #71
- Owner: drmoisan
- Status: Draft | In Progress | Complete
- Last Updated: 2025-12-24

## Story Statement

- As a corpus engineer, I want to enrich Gutenberg metadata with an original publication year plus a confidence flag while keeping `issued_date` untouched, so bulk pipelines can downweight older language without corrupting PG release data.
- As a data scientist building difficulty models, I want `original_pub_year` + `pub_year_confidence` available with nullable values when unknown, so I can weight segments by era and audit match quality.

## Problem / Why

The Gutenberg books list does not have a reliable field for first publication date by which I can filter and downweight older titles that may use older english.


## Personas & Scenarios

- Persona: Corpus engineer
  - Cares about scalable enrichment over 60k+ Gutenberg rows with rate limits, retries, checkpoint/resume, and optional response caching.
  - Constraints: external API quotas; noisy title/author strings; no manual curation; must avoid overwriting existing schema fields.
  - Goal: batch-enrich parquet with `original_pub_year` + `pub_year_confidence` while leaving `issued_date` intact and producing resumable artifacts.
  - Frustration: PG “Issued” is not original publication; bulk metadata lacks that year.
- Persona: Data scientist (Lexile estimator)
  - Cares about clean, nullable era signals to weight documents for frequency tables and calibration.
  - Constraints: avoids false positives; needs confidence bands (high/low/none) to gate usage.
  - Goal: segment corpora by era buckets and downweight older narrative while preserving PG release provenance.
- Scenario: A corpus engineer runs the enrichment CLI against `data/meta/gutenberg/gutenberg_books.parquet` with Open Library as the source plus throttling, retries, checkpoint every N rows, and optional cache. The run writes `original_pub_year`, `pub_year_confidence`, and `original_pub_source`, emits summary counts (matched high/low, unmatched/null, API errors), and can resume from checkpoint. A data scientist then filters/weights by era (e.g., favor 1990+) while keeping `issued_date` for provenance and ignoring rows with `pub_year_confidence=none`.

- Scenario: A corpus engineer runs the enrichment CLI against `data/meta/gutenberg/gutenberg_books.parquet` with Open Library as the source, throttling, retries, checkpoint every N rows, and optional cache. The run writes `original_pub_year`, `pub_year_confidence`, and `original_pub_source`, emits summary counts (matched high/low, unmatched/null, API errors), and can resume from checkpoint. A data scientist then filters/weights by era (e.g., favor 1990+) while keeping `issued_date` for provenance and ignoring rows with `pub_year_confidence=none`.


## Acceptance Criteria

- [ ] Enrichment CLI populates `original_pub_year` where a confident external match exists; leaves null otherwise and records `original_pub_source`.
- [ ] `issued_date` (PG release) remains unchanged and is never repurposed.
- [ ] Adds `pub_year_confidence` with explicit bands: `high` (exact/strong match), `low` (fuzzy above threshold), `none` (no acceptable match).
- [ ] Runs end-to-end over full Gutenberg parquet (>60k rows) with throttling, retries, and resumable checkpointing; emits run summary counts (matched high/low, null, errors).
- [ ] Unit tests cover normalization, exact vs fuzzy match selection, confidence labeling, null-handling, and source tagging.
- [ ] Integration test (mocked APIs) covers batch processing, throttling, checkpoint resume, parquet write, and summary reporting.

## Related potential feature

- Wikidata / Library of Congress fallback is tracked separately in `docs/features/potential/2026-01-06-enhance-gutenberg-wiki-loc-fallback.md`.


## Non-Goals

- Inferencing full publication dates or edition-level details; only year is targeted.
- Manual curation or hand-edited fixes at scale.
- Overwriting or repurposing PG `issued_date`.
- Guaranteeing coverage for all Gutenberg titles; null is acceptable when no confident match exists.
- Building a general-purpose bibliographic reconciler or scraping beyond documented Open Library / Wikidata/LOC endpoints.
