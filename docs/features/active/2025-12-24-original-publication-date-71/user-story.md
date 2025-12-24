# `2025-12-24-original-publication-date` — User Story

- Issue: #71
- Owner: drmoisan
- Status: Draft | In Progress | Complete
- Last Updated: 2025-12-24

## Story Statement

- As a corpus engineer, I want to enrich Gutenberg metadata with an original publication year, so that downstream filters and weighting can de-emphasize outdated language.
- As a data scientist building difficulty models, I want a confidence-scored `original_pub_year` alongside the PG `issued_date`, so that I can segment, weight, and evaluate content by era without corrupting the PG release metadata.

## Problem / Why

The Gutenberg books list does not have a reliable field for first publication date by which I can filter and downweight older titles that may use older english.


## Personas & Scenarios

- Persona: Corpus engineer
  - Cares about scalable metadata enrichment over 60k+ Gutenberg rows with checkpointing and rate-limit compliance.
  - Constraints: external API quotas; noisy title/author strings; no manual curation budget.
  - Goal: batch-enrich parquet with `original_pub_year` plus confidence flag without breaking existing schema.
  - Frustration: PG “Issued” is not the original publication year; bulk metadata lacks this field.
- Persona: Data scientist (Lexile estimator)
  - Cares about clean signals to weight documents by era for frequency tables and model calibration.
  - Constraints: needs nullable values when unknown; avoids false positives from fuzzy matches.
  - Goal: segment corpora by era buckets to counter Gutenberg age bias; retain PG release date for provenance.
- Scenario: A corpus engineer runs the enrichment CLI on `data/meta/gutenberg_books.parquet`, which reads rows, batches Open Library lookups with throttling, writes `original_pub_year` and `pub_year_confidence`, checkpoints progress every N rows, and emits a summary report (matched/unmatched/null). A data scientist then filters documents to favor 1990+ narrative content and downweights pre-1950 items using the new field while leaving PG `issued_date` intact.


## Acceptance Criteria

- [ ] New enrichment script/CLI populates `original_pub_year` for Gutenberg rows where a confident external match exists; leaves null otherwise.
- [ ] `issued_date` (PG release) remains unchanged and is not repurposed for original publication year.
- [ ] Adds `pub_year_confidence` (e.g., high/low/none) to indicate match quality.
- [ ] Runs end-to-end over the full Gutenberg parquet (>60k rows) with resumable checkpointing and rate-limit compliance.
- [ ] Unit tests cover matching logic (exact vs fuzzy), confidence labeling, and null-handling.
- [ ] Integration test (with mocked APIs) covers batch processing, throttling, and parquet update.


## Non-Goals

- Inferencing full publication dates or editions; only the year is targeted.
- Manual curation or hand-edited year fixes at scale.
- Replacing or overwriting PG `issued_date`.
- Guaranteeing coverage for all Gutenberg titles; null is acceptable when no confident match exists.
- Building a general-purpose bibliographic reconciler beyond the needs of this corpus pipeline.
