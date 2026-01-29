# corpus-builder-source-expansion - Plan

- Issue: #58
- Owner: drmoisan
- Last Updated: 2025-12-24
- Status: Complete

## Required References

- General Coding Standards: [`.github/instructions/general-code-change.instructions.md`](../../../../.github/instructions/general-code-change.instructions.md)
- General Unit Test Policy: [`.github/instructions/general-unit-test.instructions.md`](../../../../.github/instructions/general-unit-test.instructions.md)
- (Add language-specific policies as needed, e.g. `python-code-change.instructions.md`)

**All work must comply with these policies; do not duplicate their content here.**

## Implementation Plan (Atomic Tasks)

### Phase 0 — Context & Inputs
- [x] [P0-T1] Read .github/copilot-instructions.md and .github/instructions/general-code-change.instructions.md to confirm baseline policies
- [x] [P0-T2] Read .github/instructions/general-unit-test.instructions.md and .github/instructions/python-code-change.instructions.md to capture testing and Python standards
- [x] [P0-T3] Read .github/instructions/python-unit-test.instructions.md to align Pytest expectations
- [x] [P0-T4] Review docs/features/active/2025-12-19-corpus-builder-source-expansion-58/user-story.md and spec.md to internalize scope and acceptance criteria

### Phase 1 — Scope & Design Decisions
- [x] [P1-T1] Document target sources and era buckets with initial weight matrix proposal in plan notes
- [x] [P1-T2] Specify normalized schema fields (source, genre, publication_year, era_bucket, intended_audience, weight) with defaults for missing data
- [x] [P1-T3] List required raw inputs and download prerequisites (Gutenberg cache, Wikipedia dumps, OER locations) in plan notes

**Phase 1 Notes (draft):**
- Target sources: `gutenberg`, `simple_wiki`, `standard_wiki`, `oer`, `modern_fiction_cc`.
- Era buckets: `pre_1950`, `1950_2000`, `post_2000`, `unknown` (fallback when year missing).
- Initial weight matrix (source × era):
	- gutenberg: `pre_1950=0.3`, `1950_2000=0.5`, `post_2000=0.7`, `unknown=0.4`
	- simple_wiki: `post_2000=1.0`, `unknown=0.8`
	- standard_wiki: `post_2000=1.0`, `unknown=0.8`
	- oer: `post_2000=1.1`, `unknown=0.9`
	- modern_fiction_cc: `post_2000=1.0`, `unknown=0.9`
- Normalized schema fields and defaults:
	- `source` (str, required; values above)
	- `genre` (str, required; allowed: `narrative`, `expository`, `instructional`)
	- `publication_year` (int | null; nullable)
	- `era_bucket` (str, required; default `unknown` when year missing)
	- `intended_audience` (str, required; default `general` if missing)
	- `weight` (float, required; resolved per source/era, default 1.0 if unspecified)
- Raw inputs and download prerequisites:
	- Gutenberg: existing cache path required; ensure metadata files (rdf) available for year extraction.
	- Wikipedia: Simple + Standard XML dumps (.bz2); need tokenizer-compatible extraction; filter config for length/quality.
	- OER/textbook: mirrored open-license set (small fixture subset for tests) with grade band metadata when present.

### Phase 2 — Ingestion & Normalization
- [x] [P2-T1] Add Gutenberg publication_year extraction from metadata files
- [x] [P2-T2] Implement Gutenberg era_bucket inference from publication_year with defined buckets
- [x] [P2-T3] Tag Gutenberg documents with genre=narrative and infer intended_audience from available metadata
- [x] [P2-T4] Segment Gutenberg documents into 1k–3k word units with stable chunking metadata
- [x] [P2-T5] Implement Wikipedia length filter (min/max tokens) for Simple/Standard dumps before normalization
- [x] [P2-T6] Implement Wikipedia quality filter (e.g., skip disambiguation/stubs) and retain accepted pages list
- [x] [P2-T7] Tag Wikipedia documents with source (simple_wiki/standard_wiki), genre=expository, intended_audience default, and era_bucket default
- [x] [P2-T8] Segment Wikipedia documents into 1k–3k word units with stable chunking metadata
- [x] [P2-T9] Ingest selected OER/textbook source into normalized form with source tag
- [x] [P2-T10] Tag OER documents with genre=instructional/expository and apply grade band when available
- [x] [P2-T11] Segment OER documents into 1k–3k word units with stable chunking metadata

### Phase 3 — Metadata & Schema Wiring
- [x] [P3-T1] Update normalized corpus schema/constants to include source, genre, publication_year, era_bucket, intended_audience, and weight
- [x] [P3-T2] Update serializers/loaders to read/write the new schema fields across sources
- [x] [P3-T3] Implement fallback logic for missing publication_year to assign era_bucket or unknown consistently
- [x] [P3-T4] Add validation guard that rejects documents missing required metadata fields before aggregation and surfaces clear errors

### Phase 4 — Weighted Frequency Computation
- [x] [P4-T1] Implement config-driven per-document weight resolution (by source + era) in frequency aggregation
- [x] [P4-T2] Apply weights in word count aggregation and normalized frequency calculation
- [x] [P4-T3] Write weighted outputs to data/freq/weighted_word_frequencies.tsv plus accompanying metadata JSON
- [x] [P4-T4] Generate corpus stats report with token counts per source/era before and after weighting

### Phase 5 — CLI & Configuration Surface
- [x] [P5-T1] Add CLI flags to corpus download/normalize commands to select sources
- [x] [P5-T2] Add CLI flag to corpus frequencies to enable weighted mode and accept weight config path
- [x] [P5-T3] Update example config.yaml with source activation and weight matrix definitions

### Phase 6 — Tests (Scenario-Specific)
- [x] [P6-T1] Add Pytest scenario verifying Gutenberg ingestion stores publication_year and inferred era_bucket with genre=narrative tagging
- [x] [P6-T2] Add Pytest scenario verifying Wikipedia ingestion enforces length/quality filters and sets source/genre/intended_audience metadata
- [x] [P6-T3] Add Pytest scenario verifying OER ingestion emits instructional/expository genre and grade band when present
- [x] [P6-T4] Add Pytest scenario verifying weighted frequency calculator applies per-document weights and normalizes counts
- [x] [P6-T5] Add Pytest scenario rejecting documents missing required metadata before aggregation
- [x] [P6-T6] Add Pytest scenario validating CLI corpus frequencies --weighted --config parses weights and writes weighted output paths

### Phase 7 — Documentation & Handoff
- [x] [P7-T1] Update README.md to describe new sources, metadata fields, and weighted frequency usage
- [x] [P7-T2] Update docs/features/active/2025-12-19-corpus-builder-source-expansion-58/spec.md with source metadata and weighting behavior
- [x] [P7-T3] Add usage examples and open issues/risks (disk space, metadata gaps) to user-story.md or plan notes

## Test Plan

- Unit: Pytest scenarios in Phase 6 covering ingestion metadata, weighting logic, validation, and CLI parsing
- Integration: End-to-end run of corpus download → normalize → frequencies with weighted config on a small fixture subset
- Manual/CLI: Smoke run of lexile-scoring-model-pipeline corpus frequencies --weighted --config <sample> confirming output files and stats report

## Open Questions / Notes

- Are there existing lightweight OER datasets already mirrored in the repo to avoid large downloads during tests?
- What is the preferred default weight matrix per source/era for initial rollout?
