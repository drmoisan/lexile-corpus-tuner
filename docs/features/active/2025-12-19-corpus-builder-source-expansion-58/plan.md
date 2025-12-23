# corpus-builder-source-expansion - Plan

- Issue: #58
- Owner: drmoisan
- Last Updated: 2025-12-23

## Required References

- General Coding Standards: [`.github/instructions/general-code-change.instructions.md`](../../../../.github/instructions/general-code-change.instructions.md)
- General Unit Test Policy: [`.github/instructions/general-unit-test.instructions.md`](../../../../.github/instructions/general-unit-test.instructions.md)
- (Add language-specific policies as needed, e.g. `python-code-change.instructions.md`)

**All work must comply with these policies; do not duplicate their content here.**

## Implementation Plan (Atomic Tasks)

### Phase 0 — Context & Inputs
- [ ] [P0-T1] Read .github/copilot-instructions.md and .github/instructions/general-code-change.instructions.md to confirm baseline policies
- [ ] [P0-T2] Read .github/instructions/general-unit-test.instructions.md and .github/instructions/python-code-change.instructions.md to capture testing and Python standards
- [ ] [P0-T3] Read .github/instructions/python-unit-test.instructions.md to align Pytest expectations
- [ ] [P0-T4] Review docs/features/active/2025-12-19-corpus-builder-source-expansion-58/user-story.md and spec.md to internalize scope and acceptance criteria

### Phase 1 — Scope & Design Decisions
- [ ] [P1-T1] Document target source list (gutenberg, simple_wiki, wiki, oer, modern_fiction_cc) and initial weight matrix (by source/era) in the plan notes
- [ ] [P1-T2] Specify normalized schema fields (source, genre, publication_year, era_bucket, intended_audience, weight) and defaults when data is missing
- [ ] [P1-T3] Identify required raw inputs (Gutenberg cache path, Wikipedia dumps, OER locations) and note any download prerequisites

### Phase 2 — Ingestion & Normalization
- [ ] [P2-T1] Extend Gutenberg ingestion to extract publication_year/era and tag genre=narrative with intended_audience inference
- [ ] [P2-T2] Add Simple/Standard Wikipedia ingestion that filters articles by length/quality and emits normalized document chunks
- [ ] [P2-T3] Add one OER/textbook ingestion path that produces instructional/expository chunks with grade band metadata when available
- [ ] [P2-T4] Ensure all sources segment into 1k–3k word doc units with stable chunking and consistent metadata fields

### Phase 3 — Metadata & Schema Wiring
- [ ] [P3-T1] Update normalized corpus schema definitions to include source, genre, publication_year or era_bucket, intended_audience, and weight
- [ ] [P3-T2] Implement fallback logic for missing publication_year (e.g., unknown/era inference) and persist it in metadata
- [ ] [P3-T3] Add validation to reject documents without required metadata fields before frequency computation

### Phase 4 — Weighted Frequency Computation
- [ ] [P4-T1] Implement config-driven weight resolution per document (source + era) in the frequency aggregation component
- [ ] [P4-T2] Compute weighted word frequencies and emit data/freq/weighted_word_frequencies.tsv plus accompanying metadata JSON
- [ ] [P4-T3] Generate a corpus stats report showing token counts per source/era before and after weighting to demonstrate bias correction

### Phase 5 — CLI & Configuration Surface
- [ ] [P5-T1] Add CLI flags to corpus download/normalize/frequencies commands for selecting sources and enabling weighted mode
- [ ] [P5-T2] Update example config.yaml to include source activation and weight matrix definitions
- [ ] [P5-T3] Document default weights and how to override them via configuration

### Phase 6 — Tests (Scenario-Specific)
- [ ] [P6-T1] Add Pytest scenario covering Gutenberg ingestion tagging genre=narrative and storing publication_year/era when available in tests/ (choose appropriate module)
- [ ] [P6-T2] Add Pytest scenario ensuring Wikipedia ingestion filters articles by length/quality and emits source-specific metadata in tests/
- [ ] [P6-T3] Add Pytest scenario confirming OER ingestion emits instructional/expository genre with grade band when present in tests/
- [ ] [P6-T4] Add Pytest scenario verifying weighted frequency calculator applies per-document weights and normalizes counts in tests/
- [ ] [P6-T5] Add Pytest scenario that rejects documents missing required metadata before aggregation in tests/
- [ ] [P6-T6] Add Pytest scenario validating CLI `corpus frequencies --weighted --config` parses weights and produces weighted output paths in tests/

### Phase 7 — Documentation & Handoff
- [ ] [P7-T1] Update README.md and docs/features/active/2025-12-19-corpus-builder-source-expansion-58/spec.md to describe new sources, metadata, and weighting usage
- [ ] [P7-T2] Add usage examples for weighted frequencies and source selection to docs/features/active/2025-12-19-corpus-builder-source-expansion-58/user-story.md or related docs
- [ ] [P7-T3] Capture open issues/risks (disk space, metadata gaps) and mitigation steps in plan notes

## Test Plan

- Unit: Pytest scenarios in Phase 6 covering ingestion metadata, weighting logic, validation, and CLI parsing
- Integration: End-to-end run of corpus download → normalize → frequencies with weighted config on a small fixture subset
- Manual/CLI: Smoke run of lexile-scoring-model-pipeline corpus frequencies --weighted --config <sample> confirming output files and stats report

## Open Questions / Notes

- Are there existing lightweight OER datasets already mirrored in the repo to avoid large downloads during tests?
- What is the preferred default weight matrix per source/era for initial rollout?
