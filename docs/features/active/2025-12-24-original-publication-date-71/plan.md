# 2025-12-24-original-publication-date - Plan

- Issue: #71
- Owner: 2025-12-24-original-publication-date
- Last Updated: 2025-12-24

## Required References

- General Coding Standards: [`.github/instructions/general-code-change.instructions.md`](../../../../.github/instructions/general-code-change.instructions.md)
- General Unit Test Policy: [`.github/instructions/general-unit-test.instructions.md`](../../../../.github/instructions/general-unit-test.instructions.md)
- (Add language-specific policies as needed, e.g. `python-code-change.instructions.md`)

**All work must comply with these policies; do not duplicate their content here.**

## Implementation Plan (Atomic Tasks)

**Phase 0 — Context & Inputs**
- [x] [P0-T1] Read .github/copilot-instructions.md and .github/instructions/general-code-change.instructions.md to confirm baseline rules
- [x] [P0-T2] Read .github/instructions/general-unit-test.instructions.md, .github/instructions/python-code-change.instructions.md, and .github/instructions/python-unit-test.instructions.md to confirm tooling and coverage expectations
- [x] [P0-T3] Read docs/features/active/2025-12-24-original-publication-date-71/issue.md, user-story.md, and spec.md to extract acceptance criteria and constraints
- [x] [P0-T4] Read artifacts/chats/251224 Corpus Design Summary.md to capture intended behavior (confidence bands, throttling, checkpointing, caching)
- [x] [P0-T5] Run baseline Ruff (`poetry run ruff check`) and record pass/fail (no fixes)
- [x] [P0-T6] Run baseline Pyright (`poetry run pyright`) and record pass/fail
- [x] [P0-T7] Run baseline Pytest (`poetry run pytest`) and record pass/fail
- [x] [P0-T8] Run baseline JSON format check (`poetry run python -m scripts.dev_tools.format_json --check`) and record pass/fail
- [x] [P0-T9] Run baseline JSON schema validation (`poetry run python -m scripts.dev_tools.validate_json`) and record pass/fail
- [x] [P0-T10] Run baseline PowerShell format (`pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "& { Import-Module './scripts/powershell/PoshQC'; Invoke-PoshQCFormat -Root '.' }"`) and record pass/fail
- [x] [P0-T11] Run baseline PowerShell analyze (`pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "& { Import-Module './scripts/powershell/PoshQC'; Invoke-PoshQCAnalyze -Root '.' }"`) and record pass/fail
- [x] [P0-T12] Run baseline PowerShell tests (`pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "& { Import-Module './scripts/powershell/PoshQC'; Invoke-PoshQCTest -Root '.' }"`) and record pass/fail

**Phase 1 — Design Decisions**
- [x] [P1-T1] Decide column schema and defaults: original_pub_year (int/null), pub_year_confidence (high/low/none), original_pub_source (text), output path default
- [x] [P1-T2] Decide matching thresholds and rules: exact vs fuzzy thresholds, tie-breakers, null rules, fallback ordering (Open Library primary, optional Wikidata/LOC)
- [x] [P1-T3] Decide operational controls: rate-limit (req/sec), batch size, retry/backoff, checkpoint cadence, cache location/format, resume semantics; record in a short design note

**Phase 2 — Foundations**
- [x] [P2-T1] Add module skeleton for enrichment (e.g., src/lexile_corpus_tuner/lexile_scoring_model/original_pub_year.py) exporting CLI handler and core functions
- [x] [P2-T2] Implement CLI entry point wiring with argparse/typer equivalent, exposing flags for input/output, checkpoint, rate-limit, batch-size, retries/backoff, fuzzy threshold/disable, cache-dir, fallback toggles
- [x] [P2-T3] Add dataclasses/types for match result and config (including confidence enum) with type hints

**Phase 3 — Matching & Clients**
- [x] [P3-T1] Implement normalization helper for title/author (casefold, strip punctuation/whitespace) with deterministic output
- [x] [P3-T2] Implement Open Library client with throttling, retries, timeouts, and typed response parsing
- [x] [P3-T3] Implement matcher to select best candidate: prefer exact, else fuzzy above threshold, else none; emit year, confidence, source
- [x] [P3-T4] Implement fallback client wiring (Wikidata/LOC) behind feature flags without altering primary flow

**Phase 4 — Caching, Checkpoint, Summary**
- [x] [P4-T1] Implement optional response cache read/write keyed by normalized title/author or PG ID
- [x] [P4-T2] Implement checkpoint/resume mechanism capturing last processed offset/batch
- [x] [P4-T3] Implement summary aggregation (matched high/low, none/null, errors) and structured logging/printout

**Phase 5 — Parquet Processing**
- [x] [P5-T1] Implement parquet reader/writer that loads input, adds/updates columns, and writes output without altering existing fields
- [x] [P5-T2] Implement batch processing loop applying matching, cache use, checkpointing, and throttling across >60k rows
- [x] [P5-T3] Guard issued_date to ensure it remains unchanged in output

**Phase 6 — Tests (Scenario-Specific)**
- [x] [P6-T1] Add unit test for normalization helper covering punctuation/casefold behavior in tests/lexile_scoring_model/test_original_pub_year_enrichment.py
- [x] [P6-T2] Add unit test: exact title+author yields confidence high with expected year
- [x] [P6-T3] Add unit test: fuzzy candidate above threshold yields confidence low; below threshold yields confidence none/null year
- [x] [P6-T4] Add unit test: missing year in candidate returns null year with confidence none
- [x] [P6-T5] Add unit test: cache hit returns cached result and skips network call (mock client)
- [x] [P6-T6] Add unit test: fallback enabled returns result from fallback source and tags original_pub_source
- [x] [P6-T7] Add integration test (mocked APIs): batch run over fixture parquet respects rate-limit, writes new columns, and emits summary counts
- [x] [P6-T8] Add integration test (mocked APIs): checkpoint resume continues after interruption without duplicating work
- [x] [P6-T9] Add CLI test (mocked APIs): flags parse correctly, output path honored, and summary printed

**Phase 7 — Docs & Examples**
- [x] [P7-T1] Add CLI usage example and flag table (input/output, checkpoint, rate-limit, cache, fuzzy controls) to README or feature doc
- [x] [P7-T2] Document confidence bands, null-handling, and issued_date preservation in feature doc section
- [x] [P7-T3] Add example summary output snippet (matched high/low/none, errors) and sample command

**Phase 8 — QA: Toolchain Loop**
- [x] [P8-T1] Run Black on touched files (`poetry run black .`)
- [x] [P8-T2] Run Ruff (`poetry run ruff check`); if changes occur, rerun Black then repeat loop
- [x] [P8-T3] Run Pyright (`poetry run pyright`); fix issues and restart loop from Black if needed
- [x] [P8-T4] Run Pytest with coverage target ≥90% for new modules; fix and restart loop from Black if needed
- [x] [P8-T5] Run JSON format check (`poetry run python -m scripts.dev_tools.format_json --check`); if it would reformat, run without --check and restart loop from Black
- [x] [P8-T6] Run JSON schema validation (`poetry run python -m scripts.dev_tools.validate_json`); fix and restart loop from Black if it fails
- [x] [P8-T7] Run PowerShell format (`pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "& { Import-Module './scripts/powershell/PoshQC'; Invoke-PoshQCFormat -Root '.' }"`) if any PowerShell files were touched; otherwise record N/A
- [x] [P8-T8] Run PowerShell analyze (`pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "& { Import-Module './scripts/powershell/PoshQC'; Invoke-PoshQCAnalyze -Root '.' }"`) if any PowerShell files were touched; otherwise record N/A
- [x] [P8-T9] Run PowerShell tests (`pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "& { Import-Module './scripts/powershell/PoshQC'; Invoke-PoshQCTest -Root '.' }"`) if any PowerShell files were touched; otherwise record N/A
- [x] [P8-T10] Report final toolchain pass status and coverage for new code

**Phase 9 — Validation & Rollback Readiness**
- [ ] [P9-T1] Smoke-test CLI on a small sample parquet with mocked or cached responses to confirm end-to-end behavior
- [ ] [P9-T2] Verify output parquet schema compatibility with existing consumers and document rollback: keep original parquet untouched when output path differs; backup if in-place

## Test Plan

- Unit: normalization, matching (exact/fuzzy/none), missing year handling, cache hits, fallback tagging
- Integration (mocked APIs): batch run with throttling, summary counts, checkpoint resume
- CLI: flag parsing, output path application, summary emission

## Open Questions / Notes
- Confirm cache directory naming convention; align with existing pipeline cache paths if any
- Confirm default rate-limit/batch sizes acceptable for Open Library and CI runtimes
