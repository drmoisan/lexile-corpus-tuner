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
- [ ] [P0-T1] Read .github/copilot-instructions.md and .github/instructions/general-code-change.instructions.md to confirm baseline repo rules
- [ ] [P0-T2] Read .github/instructions/general-unit-test.instructions.md, .github/instructions/python-code-change.instructions.md, and .github/instructions/python-unit-test.instructions.md to capture testing/tooling expectations referenced in acceptance criteria
- [ ] [P0-T3] Read docs/features/active/2025-12-24-original-publication-date-71/issue.md to extract issue #71 acceptance criteria and constraints
- [ ] [P0-T4] Read docs/features/active/2025-12-24-original-publication-date-71/user-story.md and spec.md to understand the current draft
- [ ] [P0-T5] Read artifacts/chats/251224 Corpus Design Summary.md to pull intended end-state behavior and motivations

**Phase 1 — Requirements Alignment**
- [ ] [P1-T1] Extract key requirements from the corpus design summary (API sources, confidence scoring, throttling, checkpointing, null-handling) into an internal note
- [ ] [P1-T2] Map issue #71 acceptance criteria to specific sections in user-story.md and spec.md to ensure every criterion has a destination
- [ ] [P1-T3] Identify gaps or inconsistencies between current drafts and the design summary (e.g., issued_date preservation, caching, fuzzy thresholds)

**Phase 2 — User Story Finalization**
- [ ] [P2-T1] Update Story Statement in user-story.md to reflect the finalized goal: add original_pub_year with confidence while preserving PG issued_date
- [ ] [P2-T2] Refine Personas in user-story.md to mirror constraints from the design summary (corpus engineer scaling concerns; data scientist weighting needs)
- [ ] [P2-T3] Refine Scenario in user-story.md to describe end-to-end enrichment run including throttling, checkpointing, summary reporting, and downstream usage
- [ ] [P2-T4] Align Acceptance Criteria in user-story.md with issue #71 and design summary (confidence flags, null-handling, checkpointing, tests)
- [ ] [P2-T5] Confirm Non-Goals in user-story.md exclude overwriting issued_date, full-date inference, and manual curation

**Phase 3 — Spec Finalization**
- [ ] [P3-T1] Update Behavior in spec.md to capture primary Open Library flow, optional fallbacks, match strategy, and null/default behavior
- [ ] [P3-T2] Finalize Inputs/Outputs in spec.md with required flags, defaults (including default output `gutenberg_books_enhanced.parquet`), and artifacts (parquet columns, checkpoint, cache, summary)
- [ ] [P3-T3] Refine CLI/API Surface in spec.md with exact command, flags (rate-limit, batch-size, retries, fuzzy controls, cache), and request shape
- [ ] [P3-T4] Detail Data & State in spec.md (normalization steps, selection rules, checkpoint/caching) to mirror the design summary
- [ ] [P3-T5] Strengthen Constraints & Risks in spec.md (rate limits, false positives, partial coverage, schema stability)
- [ ] [P3-T6] Enumerate Seeded Test Conditions in spec.md with concrete scenarios (exact vs fuzzy match, confidence scoring, nulls, mocked batch run with checkpoint resume)

**Phase 4 — Consistency & QA**
- [ ] [P4-T1] Cross-check user-story.md and spec.md for consistent terminology (original_pub_year, pub_year_confidence, issued_date), APIs, and flags
- [ ] [P4-T2] Verify Definition of Done in spec.md covers docs and tests and aligns with acceptance criteria
- [ ] [P4-T3] Proofread both documents for clarity, completeness, and adherence to scope of issue #71

## Test Plan

- Unit: Validate matching/confidence logic and null-handling examples cited in spec/user story once drafted
- Integration: Mocked API batch run with checkpoint resume over fixture parquet as outlined in spec
- Manual/CLI: Dry-run the documented CLI command against a small sample parquet to confirm flags, summary reporting, and that output lands in `gutenberg_books_enhanced.parquet` without overwriting the input

## Open Questions / Notes
- Are there repository conventions for caching location naming that should be mirrored here?
- Do we need to document telemetry/logging expectations in spec.md beyond summary counts?
