# awesome-copilot-adapted - Plan

- Issue: #42
- Owner: drmoisan
- Last Updated: 2025-12-15

## Required References

- General Coding Standards: [`.github/instructions/general-code-change.instructions.md`](../../../../.github/instructions/general-code-change.instructions.md)
- General Unit Test Policy: [`.github/instructions/general-unit-test.instructions.md`](../../../../.github/instructions/general-unit-test.instructions.md)
- (Add language-specific policies as needed, e.g. `python-code-change.instructions.md`)

**All work must comply with these policies; do not duplicate their content here.**

## Implementation Plan (Atomic Tasks)

### Phase 0 — Context & Inputs
- [x] [P0-T1] Read repo policy files listed in "Required References" and note any agent-specific constraints (temp files, toolchain loop, secrets)
- [x] [P0-T2] Review upstream awesome-copilot agent repository index to identify candidate agents and confirm license (MIT)

### Phase 1 — Source & Attribution Blueprint
- [x] [P1-T1] List selected upstream agents with source paths and retrieval commit/URL in a working note
- [x] [P1-T2] Copy the MIT license text and source URL for awesome-copilot into the working note for reuse in notices
- [x] [P1-T3] Decide THIRD_PARTY_NOTICES location/format for this repo and record the decision (include adapted file paths and source URL/SHA)
- [x] [P1-T4] Draft a standard per-agent provenance header snippet referencing THIRD_PARTY_NOTICES and upstream source

### Phase 2 — Adapt Pilot Agents with Guardrails
- [x] [P2-T1] Import one pilot agent from awesome-copilot into `.github/agents/` preserving upstream content
- [x] [P2-T2] Insert repo-policy precedence section into the pilot agent referencing required instruction files
- [x] [P2-T3] Remove or override upstream instructions that conflict with repo policy (e.g., auto `.env`, temp files in tests, skipping toolchain loop) and summarize changes
- [x] [P2-T4] Add the provenance header snippet with source path, retrieval date, and link to THIRD_PARTY_NOTICES
- [x] [P2-T5] Document an “Add another upstream agent” checklist in `docs/features/active/awesome-copilot-adapted/` for repeatable imports

### Phase 3 — Validate & Roll Out
- [x] [P3-T1] Add a Pester test that fails if any adapted agent file lacks the repo-policy precedence block
- [x] [P3-T2] Add a Pester test that fails if any adapted agent file lacks the provenance header pointing to THIRD_PARTY_NOTICES and the upstream source
- [x] [P3-T3] Dry-run `potential-to-issue.ps1` on a sample potential file to confirm promotion still works after agent changes (stubbed gh, verified cleanup)
- [x] [P3-T4] Update active docs (spec, user-story, plan) to reflect completed attribution work and ensure backlog/potential entries reference issue #42
- [x] [P3-T5] Add rollout notes in README or docs referencing where to place adapted agents and how to run the Pester checks

## Test Plan

- Unit: Pester tests for agent file headers (repo-policy precedence + provenance), update potential-to-issue tests if needed
- Integration: Dry-run agent adoption workflow on one pilot agent; ensure promotion script still succeeds with PromotionType
- Manual/CLI: Run `potential-to-issue.ps1` via task, confirm issue creation path; spot-check adapted agent instructions render correctly in VS Code

## Open Questions / Notes

- Which upstream commit/tag of `awesome-copilot` should we pin for provenance?
- Do we need language-specific variants beyond Python/PowerShell/Actions in this repo?
- Where should THIRD_PARTY_NOTICES live (root vs docs/) for easiest discoverability?

