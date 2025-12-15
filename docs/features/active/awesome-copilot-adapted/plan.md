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

> **Instructions for this section:**
> - Break work into **Phases** (broad buckets) and **Atomic Tasks** (binary, 5-30 min units).
> - Use `- []` for every task (no `- [ ]`).
> - Start every task with a **strong verb** (Implement, Create, Update, Verify).
> - No "bucket" tasks like "Refactor module" or "Write tests"; split them into specific, verifiable steps.

### Phase 1: Source & Attribution Blueprint
- [ ] Inventory upstream `awesome-copilot` agents to target (names, paths, licenses)
- [ ] Capture upstream MIT license text and source URLs for each selected agent
- [ ] Draft THIRD_PARTY_NOTICES entry format for adapted agents (repo + file paths + commit/URL)
- [ ] Define per-agent header snippet for provenance + repo-policy precedence

### Phase 2: Adapt Pilot Agents with Guardrails
- [ ] Import one pilot agent into `.github/agents/` (no behavior changes yet)
- [ ] Apply repo-policy compliance section and remove conflicting instructions (no auto `.env`, no temp files in tests, enforce toolchain loop)
- [ ] Add provenance note (“Adapted from awesome-copilot …”) to the pilot agent
- [ ] Document a short “Add another upstream agent” checklist in `docs/features/active/awesome-copilot-adapted/`

### Phase 3: Validate & Roll Out
- [ ] Add Pester checks ensuring adapted agents include repo-policy precedence and provenance note
- [ ] Verify potential-to-issue workflow still promotes potential docs after agent updates
- [ ] Update active feature docs (spec, user-story) with owner/status/links; ensure backlog and potential references point to issue #42
- [ ] Present rollout notes (where to place new agents, how to run checks) in README/docs reference section

## Test Plan

- Unit: Pester tests for agent file headers (repo-policy precedence + provenance), update potential-to-issue tests if needed
- Integration: Dry-run agent adoption workflow on one pilot agent; ensure promotion script still succeeds with PromotionType
- Manual/CLI: Run `potential-to-issue.ps1` via task, confirm issue creation path; spot-check adapted agent instructions render correctly in VS Code

## Open Questions / Notes

- Which upstream commit/tag of `awesome-copilot` should we pin for provenance?
- Do we need language-specific variants beyond Python/PowerShell/Actions in this repo?
- Where should THIRD_PARTY_NOTICES live (root vs docs/) for easiest discoverability?

