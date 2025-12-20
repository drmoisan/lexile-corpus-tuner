# 2025-12-18-docs-v3-upgrade - Plan

- Issue: #54
- Owner: 2025-12-18-docs-v3-upgrade
- Last Updated: 2025-12-19

## Required References

- General Coding Standards: [`.github/instructions/general-code-change.instructions.md`](../../../../.github/instructions/general-code-change.instructions.md)
- General Unit Test Policy: [`.github/instructions/general-unit-test.instructions.md`](../../../../.github/instructions/general-unit-test.instructions.md)
- (Add language-specific policies as needed, e.g. `python-code-change.instructions.md`)

**All work must comply with these policies; do not duplicate their content here.**

## Implementation Plan (Atomic Tasks)

### Phase 0 — Context & Inputs
- [x] [P0-T1] Re-read chat-history.md to capture required behaviors (issue/PR separation, GH validation, feature-doc excerpts, PR-intent scaffold, deterministic autoclose).
- [x] [P0-T2] Re-read user-story.md and spec.md to align goals, acceptance criteria, and non-goals.
- [x] [P0-T3] Identify current versions/paths of generate-pr.prompt.md, pr-author.agent.md, and scripts/dev_tools/collect_pr_context.py to be updated.

### Phase 1 — Collector: issue/PR validation and autoclose
- [x] [P1-T1] Implement GH-aware classification that separates merged PRs (from merge commits) from issue references in pr_context output.
- [x] [P1-T2] Add verified “Issues to autoclose” block using gh closingIssuesReferences or gh issue view; when offline/unauthed, emit only unverified references and suppress autoclose claims.
- [x] [P1-T3] Ensure referenced issues vs non-closing references are emitted in distinct buckets, never mixing merge-PR numbers into issues.
- [ ] [P1-T4] Delete `scripts/dev_tools/collect_pr_context.py` and redirect any tasks, testing, or other tooling to the pr_context package.
- [ ] [P1-T5] Delete `scripts/dev_tools/pr_context_gh.py` and redirect any tasks, testing, or other tooling to the pr_context package.
- [ ] [P1-T6] Delete `scripts/dev_tools/pr_context_git.py` and redirect any tasks, testing, or other tooling to the pr_context package.
- [ ] [P1-T7] Delete `scripts/dev_tools/pr_context_models.py` and redirect any tasks, testing, or other tooling to the pr_context package.
- [ ] [P1-T8] Delete `scripts/dev_tools/pr_context_render.py` and redirect any tasks, testing, or other tooling to the pr_context package.
- [ ] [P1-T9] Regenerate pr_context.txt and confirm #51/#52/#53 are classified as merged PRs (not issues) and that #50/#44 appear when verified for autoclose.

### Phase 2 — Collector: feature-doc and intent embedding
- [ ] [P2-T1] If one or more active user-story.md files exist, extract the "Story Statement" bullets to support the explanation of the desired end state.
- [ ] [P2-T2] If one or more active user-story.md files exist, extract the "Problem / Why" section to support the why.
- [ ] [P2-T3] If no active user-story.md file exists, please search for an active file in `docs/features/potential/promoted` and extract the problem from either a section labeled "Problem / Why" (for features) or a section labeled "Summary" (for bugs). 
- [ ] [P2-T4] Extract spec.md sections (Context, Root Cause/Problem, Proposed Fix, Acceptance Criteria) into pr_context as concise excerpts.
- [ ] [P2-T5] Extract plan.md completed tasks/verification notes into pr_context to support “Why/Verification”.
- [ ] [P2-T6] Add PR-intent scaffold fields (primary outcome, impact, risks, asserted autoclose issues) to pr_context for author completion.
- [ ] [P2-T7] Add an enumerated “Additional Context Files” list in pr_context to authorize PR author access only to embedded excerpts and listed files.

### Phase 3 — Prompt and agent alignment
- [ ] [P3-T1] Update generate-pr.prompt.md to consume only pr_context and listed additional files; enforce `## GitHub Auto-close` with `- Closes #N` lines; forbid guessing when unverified.
- [ ] [P3-T2] Update pr-author.agent.md to reject merge-PR numbers as issues, use only verified autoclose lists, and route non-closing refs to `## Related`.
- [ ] [P3-T3] Define offline/no-gh handling in prompt/agent: omit autoclose section or mark references as unverified; never emit `Closes` without verification.
- [ ] [P3-T4] Instruct agent to use embedded feature-doc excerpts and PR-intent fields as the sole “Why” sources (no external guessing).

### Phase 4 — Testing and verification
- [ ] [P4-T1] Add unit tests for collector issue/PR classification with merge commits present to ensure PR numbers are excluded from issues.
- [x] [P4-T2] Add unit tests for GH-validated autoclose list (online) and fallback behavior (offline) to ensure no autoclose claims when unverified.
- [ ] [P4-T3] Add unit tests for feature-doc extraction and PR-intent scaffold inclusion in pr_context.
- [ ] [P4-T4] Integration test end-to-end pr_context generation across scenarios (feature, bugfix, docs-only) confirming deterministic autoclose lists and embedded excerpts.
- [ ] [P4-T5] Dry-run PR author flow with updated prompt/agent to verify sections (Why, Verification, GitHub Auto-close, Related) render correctly from new pr_context.

### Phase 5 — Governance, rollout, and telemetry
- [ ] [P5-T1] Document governance cadence (owners, audit schedule, escalation) and storage location for pr_context artifacts.
- [ ] [P5-T2] Define telemetry/metrics collection for template usage, PR summary completeness, and reviewer satisfaction; specify reporting path to governance owners.
- [ ] [P5-T3] Publish contributor-facing quick-start/checklist updates covering new pr_context fields, autoclose rules, and use of embedded excerpts.
- [ ] [P5-T4] Update spec.md/user-story.md status fields and dates once implementation milestones complete.

## Test Plan

- Unit: Collector functions for issue/PR classification, gh fallbacks, feature-doc extraction, PR-intent inclusion; prompt/agent linting if automated.
- Integration: End-to-end pr_context generation with gh available/unavailable; PR body generation consuming enumerated context files.
- Manual/CLI: Spot-check pr_context.txt for mislabeling (#51/#52/#53) and verified autoclose (#44/#50); run PR author flow to confirm sections (Why, GitHub Auto-close, Related) populate correctly.

## Open Questions / Notes

- Which metrics and telemetry endpoints will be used to report template usage, PR summary completeness, and reviewer satisfaction?
- What is the fallback behavior for environments without gh authentication—omit autoclose entirely or include an explicit “unverified” note?
- Do governance owners require additional audit artifacts beyond pr_context.txt (e.g., saved PR bodies, survey exports)?

---
<small>Generated with GitHub Copilot as directed by {USER_NAME_PLACEHOLDER}</small>

