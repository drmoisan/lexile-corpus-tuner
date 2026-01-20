# <feature-name> - Plan

- **Issue:** <issue>
- **Parent (optional):** <parent-id>
- **Owner:** <name>
- **Last Updated:** <yyyy-MM-ddTHH-mm>
- **Status:** <template>
- **Version:** <version_number>

## Required References

- General Coding Standards: [`.github/instructions/general-code-change.instructions.md`](../../../../.github/instructions/general-code-change.instructions.md)
- General Unit Test Policy: [`.github/instructions/general-unit-test.instructions.md`](../../../../.github/instructions/general-unit-test.instructions.md)
- (Add language-specific policies as needed, e.g. `python-code-change.instructions.md`)

**All work must comply with these policies; do not duplicate their content here.**

## Implementation Plan (Atomic Tasks)

> **Instructions for this section:**
> - Break work into **Phases** (broad buckets) and **Atomic Tasks** (binary, 5-30 min units).
> - Use `- [ ] [P#-T#]` for every task.
> - Start every task with a **strong verb** (Implement, Create, Update, Verify).
> - No "bucket" tasks like "Refactor module" or "Write tests"; split them into specific, verifiable steps.
> - **Self-Validating Phases:** Include necessary test creation/update tasks *within* the phase that implements the code. Do not defer verification to a final "Testing" phase.

### Phase 0: Compliance & Context
- [ ] [P0-T1] Confirm alignment with repo policies by reading `.github/instructions/general-code-change.instructions.md`, `.github/instructions/python-code-change.instructions.md`, `.github/instructions/general-unit-test.instructions.md`, and `.github/instructions/python-unit-test.instructions.md` before touching code
  - Acceptance: Development log contains policy review timestamp prior to Phase 1 commits

### Phase 1: <Phase Name>
- [ ] [P1-T1] <Atomic task with strong verb>
- [ ] [P1-T2] <Atomic task>
  - Preconditions: <optional>
  - Acceptance: <optional>

### Phase 2: <Phase Name>
- [ ] [P2-T1] <Atomic task>
- [ ] [P2-T2] <Atomic task>

## Test Plan

- Unit: ...
- Integration: ...
- Manual/CLI: ...

## Open Questions / Notes

- ...
