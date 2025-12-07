# <refactor-name> - Refactor Plan

- Issue: #<id>
- Parent Initiative (optional): #<parent-id>
- Owner: name
- Last Updated: YYYY-MM-DD

## Required References (read, do not restate)

- Coding workflow and standards: [`docs/code-change.instructions.md`](../../code-change.instructions.md)
- Unit test policy: [`docs/unit-test-policy.md`](../../unit-test-policy.md)

## Strategy

Brief approach to reach the target structure while keeping behavior stable.

## Work Breakdown

### Phase 1: Inventory & Plan [0%]
- [ ] Enumerate current entry points/paths/imports to touch
- [ ] Confirm invariants and non-goals

### Phase 2: Execute Structural Changes [0%]
- [ ] Apply moves/renames to reach the target layout
- [ ] Update imports/tooling/entry points
- [ ] Remove or redirect legacy paths

### Phase 3: Verification & Cleanup [0%]
- [ ] Run tests/type checks; fix fallout
- [ ] Update docs/tasks/initiative references
- [ ] Final pass for stray references to old locations

## Test Plan

- Unit/Integration: impacted modules and any regression tests for invariants
- CLI/Workflow: end-to-end commands/tasks expected to remain stable
- Tooling: lint/type checks after path updates

## Rollback / Contingency

How to revert or isolate if the refactor breaks downstream consumers (e.g., keep branch snapshot, git move plan).

## Open Questions / Notes

Capture decisions, risks, and follow-ups.
