# lexile-refactor-pipeline-structure - Refactor Plan

- Issue: #15
- Parent Initiative (optional): #<parent-id>
- Owner: drmoisan
- Last Updated: 2025-12-06

## Required References (read, do not restate)

- Coding workflow and standards: [`docs/code-change.instructions.md`](../../code-change.instructions.md)
- Unit test policy: [`docs/unit-test-policy.md`](../../unit-test-policy.md)

## Strategy

Move the pipeline scripts from `scripts/production` into `src/lexile_corpus_tuner/...`, switch imports to package-absolute, and update tasks/launch/docs so callers use the new module paths. Keep behavior invariant; add stubs or clear redirects only if necessary.

## Work Breakdown

### Phase 1: Inventory & Plan [100%]
- [x] Enumerate all `scripts/production/*` scripts, entry points, and their imports
- [x] Grep for references to `scripts/production` in tasks/docs/tests/launch configs
- [x] Confirm invariants and non-goals with issue #10/#15 context

### Phase 2: Execute Structural Changes [100%]
- [x] Create target package layout under `src/lexile_corpus_tuner/pipeline_scripts/` with `__init__.py`
- [x] `git mv` scripts into the new layout; convert relative imports to package imports
- [x] Update VS Code launch/tasks and any docs referencing old paths
- [x] Remove legacy `scripts/production` content (no shims retained)

### Phase 3: Verification & Cleanup [90%]
- [x] Run Ruff/Pyright/Pytest; fix import/path fallout
- [ ] Smoke CLI: corpus download/normalize/frequencies; calibration build-dataset/fit; analyze text (sanity run)
- [x] Update initiative/task/docs references; final scan for `scripts/production` references

## Test Plan

- Unit/Integration: existing suites must pass after import/path updates
- CLI/Workflow: `lexile_corpus_tuner pipeline_entry` commands for corpus/analyzer/calibration and the text-difficulty pipeline workflow
- Tooling: `poetry run ruff check`, `poetry run pyright`, `poetry run pytest`
- Completed: Ruff/Pyright/Pytest passing (466 tests). CLI smoke pending (data-dependent).

## Rollback / Contingency

- Keep branch/PR scoped; `git mv` history makes rollback straightforward
- If needed, retain temporary stubs in `scripts/production` that import from new locations until downstream callers are updated

## Open Questions / Notes

- `scripts/production` fully retired; callers must use package entry points.

