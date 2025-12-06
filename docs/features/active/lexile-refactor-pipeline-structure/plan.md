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

### Phase 1: Inventory & Plan [0%]
- [ ] Enumerate all `scripts/production/*` scripts, entry points, and their imports
- [ ] Grep for references to `scripts/production` in tasks/docs/tests/launch configs
- [ ] Confirm invariants and non-goals with issue #10/#15 context

### Phase 2: Execute Structural Changes [0%]
- [ ] Create target package layout under `src/lexile_corpus_tuner/<pipeline_subdir>/` with `__init__.py`
- [ ] `git mv` scripts into the new layout; convert relative imports to package imports
- [ ] Update VS Code launch/tasks and any docs referencing old paths
- [ ] Remove or add stub redirects in `scripts/production` (if needed) with guidance to new commands

### Phase 3: Verification & Cleanup [0%]
- [ ] Run Ruff/Pyright/Pytest; fix import/path fallout
- [ ] Smoke CLI: corpus download/normalize/frequencies; calibration build-dataset/fit; analyze text (sanity run)
- [ ] Update initiative/task/docs references; final scan for `scripts/production` references

## Test Plan

- Unit/Integration: existing suites must pass after import/path updates
- CLI/Workflow: `lexile_corpus_tuner pipeline_entry` commands for corpus/analyzer/calibration and the text-difficulty pipeline workflow
- Tooling: `poetry run ruff check`, `poetry run pyright`, `poetry run pytest`

## Rollback / Contingency

- Keep branch/PR scoped; `git mv` history makes rollback straightforward
- If needed, retain temporary stubs in `scripts/production` that import from new locations until downstream callers are updated

## Open Questions / Notes

- Do we need temporary stubs in `scripts/production` for downstream consumers, or can we remove the folder entirely once tasks/docs are updated?

