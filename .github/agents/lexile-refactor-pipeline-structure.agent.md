# Agent 15 Instructions: lexile-refactor-pipeline-structure (Issue #15, Parent #10)

Read first: `.github/instructions/general-code-change.instructions.md`, `.github/instructions/python-code-change.instructions.md`, `.github/instructions/general-unit-test.instructions.md`, `.github/instructions/python-unit-test.instructions.md`. Follow repo quality gates (Black/Ruff/Pyright/Pytest).

Primary sources for this task:

- Refactor spec: `docs/features/active/lexile-refactor-pipeline-structure/spec.md`
- Refactor plan: `docs/features/active/lexile-refactor-pipeline-structure/plan.md`
- Parent initiative context: `docs/features/active/lexile-faithful-text-difficulty-pipeline/initiative.md`

Goal (from spec/plan):

- Move pipeline scripts from `scripts/production/` into `src/lexile_corpus_tuner/pipeline_scripts/` with proper package layout and imports (no stubs kept).
- Keep behavior, flags, IO paths, determinism identical. No UX/flag changes. Preserve VS Code task/launch usability.
- Update docs/tasks/automation to point to new paths; retire or stub old locations if needed.

Invariants (must hold):

- CLI surfaces/flags unchanged; same inputs/outputs and default data paths (`data/corpus/*`, `data/freq/word_frequencies.tsv`, `data/calibration/*`, `data/model/lexile_regression_model.json`).
- Deterministic processing remains; no new side effects/logging changes.
- Existing tasks/launch configurations still runnable (possibly via updated paths or thin stubs).

Scope (do these):

- Create target package layout under `src/lexile_corpus_tuner/pipeline_scripts/` with `__init__.py` as needed.
- `git mv`/relocate `scripts/production/*` into the package; convert imports to package-absolute.
- Update VS Code tasks/launch configs, docs, and references from `scripts/production` to new module paths.
- Retire `scripts/production` (remove remaining shims).

Non-goals (do NOT do):

- No new behavior, flags, UX changes, or performance tuning.
- No changes to data/model formats or acceptance criteria of child features (#11/#12/#13).

Working steps (align with plan):

1) Inventory: List all files under `scripts/production/` and grep for `scripts/production` references (tasks, docs, tests, launch configs). Note entry points/CLIs.
2) Move/rename: Create package subdirs; move scripts; add `__init__.py` where needed; update imports to package-absolute.
3) References: Update tasks (`.vscode/tasks.json`), launch configs (`.vscode/launch.json`), docs/README snippets, and any code/tests importing from old paths.
4) Cleanup: Remove or replace old scripts with stubs that print new usage (if required by plan); ensure no stale references remain.
5) Verification: Run Ruff, Pyright, and Pytest. Smoke CLI flows: corpus download/normalize/frequencies; calibration build-dataset/fit; analyze text (sanity inputs). Fix fallout.

Tests to run (per plan/spec):

- `poetry run ruff check`
- `poetry run pyright`
- `poetry run pytest`
- CLI smoke: `lexile_corpus_tuner pipeline_entry corpus download/normalize/frequencies`, `calibration build-dataset/fit`, `analyze text` (with sample inputs/configs).

Docs to update:

- `docs/features/active/lexile-refactor-pipeline-structure/*` if details shift.
- Parent initiative `docs/features/active/lexile-faithful-text-difficulty-pipeline/initiative.md` if paths/milestones need refresh.
- Any README/snippets referencing `scripts/production`.

Deliverable:

- Code/tree reflecting the new structure with imports fixed.
- Updated tasks/launch/docs, plus optional stubs if used.
- Passing lint/type/tests and noted smoke results. Document any deviations or remaining questions in the refactor plan notes section.
