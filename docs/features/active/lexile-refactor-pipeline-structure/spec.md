# lexile-refactor-pipeline-structure - Refactor Spec

- Issue: #15
- Parent Initiative (optional): #10
- Owner: drmoisan
- Last Updated: 2025-12-06

## Intent & Outcomes

Retire the ad-hoc `scripts/production` layout and house the Lexile-faithful pipeline scripts under `src/` so they live with the rest of the package. Outcomes: clear ownership under the epic (#10), consistent imports/packaging, stable tooling (pyright/ruff/pytest), and discoverable entry points for corpus/analyzer/calibration flows without changing runtime behavior.

## Invariants (must not change)

- CLI surfaces and flags for corpus/analyzer/calibration/lexile-scoring-model-pipeline flows remain identical (legacy alias: text-difficulty-pipeline).
- Input/output file shapes and default data locations (e.g., `data/corpus/*`, `data/freq/word_frequencies.tsv`, `data/calibration/*`, `data/model/lexile_regression_model.json`) remain the same.
- Pipeline determinism and processing logic stay unchanged; only module paths and imports move.
- Existing VS Code launch/task behaviors are preserved (commands still runnable, even if the underlying path moves).
- Logging/telemetry (if any) unchanged; no new side effects.

## Scope (structural changes)

- Move `scripts/production/*` into `src/lexile_corpus_tuner/pipeline_scripts/` with `__init__.py` as needed; update imports to package-style absolute imports.
- Align module names/entry points to the existing CLI (`lexile_corpus_tuner.cli` lexile-scoring-model group; alias text-difficulty) and any UI modules (e.g., Gutenberg query builder) under the package.
- Update VS Code launch/task references, docs, and automation to point to the new paths.
- Retire `scripts/production` (no shims); all entry points live under the package.

## Non-Goals

- No new behavior, flags, or UX changes.
- No data/model format changes or performance tuning.
- No changes to acceptance criteria of child features (#11/#12/#13) beyond path alignment.

## Dependencies / Touchpoints

- VS Code tasks/launch configs referencing `scripts/production` paths.
- Any docs or README snippets pointing to `scripts/production/*`.
- Downstream imports/tests that reach into `lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts` modules.
- Packaging/entry-points (CLI) that may assume module locations.

## Risks & Mitigations

- Risk: import/relative-path breakage after moves. Mitigation: use absolute package imports; run pyright/pytest; add `__init__.py` where needed.
- Risk: VS Code tasks/launch or docs still pointing to old paths. Mitigation: search/update all `scripts/production` references; optionally add stubs that print the new invocation.
- Risk: tooling (ruff/pyright) or packaging missing new modules. Mitigation: update package init and ensure moved modules are within `src/` path; run lint/type checks.
- Risk: hidden data-path assumptions inside scripts. Mitigation: smoke the pipeline CLIs end-to-end after moves.

## Definition of Done

- [x] Structure matches this spec; legacy paths retired
- [x] Behavior unchanged (validated against invariants)
- [x] Imports/tooling/entry points updated
- [x] Tests and type checks clean
- [x] Docs updated (initiative/README/tasks as needed)



