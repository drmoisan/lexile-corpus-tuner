# pipeline-structure (Issue #15)

- Date captured: 2025-12-05
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/pipeline-structure/ (Issue #15)

- Issue: #15
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/15
- Last Updated: 2025-12-06
## Problem / Why

Pipeline scripts for the Lexile-faithful text difficulty epic live in a `production` folder outside `src`, so they are not clearly associated with the initiative, don’t reuse package structure, and make imports/tooling brittle. This structure obscures ownership (initiative #10), slows onboarding, and complicates packaging, tests, and reuse across corpus/analyzer/calibration layers.

## Proposed Behavior

Consolidate the pipeline scripts under `src/` (aligned to the `lexile-faithful-text-difficulty-pipeline` initiative), with a clear package/module layout and updated imports/entry points. Preserve behavior/CLIs while making the code discoverable, testable, and ready for packaging and future maintenance. Use `lexile-scoring-model-pipeline` as the entry point (keep `text-difficulty-pipeline` as a legacy alias if needed during transition).

## Acceptance Criteria (early draft)

- [ ] `production/` is retired; scripts live under `src/lexile_corpus_tuner/...` with package init in place.
- [ ] All moved scripts keep identical runtime behavior/CLIs; commands still reachable via entry points/tasks (`lexile-scoring-model-pipeline`, legacy alias `text-difficulty-pipeline`).
- [ ] Imports/config paths updated; Black/Ruff/Pyright/Pytest clean after the move.
- [ ] Initiative docs point to the new structure (no stale references to `production/`).

## Constraints & Risks

- No behavior changes; scope is structural only (moves/renames, imports, minimal glue).
- Must avoid breaking downstream consumers, automation, or data paths used by corpus/analyzer/calibration CLIs.
- Watch for implicit relative paths and hard-coded locations that may shift when moving into `src`.

## Test Conditions to Consider

- [ ] Run existing unit/integration/feature tests; verify Pyright after import path updates.
- [ ] Full pipeline CLI smoke: `corpus download/normalize/frequencies`, `calibration build-dataset/fit`, `analyze text`.
- [ ] VS Code tasks/entry points still resolve the commands under the new module layout.

## Next Step

- [ ] Promote to GitHub issue (feature request template)
- [ ] Create `docs/features/active/pipeline-structure/` folder from the template


