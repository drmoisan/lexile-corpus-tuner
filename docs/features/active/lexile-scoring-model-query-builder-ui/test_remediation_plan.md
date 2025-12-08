# Test Remediation Plan – gutenberg_query_builder_ui

- [ ] Objective: Restore the gutenberg_query_builder_ui test suite to full policy compliance with ≥90% coverage on every package file, zero unauthorized `type: ignore` usage, clean lint/type checks, and all tests passing, while reorganizing tests to mirror the package structure and maintaining independence/determinism.

## Phase 1 – Baseline and Gap Analysis

- [ ] Confirm scope against current failures: 23 pytest failures across 519 tests, Ruff (3 errors), Pyright (75 errors plus unauthorized type ignores), and coverage at 68% overall for the package.
- [ ] Inventory policy gaps: locate unauthorized `type: ignore` comments (tests and src), shared mocks that leak across tests, and any temporary-file or external dependency usage.
- [ ] Produce a short baseline report capturing failing test names, affected modules, current coverage per file (constants.py, tk_helpers.py, widgets.py, app.py, __init__.py, __main__.py), and existing lint/type errors to track closure.
- [ ] Deliverables: Baseline findings doc (failures, lint/type errors, coverage by file, list of policy violations).

## Phase 2 – Test Architecture Refactor (Mirror Package)

- [ ] Restructure tests to align 1:1 with modules (e.g., `tests/gutenberg_query_builder_ui/test_constants.py`, `.../test_tk_helpers.py`, `.../test_widgets.py`, `.../test_app.py`, plus minimal `test_init_main.py` as needed). Remove monolithic overlap between existing component and coverage files.
- [ ] Introduce scoped fixtures to isolate tkinter/pandas patching and sys.modules manipulation; ensure each test/module restores state and keeps mocks local (no module-level monkeypatch). Avoid shared mutable globals; prefer factory helpers per test.
- [ ] Extract common sample data/builders (e.g., constraint/group models, DataFrame rows) into fixtures with function or module scope to keep tests small, deterministic, and independent.
- [ ] Remove any use of temporary files; stub file dialogs and filesystem access via mocks only.
- [ ] Create a baseline-to-new-structure mapping document that ties each Phase 1 finding to its resolution in the refactored test layout/fixtures.
- [ ] Deliverables: New test layout matching package structure, shared fixtures module with scope rules documented, removal of cross-test coupling.

## Phase 3 – Behavior Coverage and Defect Remediation

- [ ] Triage the 23 failing tests to distinguish true product defects from test issues; fix real defects in src modules if confirmed, otherwise correct the tests. Keep fixes minimal, typed, and covered by targeted unit tests.
- [ ] Expand/adjust tests to reach ≥90% line coverage per file:

  - [ ] constants.py: constant values and field/operator maps.
  - [ ] tk_helpers.py: happy/error paths for wrapper functions, especially pandas/tkinter shims.
  - [ ] widgets.py: constraint/operator selection, value parsing/validation, tooltip wiring, group add/delete/nesting, model conversion, and callbacks.
  - [ ] app.py: menu actions (new/open/save/save as/export), query execution flow, status updates, result display limits, file-missing/invalid query handling, and integration with BooleanQueryEngine stubs.
  - [ ] __init__.py/__main__.py: import/export surface and entry point wiring.
- [ ] Eradicate unauthorized `type: ignore` usage: retain only tkinter/pandas ignores placed in tk_helpers with inline justification per policy; remove or replace others via typing-friendly refactors. Ensure tests rely on typing-friendly stubs instead of ignores.
- [ ] Ensure every test documents intent (names/docstrings), follows Arrange-Act-Assert, and produces clear failure messages.
- [ ] Deliverables: Updated tests with ≥90% coverage per package file, resolved failing tests/defects, zero unauthorized `type: ignore`, and documented scenario coverage per module.

## Phase 4 – Quality Gates and Validation

- [ ] Run the full toolchain loop until clean: `black`, `ruff`, `pyright`, then `pytest --cov=lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui --cov-report=term-missing`.
- [ ] Enforce coverage verification per file (fail if any file <90%); capture final coverage report.
- [ ] Confirm tests are order-independent (e.g., via `pytest --maxfail=1 --disable-warnings -q` with random-order plugin if available or manual reordering checks).
- [ ] Deliverables: Final toolchain pass log (all four steps green in one pass) and per-file coverage report meeting the 90% threshold.

## Phase 5 – Compliance Documentation and Handoff

- [ ] Produce a detailed compliance document (see mapping below) explaining how the remediated suite satisfies every requirement in `general-unit-test.instructions.md` and `python-unit-test.instructions.md`, including independence, isolation, determinism, coverage, structure, naming, mocking discipline, and use of Pytest.
- [ ] Document type-ignore handling (locations, justifications, tkinter/pandas-only) and any approved exceptions (expected to be none beyond policy allowances).
- [ ] Summarize remaining risks or follow-ups (if any) and the final test layout for maintainers.
- [ ] Deliverables: Completed compliance document, final coverage summary per file, and handoff notes.

## Compliance Mapping (to be finalized after remediation)

- [ ] General Unit Test Policy

  - [ ] Core Principles (independence, isolation, fast, deterministic, readable): demonstrate fixtures/local mocks, no shared state, and concise assertions.
  - [ ] Coverage and Scenarios: show ≥90% per module with positive/negative/edge/error paths enumerated.
  - [ ] Test Structure and Diagnostics: confirm Arrange-Act-Assert usage and descriptive names/docstrings.
  - [ ] External Dependencies and Environment: evidence of mocked file dialogs/engines, zero network/disk writes, no temporary files.
  - [ ] Policy Audit: note final self-review against the policy and absence of exceptions.
- [ ] Python Unit Test Policy

  - [ ] Framework and Scope: Pytest-only execution and coverage targets met.
  - [ ] Test Style and Structure: focused tests per function/method, minimal mocking, fixtures scoped narrowly.
  - [ ] Naming and Readability: `test_...` naming plus intent docstrings where needed.
  - [ ] Toolchain Loop: confirmation that the final pass ran Black → Ruff → Pyright → Pytest in order without errors.
- [ ] Audit Note: Plan reviewed twice for alignment with general-code-change, python-code-change, general-unit-test, and python-unit-test instructions; no conflicts identified.
