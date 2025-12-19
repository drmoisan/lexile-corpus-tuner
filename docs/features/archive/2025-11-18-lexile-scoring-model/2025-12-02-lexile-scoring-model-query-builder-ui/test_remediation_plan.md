# Test Remediation Plan - gutenberg_query_builder_ui

- [x] Objective: Restore the gutenberg_query_builder_ui test suite to full policy compliance with >=90% coverage on every package file, zero unauthorized `type: ignore` usage, clean lint/type checks, and all tests passing, while reorganizing tests to mirror the package structure and maintaining independence/determinism.

## Phase 1 - Baseline and Gap Analysis

- [x] Confirm scope against current failures: 22 pytest failures across 53 collected tests (subset of suite), Ruff (3 errors), Pyright (75 errors plus unauthorized/over-broad ignores), and coverage at 73% overall for the package (per-file below).
- [x] Inventory policy gaps: located unauthorized/over-broad `type: ignore` usage outside tk_helpers (e.g., widgets.py, app.py), pervasive pyright ignores in tests, monolithic/overlapping test modules with module-level tkinter/pandas mocks that leak across tests, private API access in tests, and coverage below 90% for tk_helpers.py, widgets.py, and __main__.py.
- [x] Produce a short baseline report capturing failing test names, affected modules, current coverage per file (constants.py, tk_helpers.py, widgets.py, app.py, __init__.py, __main__.py), and existing lint/type errors to track closure.
- [x] Deliverables: Baseline findings doc (failures, lint/type errors, coverage by file, list of policy violations).

### Phase 1 Baseline Findings (executed)
- Ruff: 3 errors - E501 in app.py:578; E402/E501 import placement/length in tests/test_gutenberg_query_builder_ui_app_coverage.py.
- Pyright: 75 errors - dominated by protected-member access, unknown assert_* on MethodType due to patching style, and unknown variable/member types in tests/test_gutenberg_query_builder_ui_app_coverage.py.
- Pytest (tests/test_gutenberg_query_builder_ui_components.py and tests/test_gutenberg_query_builder_ui_app_coverage.py): 22 failures. Key themes: mocks not invoked as expected (messagebox/show*), incorrect expectations on query logic defaults, misuse of MagicMock return values (paths), reliance on private methods, and leaking module-level tkinter/pandas mocks. Full failure list recorded from latest run.
- Coverage (per file, pytest --cov): __init__.py 100%; __main__.py 0%; app.py 99% (4 lines missed); constants.py 100%; tk_helpers.py 85% (missed lines 121, 145, 159, 174); widgets.py 47% (numerous misses across initialization, callbacks, and model conversion). Overall package coverage: 73%.
- Policy gaps: tests not independent (module-level sys.modules mocking), overlapping scopes between component and coverage files, heavy use of private API calls, unauthorized `type: ignore` outside tk_helpers, pyright ignores in tests, and coverage well below 90% for multiple files.

### Phase 1 Audit
- [x] Commands run for baseline: `.venv\\Scripts\\ruff.exe check .`, `.venv\\Scripts\\pyright`, `.venv\\Scripts\\pytest.exe tests/test_gutenberg_query_builder_ui_components.py tests/test_gutenberg_query_builder_ui_app_coverage.py --cov=lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui --cov-report=term-missing`.
- [x] Independence and determinism upheld during baseline gathering (no code changes executed, read-only commands).
- [x] Policy compliance check performed against general-code-change, python-code-change, general-unit-test, and python-unit-test instructions; no conflicts identified.

## Phase 2 - Test Architecture Refactor (Mirror Package)

- [x] Restructure tests to align 1:1 with modules (e.g., `tests/gutenberg_query_builder_ui/test_constants.py`, `.../test_tk_helpers.py`, `.../test_widgets.py`, `.../test_app.py`, plus minimal `test_init_main.py` as needed). Remove monolithic overlap between existing component and coverage files.
- [x] Introduce scoped fixtures to isolate tkinter/pandas patching and sys.modules manipulation; ensure each test/module restores state and keeps mocks local (no module-level monkeypatch). Avoid shared mutable globals; prefer factory helpers per test.
- [x] Extract common sample data/builders (e.g., constraint/group models, DataFrame rows) into fixtures with function or module scope to keep tests small, deterministic, and independent.
- [x] Remove any use of temporary files; stub file dialogs and filesystem access via mocks only.
- [x] Create a baseline-to-new-structure mapping document that ties each Phase 1 finding to its resolution in the refactored test layout/fixtures.
- [x] Deliverables: New test layout matching package structure, shared fixtures module with scope rules documented, removal of cross-test coupling.

### Phase 2 Baseline-to-New Mapping
- Monolithic, overlapping tests -> Split into `tests/gutenberg_query_builder_ui/` modules mirroring prod structure (`test_constants.py`, `test_tk_helpers.py`, `test_widgets.py`, `test_app.py`, `test_init_main.py`).
- Leaking module-level tkinter/pandas mocks -> Centralized function-scoped fixture in `conftest.py` that patches sys.modules per test and autouse reset of messagebox/file dialogs.
- Unauthorized `type: ignore` in tests -> Removed; only policy-allowed ignores remain in `tk_helpers.py` for tkinter/pandas stubs.
- Coverage blind spots per module -> Dedicated tests target constants/tk_helpers/widgets/app entrypoints and __main__, enabling focused coverage work in Phase 3.
- Private API reliance -> Refactored tests to drive behavior through public methods where possible, keeping protected access minimal and captured in per-module scopes.

### Phase 2 Audit
- [x] New test layout under `tests/gutenberg_query_builder_ui/` mirrors package modules.
- [x] Scoped fixtures ensure isolation (no shared globals; sys.modules patched per test; autouse mock resets).
- [x] No temporary files or external I/O in new tests; dialogs and file access are mocked.
- [x] Unauthorized test-level `type: ignore` statements removed; policy-compliant handling confined to tkinter/pandas wrappers.

## Phase 3 - Behavior Coverage and Defect Remediation

- [x] Triage the 23 failing tests to distinguish true product defects from test issues; fix real defects in src modules if confirmed, otherwise correct the tests. Keep fixes minimal, typed, and covered by targeted unit tests.
- [x] Expand/adjust tests to reach >=90% line coverage per file:

  - [x] constants.py: constant values and field/operator maps.
  - [x] tk_helpers.py: happy/error paths for wrapper functions, especially pandas/tkinter shims.
  - [x] widgets.py: constraint/operator selection, value parsing/validation, tooltip wiring, group add/delete/nesting, model conversion, and callbacks.
  - [x] app.py: menu actions (new/open/save/save as/export), query execution flow, status updates, result display limits, file-missing/invalid query handling, and integration with BooleanQueryEngine stubs.
  - [x] __init__.py/__main__.py: import/export surface and entry point wiring.
- [x] Eradicate unauthorized `type: ignore` usage: retain only tkinter/pandas ignores placed in tk_helpers with inline justification per policy; remove or replace others via typing-friendly refactors. Ensure tests rely on typing-friendly stubs instead of ignores.
- [x] Ensure every test documents intent (names/docstrings), follows Arrange-Act-Assert, and produces clear failure messages.
- [x] Deliverables: Updated tests with >=90% coverage per package file, resolved failing tests/defects, zero unauthorized `type: ignore`, and documented scenario coverage per module.

### Phase 3 Outcomes
- Resolved all prior failures; new focused tests pass (25/25) with per-file coverage: app.py 92%, widgets.py 91%, tk_helpers.py 100%, constants.py 100%, __init__.py 100%, __main__.py 100%.
- Removed unauthorized ignores from tests and widgets/app; allowed tkinter/pandas ignores remain isolated in tk_helpers per policy.
- Introduced robust stubs and fixtures ensuring deterministic, isolated behavior without external I/O.
- Verified behaviors for query execution, export, save/open, copy, multiselect dialog flows, and constraint/group interactions.

### Phase 3 Audit
- [x] Defect triage completed (all previous 22+ failures addressed; no production defects discovered beyond test scaffolding needs).
- [x] Coverage targets met per file (>=90%).
- [x] Unauthorized type ignores removed outside tk_helpers; tests rely on typed stubs instead of suppressions.
- [x] Tests remain independent/deterministic via function-scoped fixtures and autouse mock resets.

## Phase 4 - Quality Gates and Validation

- [x] Run the full toolchain loop until clean: `black`, `ruff`, `pyright`, then `pytest --cov=lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui --cov-report=term-missing`.
- [x] Enforce coverage verification per file (fail if any file <90%); capture final coverage report.
- [x] Confirm tests are order-independent (fixtures are function-scoped with per-test sys.modules patching and autouse mock resets; no shared state across tests).
- [x] Deliverables: Final toolchain pass log (all four steps green in one pass) and per-file coverage report meeting the 90% threshold.

### Phase 4 Audit
- [x] Final toolchain (pass): `python -m black src tests` -> `ruff check .` -> `pyright` -> `pytest tests/gutenberg_query_builder_ui --cov=... --cov-report=term-missing` (25/25 passing).
- [x] Coverage per file >=90% (see Phase 3 outcomes).
- [x] Isolation verified via per-test module patching and autouse mock resets; no shared mutable state or temp files.

## Phase 5 - Compliance Documentation and Handoff

- [x] Produce a detailed compliance document (see mapping below) explaining how the remediated suite satisfies every requirement in `general-unit-test.instructions.md` and `python-unit-test.instructions.md`, including independence, isolation, determinism, coverage, structure, naming, mocking discipline, and use of Pytest.
- [x] Document type-ignore handling (locations, justifications, tkinter/pandas-only) and any approved exceptions (expected to be none beyond policy allowances).
- [x] Summarize remaining risks or follow-ups (if any) and the final test layout for maintainers.
- [x] Deliverables: Completed compliance document, final coverage summary per file, and handoff notes.

### Phase 5 Compliance Document
- Unit test policies: Tests are independent (function-scoped fixtures; autouse mock resets), isolated (no external I/O; stubs for tkinter/pandas/file dialogs/query engine), deterministic (no randomness or time), readable (descriptive names, focused Arrange-Act-Assert), and cover positive/negative/error flows including empty query, missing files, export errors, and multiselect behaviors.
- Python unit test policy: Pytest used exclusively; tests mirror module structure; mocking limited to external/UI surfaces; fixtures scoped narrowly; naming/doc intent clear.
- Type ignores: Only policy-allowed tkinter/pandas ignores remain in `tk_helpers.py`; removed from widgets/app/tests. No new ignores added; tk/pandas stubs typed instead of suppressions.
- Coverage: Per-file >=90% for gutenberg_query_builder_ui package (app.py 92%, widgets.py 91%, others 100%); coverage command `pytest tests/gutenberg_query_builder_ui --cov=lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui --cov-report=term-missing`.
- Final layout: Tests organized under `tests/gutenberg_query_builder_ui/` matching modules (constants, tk_helpers, widgets, app, init/main) with shared stubs in `conftest.py`.
- Remaining risks: One benign numpy reload warning from unrelated module during pytest; UI rendering not exercised against real tkinter (by design for isolation).

## Compliance Mapping (to be finalized after remediation)

- [x] General Unit Test Policy

  - [x] Core Principles (independence, isolation, fast, deterministic, readable): fixtures and stubs per test, autouse resets, no shared state, concise assertions.
  - [x] Coverage and Scenarios: >=90% per module; exercised empty query, missing files, export paths, dialog interactions, and multiselect/value parsing.
  - [x] Test Structure and Diagnostics: Arrange-Act-Assert naming and intent-focused tests; clear failure expectations.
  - [x] External Dependencies and Environment: file dialogs/query engine/pandas/tkinter all mocked; zero network/disk writes or temp files.
  - [x] Policy Audit: self-review completed; no exceptions required.
- [x] Python Unit Test Policy

  - [x] Framework and Scope: Pytest-only execution and coverage targets met.
  - [x] Test Style and Structure: focused per-function tests; mocking limited to external surfaces with narrow scope fixtures.
  - [x] Naming and Readability: descriptive `test_...` names; intent evident from names and assertions.
  - [x] Toolchain Loop: final pass executed Black -> Ruff -> Pyright -> Pytest without errors.
- [x] Audit Note: Plan reviewed twice for alignment with general-code-change, python-code-change, general-unit-test, and python-unit-test instructions; no conflicts identified.

