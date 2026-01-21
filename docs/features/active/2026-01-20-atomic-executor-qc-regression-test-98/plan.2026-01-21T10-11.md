# 2026-01-20-atomic-executor-qc-regression-test (Plan)

- **Issue:** #98
- **Parent (optional):** none
- **Owner:** drmoisan
- **Last Updated:** 2026-01-21T10-11
- **Status:** Planned
- **Version:** 0.1

This plan implements the spec at:
`docs/features/active/2026-01-20-atomic-executor-qc-regression-test-98/spec.md`.

Notes:
- The fix is intentionally plan-driven: only explicitly referenced tests may be treated as expected failures.
- The executor must apply the same plan-aware pytest gating in both:
	- Pre-flight QC (`scripts/dev_tools/atomic_executor/cli.py`), and
	- Phase-end QC (`scripts/dev_tools/atomic_executor/qc_runner.py`).

### Phase 0 — Context & Inputs
- [ ] [P0-T1] Read `.github/copilot-instructions.md` and record the policy source paths in `artifacts/atomic_executor_98_policy_sources.txt`
	- Acceptance: `artifacts/atomic_executor_98_policy_sources.txt` contains the exact string `.github/copilot-instructions.md`
- [ ] [P0-T2] Read `.github/instructions/general-code-change.instructions.md` and append its path to `artifacts/atomic_executor_98_policy_sources.txt`
	- Acceptance: `artifacts/atomic_executor_98_policy_sources.txt` contains the exact string `.github/instructions/general-code-change.instructions.md`
- [ ] [P0-T3] Read `.github/instructions/general-unit-test.instructions.md` and append its path to `artifacts/atomic_executor_98_policy_sources.txt`
	- Acceptance: `artifacts/atomic_executor_98_policy_sources.txt` contains the exact string `.github/instructions/general-unit-test.instructions.md`
- [ ] [P0-T4] Read `.github/instructions/python-code-change.instructions.md` and append its path to `artifacts/atomic_executor_98_policy_sources.txt`
	- Acceptance: `artifacts/atomic_executor_98_policy_sources.txt` contains the exact string `.github/instructions/python-code-change.instructions.md`
- [ ] [P0-T5] Read `.github/instructions/python-unit-test.instructions.md` and append its path to `artifacts/atomic_executor_98_policy_sources.txt`
	- Acceptance: `artifacts/atomic_executor_98_policy_sources.txt` contains the exact string `.github/instructions/python-unit-test.instructions.md`
- [ ] [P0-T6] Read the approved spec `docs/features/active/2026-01-20-atomic-executor-qc-regression-test-98/spec.md` and record its path in `artifacts/atomic_executor_98_inputs.txt`
	- Acceptance: `artifacts/atomic_executor_98_inputs.txt` contains the exact string `docs/features/active/2026-01-20-atomic-executor-qc-regression-test-98/spec.md`
- [ ] [P0-T7] Capture baseline Ruff output to `artifacts/atomic_executor_98_baseline_ruff.txt`
	- Acceptance: command `poetry run ruff check > artifacts/atomic_executor_98_baseline_ruff.txt 2>&1` exits 0 OR non-0, and `artifacts/atomic_executor_98_baseline_ruff.txt` is created
- [ ] [P0-T8] Capture baseline Pyright output to `artifacts/atomic_executor_98_baseline_pyright.txt`
	- Acceptance: command `poetry run pyright > artifacts/atomic_executor_98_baseline_pyright.txt 2>&1` exits 0 OR non-0, and `artifacts/atomic_executor_98_baseline_pyright.txt` is created
- [ ] [P0-T9] Capture baseline Pytest output to `artifacts/atomic_executor_98_baseline_pytest.txt`
	- Acceptance: command `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing > artifacts/atomic_executor_98_baseline_pytest.txt 2>&1` exits 0 OR non-0, and `artifacts/atomic_executor_98_baseline_pytest.txt` is created

### Phase 1 — [expect-fail] Parser semantics for plan-linked expectations (TDD Red)
- [ ] [P1-T1] [expect-fail] Add fixture plan file `tests/fixtures/atomic_executor/plan_expect_pass.md` containing one task with `[expect-pass]` and an explicit `pytest <nodeid>` reference
	- Acceptance: file exists and contains the exact substring `[expect-pass] pytest tests/`
- [ ] [P1-T2] [expect-fail] Add fixture plan file `tests/fixtures/atomic_executor/plan_expect_fail_with_pytest_ref.md` containing one checked task with `[expect-fail]` and an explicit `pytest <nodeid>` reference
	- Acceptance: file exists and contains the exact substring `- [x] [P1-T1] [expect-fail] pytest `
- [ ] [P1-T3] [expect-fail] Add fixture plan file `tests/fixtures/atomic_executor/plan_expect_fail_with_prose_ref.md` containing one checked task using the prose form `Add pytest `...` in `...``
	- Acceptance: file exists and contains the exact substring `Add pytest `
- [ ] [P1-T4] [expect-fail] Add unit test asserting `PlanParser.parse()` strips `[expect-pass]` and sets a new `expect_pass` flag in `tests/scripts/dev_tools/atomic_executor/test_plan_parser.py`
	- Acceptance: `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_plan_parser.py -k expect_pass` fails
- [ ] [P1-T5] [expect-fail] Add unit test asserting `PlanParser.parse()` extracts `test_ref` from the `pytest <nodeid>` form in `tests/scripts/dev_tools/atomic_executor/test_plan_parser.py`
	- Acceptance: `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_plan_parser.py -k test_ref` fails
- [ ] [P1-T6] [expect-fail] Add unit test asserting `PlanParser.parse()` extracts `test_ref` from the prose form in `tests/scripts/dev_tools/atomic_executor/test_plan_parser.py`
	- Acceptance: `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_plan_parser.py -k prose_ref` fails

### Phase 2 — Implement plan parsing for `[expect-pass]` + `test_ref`
- [ ] [P2-T1] Extend `PlanTask` in `scripts/dev_tools/atomic_executor/plan_parser.py` with `expect_pass: bool` and `test_ref: str | None` fields (preserve existing `expect_fail` behavior)
	- Acceptance: `PlanTask` includes both attributes and `poetry run pyright` passes for `plan_parser.py`
- [ ] [P2-T2] Update `PlanParser.parse()` in `scripts/dev_tools/atomic_executor/plan_parser.py` to detect `[expect-pass]` and strip it from `title`
	- Acceptance: `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_plan_parser.py -k expect_pass` exits 0
- [ ] [P2-T3] Add deterministic `test_ref` extraction in `scripts/dev_tools/atomic_executor/plan_parser.py` for the `pytest <nodeid-or-prefix>` form
	- Acceptance: `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_plan_parser.py -k test_ref` exits 0
- [ ] [P2-T4] Add deterministic `test_ref` extraction in `scripts/dev_tools/atomic_executor/plan_parser.py` for the prose form `Add pytest `name` in `path``
	- Acceptance: `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_plan_parser.py -k prose_ref` exits 0

### Phase 3 — Pure logic helpers for expectations + pytest failure parsing
- [ ] [P3-T1] Add new module `scripts/dev_tools/atomic_executor/pytest_expectations.py` with typed function stubs for expectation resolution and pytest failure parsing
	- Acceptance: file exists and can be imported by `poetry run python -c "import scripts.dev_tools.atomic_executor.pytest_expectations"`
- [ ] [P3-T2] [expect-fail] Add unit test asserting expected-pass overrides expected-fail for the same `test_ref` in `tests/scripts/dev_tools/atomic_executor/test_pytest_expectations.py`
	- Acceptance: `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_pytest_expectations.py -k overrides` fails
- [ ] [P3-T3] [expect-fail] Add unit test asserting only checked tasks contribute to resolved expectations in `tests/scripts/dev_tools/atomic_executor/test_pytest_expectations.py`
	- Acceptance: `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_pytest_expectations.py -k checked` fails
- [ ] [P3-T4] [expect-fail] Add unit test asserting failing nodeids are parsed (including parameterized suffixes) from representative pytest output text in `tests/scripts/dev_tools/atomic_executor/test_pytest_expectations.py`
	- Acceptance: `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_pytest_expectations.py -k nodeid` fails
- [ ] [P3-T5] [expect-fail] Add unit test asserting pytest collection/import errors are detected and treated as gate failures in `tests/scripts/dev_tools/atomic_executor/test_pytest_expectations.py`
	- Acceptance: `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_pytest_expectations.py -k collect` fails
- [ ] [P3-T6] Implement the expectation-resolution helper in `scripts/dev_tools/atomic_executor/pytest_expectations.py` and make the override/checked-only tests pass
	- Acceptance: `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_pytest_expectations.py -k overrides` exits 0
- [ ] [P3-T7] Implement the pytest failure parsing helper in `scripts/dev_tools/atomic_executor/pytest_expectations.py` and make the parsing/collection tests pass
	- Acceptance: `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_pytest_expectations.py -k nodeid` exits 0

### Phase 4 — [expect-fail] Pre-flight QC becomes plan-aware (TDD Red)
- [ ] [P4-T1] [expect-fail] Add unit test asserting `_run_preflight_qc_with_capture(...)` allows pytest to fail when all failing nodeids match checked expected-fail refs in `tests/scripts/dev_tools/atomic_executor/test_cli.py`
	- Acceptance: `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_cli.py -k preflight_expected_fail` fails
- [ ] [P4-T2] [expect-fail] Add unit test asserting `_run_preflight_qc_with_capture(...)` fails when a failing nodeid is covered by both expected-fail and expected-pass (expected-pass wins) in `tests/scripts/dev_tools/atomic_executor/test_cli.py`
	- Acceptance: `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_cli.py -k preflight_expected_pass_wins` fails
- [ ] [P4-T3] [expect-fail] Add unit test asserting pre-flight QC fails fast when a checked expectation task lacks a resolvable `test_ref` in `tests/scripts/dev_tools/atomic_executor/test_cli.py`
	- Acceptance: `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_cli.py -k preflight_missing_test_ref` fails

### Phase 5 — Implement plan-aware pre-flight QC in `cli.py`
- [ ] [P5-T1] Update `scripts/dev_tools/atomic_executor/cli.py` so pre-flight pytest is invoked with `--color=no` to stabilize output parsing
	- Acceptance: `_run_preflight_qc_with_capture` includes `--color=no` in the pytest argv and `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_cli.py -k run_preflight_qc_with_capture` exits 0
- [ ] [P5-T2] Add a collect-only probe in `scripts/dev_tools/atomic_executor/cli.py` that validates checked expectation `test_ref` values can be collected before running the full pytest step
- [ ] [P5-T2] Add a collect-only probe in `scripts/dev_tools/atomic_executor/cli.py` that validates checked expectation `test_ref` values can be collected before running the full pytest step
	- Acceptance: `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_cli.py -k preflight_missing_test_ref` exits 0
- [ ] [P5-T3] Apply plan-linked expectation filtering to the pre-flight pytest step in `scripts/dev_tools/atomic_executor/cli.py` using `scripts/dev_tools/atomic_executor/pytest_expectations.py`
	- Acceptance: `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_cli.py -k preflight_expected_fail` exits 0
- [ ] [P5-T4] Plumb plan-derived expectations into the pre-flight QC path in `scripts/dev_tools/atomic_executor/cli.py` without changing default behavior for plans that have no checked expectations
	- Acceptance: `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_cli.py -k preflight_` exits 0

### Phase 6 — [expect-fail] Phase-end QC becomes plan-aware (TDD Red)
- [ ] [P6-T1] [expect-fail] Add unit test asserting `QCRunner.run_full(...)` tolerates expected-fail pytest failures when provided plan expectations in `tests/scripts/dev_tools/atomic_executor/test_qc_runner.py`
	- Acceptance: `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_qc_runner.py -k phase_expected_fail` fails
- [ ] [P6-T2] [expect-fail] Add unit test asserting `QCRunner.run_full(...)` fails when an unexpected failing nodeid is present in `tests/scripts/dev_tools/atomic_executor/test_qc_runner.py`
	- Acceptance: `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_qc_runner.py -k phase_unexpected_fail` fails
- [ ] [P6-T3] [expect-fail] Add unit test asserting the CLI passes resolved expectations into `qc_runner.run_full(...)` on phase completion in `tests/scripts/dev_tools/atomic_executor/test_cli.py`
	- Acceptance: `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_cli.py -k phase_end_expectations` fails

### Phase 7 — Implement plan-aware phase-end QC in `qc_runner.py` and wire it in
- [ ] [P7-T1] Extend `QCRunner.run_full(...)` in `scripts/dev_tools/atomic_executor/qc_runner.py` to accept optional resolved expectations (do not read plan files directly inside `QCRunner`)
	- Acceptance: `poetry run pyright` exits 0
- [ ] [P7-T2] Apply expectation filtering to the pytest step in `QCRunner.run_full(...)` using `scripts/dev_tools/atomic_executor/pytest_expectations.py`
	- Acceptance: `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_qc_runner.py -k phase_expected_fail` exits 0
- [ ] [P7-T3] Wire phase-end QC in `scripts/dev_tools/atomic_executor/cli.py` so the call site passes expectations derived from the current plan parser state
	- Acceptance: `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_cli.py -k phase_end_expectations` exits 0

### Phase 8 — Final QA (toolchain loop)
- [ ] [P8-T1] Run formatting (Black) and verify it exits 0: `poetry run black .`
	- Acceptance: command exits 0
- [ ] [P8-T2] Run linting (Ruff) and verify it exits 0: `poetry run ruff check`
	- Acceptance: command exits 0
- [ ] [P8-T3] Run type checking (Pyright) and verify it exits 0: `poetry run pyright`
	- Acceptance: command exits 0
- [ ] [P8-T4] Run tests (Pytest with coverage) and verify it exits 0: `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`
	- Acceptance: command exits 0
