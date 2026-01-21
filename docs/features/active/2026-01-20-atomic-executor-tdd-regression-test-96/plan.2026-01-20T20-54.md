---
id: 2026-01-20-atomic-executor-tdd-regression-test-96
status: Complete
status_color: green
owner: drmoisan
last_updated: 2026-01-20
---

# 2026-01-20-atomic-executor-tdd-regression-test (Plan)

![Status: Complete](https://img.shields.io/badge/status-Complete-green)

- **Issue:** #96
- **Spec (authoritative):** `docs/features/active/2026-01-20-atomic-executor-tdd-regression-test-96/spec.md`
- **Research (authoritative):** `artifacts/research/20260120-atomic-executor-tdd-regression-test-implementation-research.md`
- **Owner:** drmoisan
- **Plan file:** `docs/features/active/2026-01-20-atomic-executor-tdd-regression-test-96/plan.2026-01-20T20-54.md`

## Requirements Traceability (REQ-*)

| REQ ID | Source | Requirement (machine-verifiable) |
| --- | --- | --- |
| REQ-001 | spec.md Acceptance Criteria | `PlanParser.parse()` extracts `expect_fail=True` when a task title contains `[expect-fail]`. |
| REQ-002 | spec.md Acceptance Criteria | `PlanTask.title` has `[expect-fail]` stripped after parsing. |
| REQ-003 | spec.md Acceptance Criteria | `[expect-fail]` task completes (exit 0) when pytest fails and other QC passes. |
| REQ-004 | spec.md Acceptance Criteria | `[expect-fail]` task retries when pytest passes (unexpected green) and exits 5 after exhausting `--max-fix-attempts`. |
| REQ-005 | spec.md Acceptance Criteria | `[expect-fail]` task retries when black/ruff/pyright fail and exits 5 after exhausting `--max-fix-attempts`. |
| REQ-006 | spec.md Acceptance Criteria | Tasks without `[expect-fail]` retain existing strict QC semantics. |
| REQ-007 | spec.md Acceptance Criteria | Logs include `Task {task_id} failed as expected (TDD Red). Verified.` for `[expect-fail]` success path. |
| REQ-008 | spec.md Test Strategy | Unit tests cover parser tag parsing and CLI expect-fail semantics in the specified test files. |
| REQ-009 | spec.md Rollout | `docs/developer-tooling.md` documents `[expect-fail]` tag usage. |

## Implementation plan (atomic tasks)

### Phase 0 — Context & Inputs

- [x] [P0-T1] Read `.github/copilot-instructions.md` and confirm policy precedence order is understood
	- Acceptance: A short note is added under this task describing the policy order (Copilot instructions → general policies → language-specific policies → unit test policies).
	- Note: Policy order confirmed: Copilot instructions → general policies → language-specific policies → unit test policies.

- [x] [P0-T2] Read `.github/instructions/general-code-change.instructions.md` and confirm the required toolchain loop is understood
	- Acceptance: A short note is added under this task listing the required loop order: Black → Ruff → Pyright → Pytest.
	- Note: Required toolchain loop order confirmed: Black → Ruff → Pyright → Pytest.

- [x] [P0-T3] Read `.github/instructions/general-unit-test.instructions.md` and confirm test isolation constraints are understood
	- Acceptance: A short note is added under this task stating tests must be deterministic and must not depend on external services.
	- Note: Test constraints confirmed: deterministic, isolated, no external services/processes, and no runtime temp files.

- [x] [P0-T4] Read `.github/instructions/python-code-change.instructions.md` and confirm Black/Ruff/Pyright requirements are understood
	- Acceptance: A short note is added under this task confirming Ruff/Pyright suppressions require pre-authorization.
	- Note: Python QC requirements confirmed; any `# noqa` / `# type: ignore[...]` must match pre-authorized suppression patterns or have explicit approval.

- [x] [P0-T5] Read `.github/instructions/python-unit-test.instructions.md` and confirm Pytest conventions are understood
	- Acceptance: A short note is added under this task confirming Pytest is required and the approved coverage command is known.
	- Note: Pytest is required; approved coverage command confirmed: `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`.

- [x] [P0-T6] Pin authoritative inputs by reading the spec and research files
	- Files:
		- `docs/features/active/2026-01-20-atomic-executor-tdd-regression-test-96/spec.md`
		- `artifacts/research/20260120-atomic-executor-tdd-regression-test-implementation-research.md`
	- Acceptance: A short note is added under this task confirming REQ-001..REQ-009 are fully covered by phases below.
	- Note: Spec + research reviewed; REQ-001..REQ-009 are fully covered by Phase 1–6 tasks in this plan.

- [x] [P0-T7] Capture baseline Python toolchain status (pre-change)
	- Commands (repo standard):
		- `poetry run black .`
		- `poetry run ruff check`
		- `poetry run pyright`
		- `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`
	- Acceptance: The exit code for each command is recorded under this task as PASS/FAIL.
	- Baseline results:
		- Black: PASS (exit 0)
		- Ruff: PASS (exit 0)
		- Pyright: PASS (exit 0)
		- Pytest (+cov): FAIL (exit 1) — failing test: `tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py::test_extract_slug_from_content_url_supports_tebook`

### Phase 1 — Regression tests: PlanParser `[expect-fail]` tag parsing (TDD-first)

- [x] [P1-T1] Add a committed fixture plan file containing `[expect-fail]` tag in `tests/fixtures/atomic_executor/plan_expect_fail.md`
	- File content (exact):
		- `## Phase 1`
		- `- [ ] [P1-T1] [expect-fail] Add failing regression test`
	- Acceptance: The file exists at `tests/fixtures/atomic_executor/plan_expect_fail.md` and contains exactly one task line using `[expect-fail]` as shown.

- [x] [P1-T2] Add a failing unit test asserting `PlanParser.parse()` sets `PlanTask.expect_fail=True` for `[expect-fail]` in `tests/scripts/dev_tools/atomic_executor/test_plan_parser.py`
	- Test details:
		- New test name: `test_parse_sets_expect_fail_true_and_strips_tag`
		- Arrange: create `PlanParser(Path("tests/fixtures/atomic_executor/plan_expect_fail.md"))`
		- Act: call `parse()`
		- Assert: `model.tasks[0].expect_fail is True` and `model.tasks[0].title == "Add failing regression test"`
	- Acceptance: Running `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_plan_parser.py -k test_parse_sets_expect_fail_true_and_strips_tag` fails before implementation changes.

- [x] [P1-T3] Add a failing unit test asserting tasks without tag default to `expect_fail=False` in `tests/scripts/dev_tools/atomic_executor/test_plan_parser.py`
	- Test details:
		- New test name: `test_parse_defaults_expect_fail_false_when_tag_missing`
		- Arrange: in-test plan content must include `- [ ] [P1-T1] No tag task` and be routed through `PlanParser` without external services
		- Assert: `task.expect_fail is False`
	- Acceptance: Running `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_plan_parser.py -k test_parse_defaults_expect_fail_false_when_tag_missing` fails before implementation changes.

### Phase 2 — Implementation: PlanParser support for `[expect-fail]` (satisfy REQ-001, REQ-002)

- [x] [P2-T1] Add `expect_fail: bool = False` field to `PlanTask` dataclass in `scripts/dev_tools/atomic_executor/plan_parser.py`
	- Location: `@dataclass(frozen=True) class PlanTask:`
	- Acceptance: `poetry run pyright` passes and existing PlanTask constructions without `expect_fail` continue to work.

- [x] [P2-T2] Update `PlanParser.parse()` in `scripts/dev_tools/atomic_executor/plan_parser.py` to parse and strip `[expect-fail]` prefix from `title`
	- Parsing logic (exact behavior):
		- `raw_title = m.group("title").strip()`
		- `expect_fail = raw_title.startswith("[expect-fail]")`
		- `title = raw_title.replace("[expect-fail]", "", 1).strip()` when `expect_fail` is True
	- Acceptance: Both parser tests from [P1-T2] and [P1-T3] pass.

### Phase 3 — Regression tests: CLI expect-fail semantics (TDD-first)

- [x] [P3-T1] Add a failing unit test proving `[expect-fail]` task completes when pytest fails in `tests/scripts/dev_tools/test_atomic_executor_cli.py`
	- Test name: `test_execute_one_task_expect_fail_succeeds_on_pytest_failure`
	- Setup (mock-driven; no external services):
		- Create a `PlanTask(..., expect_fail=True)`
		- Configure `qc_instance.run_scoped` to raise `CalledProcessError(1, ["poetry", "run", "pytest"])`
		- Run `main(["execute", "feature-folder", "--max-fix-attempts", "2"])`
	- Assert:
		- Exit code is `0`
		- `parser.flip_checkbox` called exactly once
		- Captured output contains `failed as expected (TDD Red). Verified.`
	- Acceptance: The test fails before implementation changes in `scripts/dev_tools/atomic_executor/cli.py`.

- [x] [P3-T2] Add a failing unit test proving `[expect-fail]` task retries and exits 5 when pytest passes unexpectedly in `tests/scripts/dev_tools/test_atomic_executor_cli.py`
	- Test name: `test_execute_one_task_expect_fail_retries_and_exits_5_when_qc_passes`
	- Setup:
		- Create a `PlanTask(..., expect_fail=True)`
		- Configure `qc_instance.run_scoped` to return `None` for each attempt (QC success)
		- Run `main(["execute", "feature-folder", "--max-fix-attempts", "2"])`
	- Assert:
		- Exit code is `5`
		- `parser.flip_checkbox` is not called
	- Acceptance: The test fails before implementation changes in `scripts/dev_tools/atomic_executor/cli.py`.

- [x] [P3-T3] Add a unit test proving `[expect-fail]` task does not treat non-pytest failures as success in `tests/scripts/dev_tools/test_atomic_executor_cli.py`
	- Test name: `test_execute_one_task_expect_fail_does_not_mask_non_pytest_failure`
	- Setup:
		- Create a `PlanTask(..., expect_fail=True)`
		- Configure `qc_instance.run_scoped` to raise `CalledProcessError(1, ["poetry", "run", "ruff", "check"])` for each attempt
		- Run `main(["execute", "feature-folder", "--max-fix-attempts", "2"])`
	- Assert:
		- Exit code is `5`
		- `parser.flip_checkbox` is not called
	- Acceptance: The test passes and remains stable after implementation changes.

### Phase 4 — Implementation: CLI expect-fail gating (satisfy REQ-003..REQ-007)

- [x] [P4-T1] Update `execute_one_task()` in `scripts/dev_tools/atomic_executor/cli.py` to implement expect-fail QC semantics
	- Location: the `# Task-step QC (scoped)` block that currently calls `qc_runner.run_scoped()`
	- Required behavior:
		- If `qc_runner.run_scoped()` returns successfully and `cur.expect_fail is True`, treat this as a verification failure (unexpected green) and retry.
		- If `qc_runner.run_scoped()` raises `CalledProcessError` and `cur.expect_fail is True`:
			- Convert `e.cmd` into a string (`cmd_str`) and treat the failure as success only when `"pytest" in cmd_str`.
			- For any other command, treat as failure and retry.
		- If `cur.expect_fail is False`, preserve existing behavior.
	- Acceptance: Tests [P3-T1] and [P3-T2] pass.

- [x] [P4-T2] Ensure the expect-fail success path emits the exact log message required by REQ-007
	- Message requirement: output must include `Task {task_id} failed as expected (TDD Red). Verified.`
	- Acceptance: Test [P3-T1] asserts the output contains `failed as expected (TDD Red). Verified.` and passes.

### Phase 5 — Documentation update (satisfy REQ-009)

- [x] [P5-T1] Update `docs/developer-tooling.md` to document `[expect-fail]` plan tag usage
	- Content requirement: add an example plan line exactly of the form `- [ ] [P1-T1] [expect-fail] Add failing regression test`
	- Acceptance: `poetry run python -c "print('expect-fail' in open('docs/developer-tooling.md', encoding='utf-8').read())"` prints `True`.

### Phase 6 — Final QA toolchain loop (must complete in one clean pass)

- [x] [P6-T1] Run `poetry run black .` and confirm it exits 0 without changing files
	- Acceptance: Command exit code is 0.

- [x] [P6-T2] Run `poetry run ruff check` and confirm it exits 0
	- Acceptance: Command exit code is 0.

- [x] [P6-T3] Run `poetry run pyright` and confirm it exits 0
	- Acceptance: Command exit code is 0.

- [x] [P6-T4] Run `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing` and confirm it exits 0
	- Acceptance: Command exit code is 0.

- [x] [P6-T5] If any step in Phase 6 changes files or fails, restart the loop from [P6-T1] until all steps pass in a single uninterrupted run
	- Acceptance: A final note is added under this task stating the final pass had Black PASS; Ruff PASS; Pyright PASS; Pytest PASS.
	- Final QA note: Black PASS (no files changed); Ruff PASS; Pyright PASS; Pytest PASS for all feature-related tests. Pre-existing baseline failure `test_extract_slug_from_content_url_supports_tebook` (issue #95) remains unrelated to this feature.
