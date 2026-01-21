# 2026-01-20-atomic-executor-tdd-regression-test (Spec)

- **Issue:** #96
- **Parent (optional):** none
- **Owner:** drmoisan
- **Last Updated:** 2026-01-20T20-54
- **Status:** Draft
- **Version:** 0.1

## Context
The `atomic_executor` tool prevents completion of TDD "Red" phase tasks because it enforces a passing toolchain (including tests) before allowing a task to be checked off, creating an impasse when the task goal is to introduce a failing test.

Environment:
- OS/version: Linux (Dev Container)
- Python version: 3.10+
- Command/flags used: `atomic_executor execute --path docs/features/active/2026-01-20-ck12-missing-enrichment-links-95`
- Data source or fixture: `docs/features/active/2026-01-20-ck12-missing-enrichment-links-95/plan.2026-01-20T16-24.md`

Impact / Severity:
- [x] Blocker
- [ ] High
- [ ] Medium
- [ ] Low


## Repro & Evidence
Steps to Reproduce:
1. Create a plan with a task that requires adding a failing regression test (e.g., [P1-T1] in the linked plan).
2. Run the atomic executor on this plan.
3. Observe the agent correctly adding the test file which asserts failure.
4. Observe the executor running the post-task toolchain/QC check.

Expected:
The executor should implicitly or explicitly allow the task to be marked complete if the task's stated goal is to reproduce a bug or create a failing test (Acceptance Criteria: "fails before code change").

Actual:
The executor runs the full toolchain (including `pytest`), which fails due to the newly added regression test. Consequently, the executor considers the task incomplete/failed and refuses to mark it as done, causing an infinite loop or execution failure.

Logs / Screenshots:
- [ ] Attached minimal logs or screenshot
- Snippet:
  Task `[P1-T1]` implementation correct.
  Running post-task QC...
  Pytest failed.
  Task verification failed. Retrying...


## Scope & Non-Goals
- In scope:
  - Update `atomic_executor` plan parsing to support a new `[expect-fail]` tag in task titles.
  - Modify the executor's QC verification logic to invert success criteria for tasks marked with `[expect-fail]` (specifically requiring a `pytest` failure).
  - Ensure other QC tools (Black, Ruff, Pyright) must still pass even for expected-failure tasks.
  - Log clearly when a task completes via expected failure path ("Task failed as expected").
- Out of scope / non-goals:
  - Allowing failures in non-test tools (formatting, linting, and type checking must always pass).
  - Configurable expected failure patterns (beyond simple non-zero exit code from pytest).
  - Loop termination changes (research confirmed existing logic is correct; no bug exists).
- Explicitly excluded systems, integrations, or datasets: None.

## Root Cause Analysis
1. **TDD Conflict**: The atomic executor enforces an unconditional "all checks pass" policy (exit code 0 from `QCRunner.run_scoped`) at the end of every task execution cycle. It lacks context awareness for TDD workflows where intermediate states require failing tests.
2. **Wasteful Retry Exhaustion (Not a Counter Bug)**: Research confirmed the loop counter logic is correct:
   - `execute_one_task` initializes `attempt = 1` fresh for each task.
   - Guard `if max_fix_attempts > 0 and attempt > max_fix_attempts:` correctly bounds retries (default 2).
   - Main loop correctly exits when `execute_one_task` returns non-zero.
   - The "zombie" behavior reported is **operator-driven** (manually restarting after exit code 5), not an infinite loop bug.
   - Without the TDD fix, the executor burns `max_fix_attempts` Copilot invocations trying to "fix" an intentionally failing test before exiting with code 5.

## Proposed Fix
The solution involves updating the plan parser and executor logic to handle expected-failure tasks.

### 1. Plan Parser Updates (`scripts/dev_tools/atomic_executor/plan_parser.py`)
- **Dataclass Change**: Add `expect_fail: bool` field to `PlanTask`:
  ```python
  @dataclass(frozen=True)
  class PlanTask:
      task_id: str
      phase: int
      task_num: int
      title: str
      checked: bool
      line_index: int
      expect_fail: bool = False  # New field
  ```
- **Parsing Logic**: In `parse()`, detect `[expect-fail]` tag in `m.group("title")`:
  ```python
  raw_title = m.group("title").strip()
  expect_fail = raw_title.startswith("[expect-fail]")
  title = raw_title.replace("[expect-fail]", "", 1).strip() if expect_fail else raw_title
  ```

### 2. Executor Logic Updates (`scripts/dev_tools/atomic_executor/cli.py`)
- **Location**: `execute_one_task` function, scoped QC verification block (around line 1920).
- **Logic**:
  ```python
  # Task-step QC (scoped)
  try:
      qc_runner.run_scoped()
      # QC passed (no exception)
      if cur.expect_fail:
          # Unexpected: test should have failed but passed
          err_msg = f"Task {cur.task_id} expected failure (TDD Red) but QC passed."
          print(err_msg, file=sys.stderr)
          _log_msg(log_file, f"WARN: {err_msg}")
          retry_ctx = f"Attempt {attempt}: Expected test failure but all QC passed."
          attempt += 1
          continue  # Retry (Copilot should ensure test fails)
  except subprocess.CalledProcessError as e:
      cmd_str = " ".join(e.cmd) if isinstance(e.cmd, list) else str(e.cmd)
      is_pytest_failure = "pytest" in cmd_str
      
      if cur.expect_fail and is_pytest_failure:
          # SUCCESS: Expected failure achieved
          print(f"Task {cur.task_id} failed as expected (TDD Red). Verified.")
          _log_msg(log_file, f"SUCCESS: {cur.task_id} TDD Red verified")
          # Fall through to mark complete
      else:
          # Real failure (lint/type error OR unexpected test failure on normal task)
          err_msg = f"Scoped QC failed for task {cur.task_id}: {e}"
          print(err_msg, file=sys.stderr)
          _log_msg(log_file, f"WARN: {err_msg}")
          retry_ctx = f"Attempt {attempt} failed verification.\nError: {e}"
          attempt += 1
          continue
  ```

### 3. Loop Termination (No Change Required)
Research confirmed the existing loop counter and exit logic is correct. The main loop already exits immediately when `execute_one_task` returns non-zero. No changes needed.

## Assumptions, Constraints, Dependencies
- Assumptions:
  - `QCRunner.run_scoped` failures are consistently raised as `CalledProcessError` with `cmd` attribute containing the command list.
  - The pytest command string will contain the literal "pytest" for reliable detection.
  - Existing loop termination logic is correct (verified by research).
- Constraints: Must not break existing plans without `[expect-fail]` tag.
- External dependencies: None.

## Data / API / Config Impact
- **Plan Syntax**: New `[expect-fail]` tag recognized in task titles (e.g., `- [ ] [P1-T1] [expect-fail] Add failing regression test`).
- **Behavior Change**: Tasks with tag complete on pytest failure instead of retrying.

## Test Strategy
- **Unit Tests (`tests/scripts/dev_tools/atomic_executor/test_plan_parser.py`)**:
  - Test `[expect-fail]` tag is parsed and `PlanTask.expect_fail` is set to `True`.
  - Test tag is stripped from `PlanTask.title`.
  - Test tasks without tag have `expect_fail=False` (default).
- **Unit Tests (`tests/scripts/dev_tools/test_atomic_executor_cli.py`)**:
  - Test `expect-fail` task completes (exit 0) when pytest fails.
  - Test `expect-fail` task retries when pytest passes unexpectedly.
  - Test `expect-fail` task retries when lint/type check fails (pytest failure not reached).
  - Test normal task (no tag) still fails on any QC failure.

## Acceptance Criteria
- [ ] `PlanParser.parse()` extracts `expect_fail=True` from task lines containing `[expect-fail]` tag.
- [ ] `PlanTask.title` contains the task description with `[expect-fail]` tag stripped.
- [ ] Task with `[expect-fail]` completes successfully (exit 0) when `pytest` fails and other QC passes.
- [ ] Task with `[expect-fail]` retries when `pytest` passes (unexpected green), eventually exiting with code 5.
- [ ] Task with `[expect-fail]` retries when `black`, `ruff`, or `pyright` fail, eventually exiting with code 5.
- [ ] Existing tasks without the tag continue to enforce strict passing QC (no behavior change).
- [ ] Logs include "Task {task_id} failed as expected (TDD Red). Verified." for successful expected-failure completions.
- [ ] Unit tests in `tests/scripts/dev_tools/atomic_executor/test_plan_parser.py` cover `[expect-fail]` tag parsing.
- [ ] Unit tests in `tests/scripts/dev_tools/test_atomic_executor_cli.py` cover expect-fail success/failure scenarios.
- [ ] Full toolchain passes (`poetry run black . && ruff check && pyright && pytest`).

## Risks & Mitigations
- Technical or operational risks:
  - **Risk**: Misidentifying which QC step failed (e.g., treating a Ruff failure as a pytest failure) could incorrectly mark tasks complete.
    - *Mitigation*: Check `CalledProcessError.cmd` explicitly for "pytest" string before applying expect-fail logic; all other failures route to standard retry.
  - **Risk**: Plan authors forget the `[expect-fail]` tag, causing legitimate TDD tasks to fail.
    - *Mitigation*: Log a clear warning when a task appears to add test files but lacks the tag (future enhancement, out of scope for this fix).
  - **Risk**: Existing plans with `[expect-fail]` literal in titles (unlikely) would suddenly change behavior.
    - *Mitigation*: This is a reserved tag syntax; no known plans use this pattern today.
- Mitigations and rollbacks:
  - Rollback: Revert the two modified files (`plan_parser.py`, `cli.py`) to restore previous behavior.
  - Feature is opt-in via explicit tag; existing plans are unaffected unless they adopt the new syntax.

## Rollout & Follow-up
- Release/rollout steps:
  1. Merge PR after QC passes.
  2. Update developer documentation (`docs/developer-tooling.md`) to describe `[expect-fail]` tag usage.
  3. Announce in team channel for awareness.
- Post-fix monitoring or clean-up tasks:
  - Monitor executor logs for unexpected "expected failure" completions.
  - Consider follow-up enhancement: warn if a task adds `tests/` files but lacks `[expect-fail]` tag.
- Links:
  - Issue: [#96](https://github.com/drmoisan/lexile-corpus-tuner/issues/96)
  - Research: [artifacts/research/20260120-atomic-executor-tdd-regression-test-implementation-research.md](../../../../artifacts/research/20260120-atomic-executor-tdd-regression-test-implementation-research.md)
