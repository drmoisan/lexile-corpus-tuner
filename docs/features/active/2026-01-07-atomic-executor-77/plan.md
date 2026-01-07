---
id: REQ-2026-01-07-atomic-executor
status: Planned
status_color: blue
owner: drmoisan
last_updated: 2026-01-07
---

# 2026-01-07-atomic-executor - Plan

- Issue: #77
- Owner: 2026-01-07-atomic-executor
- Last Updated: 2026-01-07

**Status:** ![Planned](https://img.shields.io/badge/Status-Planned-blue)

## Required References

- General Coding Standards: [`.github/instructions/general-code-change.instructions.md`](../../../../.github/instructions/general-code-change.instructions.md)
- General Unit Test Policy: [`.github/instructions/general-unit-test.instructions.md`](../../../../.github/instructions/general-unit-test.instructions.md)
- Python Coding Standards: [`.github/instructions/python-code-change.instructions.md`](../../../../.github/instructions/python-code-change.instructions.md)
- Python Unit Test Policy: [`.github/instructions/python-unit-test.instructions.md`](../../../../.github/instructions/python-unit-test.instructions.md)

**All work must comply with these policies; do not duplicate their content here.**

## Implementation Plan (Atomic Tasks)

> **Instructions for this section:**
> - Break work into **Phases** (broad buckets) and **Atomic Tasks** (binary, 5-30 min units).
> - Use `- [ ] [P#-T#]` for every task.
> - Start every task with a **strong verb** (Implement, Create, Update, Verify).
> - No "bucket" tasks like "Refactor module" or "Write tests"; split them into specific, verifiable steps.

### Phase 0: Compliance & Context
- [ ] [P0-T1] Confirm alignment with repo policies by reading `.github/instructions/general-code-change.instructions.md`, `.github/instructions/python-code-change.instructions.md`, `.github/instructions/general-unit-test.instructions.md`, and `.github/instructions/python-unit-test.instructions.md` before touching code
  - Acceptance: Development log contains policy review timestamp prior to Phase 1 commits

### Phase 1: Single-Task Execution Refactor
- [ ] [P1-T1] Add unit test `test_execute_one_task_retries_until_success` in `tests/scripts/dev_tools/test_atomic_executor_cli.py` covering multiple QC failures followed by success with checkbox flip verification
  - Acceptance: Test fails before implementation (red) and serves as the spec for the refactor
- [ ] [P1-T2] Extract helper `def _execute_one_task(...):` in `scripts/dev_tools/atomic_executor/cli.py` encapsulating prompt creation, Copilot invocation, scoped QC loop, checkbox flipping, and retry counter handling
  - Acceptance: Unit test coverage for `_execute_one_task` demonstrates checkbox flip on success and respects retry counters; `test_execute_one_task_retries_until_success` passes
- [ ] [P1-T3] Update `execute` and `resume` code paths in `scripts/dev_tools/atomic_executor/cli.py` to call `_execute_one_task` and to interpret `--max-fix-attempts 0` as infinite retries
  - Acceptance: Running the updated commands against a mocked Copilot executor shows retries continue until mocked success and plan state remains unchanged on forced failure

### Phase 2: Execute-All Workflow Implementation
- [ ] [P2-T1] Add `execute-all` subparser in `scripts/dev_tools/atomic_executor/cli.py::parse_args` mirroring `execute`/`resume` options and updating help text
  - Acceptance: `python -m scripts.dev_tools.atomic_executor.cli --help` lists `execute-all` with correct arguments
- [ ] [P2-T2] Add unit test `test_execute_all_runs_full_qc_after_phase` in `tests/scripts/dev_tools/test_atomic_executor_cli.py` validating full toolchain invocation when a phase completes
  - Acceptance: Test asserts mock full QC runner invoked exactly once per completed phase (red state)
- [ ] [P2-T3] Add unit test `test_execute_all_respects_infinite_retry` in `tests/scripts/dev_tools/test_atomic_executor_cli.py` ensuring `--max-fix-attempts 0` never aborts prematurely and emits retry logs
  - Acceptance: Test confirms loop continues past prior retry limit and log entries exist for retries (red state)
- [ ] [P2-T4] Implement execute-all orchestrator loop in `scripts/dev_tools/atomic_executor/cli.py::main` to iterate tasks, call `_execute_one_task`, reparse plan, detect phase completion, run `QCRunner.run_full()`, and exit with code 5 on unrecovered gate failure
  - Acceptance: Integration test with stubbed Copilot executor processes all tasks in a fixture plan, executes full QC after phase completion, and records retries; Phase 2 tests pass

### Phase 3: Logging, Prompt Enhancements, and Final Verification
- [ ] [P3-T1] Extend logging in `scripts/dev_tools/atomic_executor/cli.py` to record per-attempt retries, phase-level full QC outcomes, and summary statistics to `.agent_logs/<timestamp>.log`
  - Acceptance: Automated test asserts log file includes retry counts, phase summaries, and final status for each task
- [ ] [P3-T2] Update prompt builder usage in `_execute_one_task` to include retry context and ensure clipboard copy occurs each attempt while honoring `--print-prompt` / `--copy-prompt`
  - Acceptance: Unit test confirms prompt text includes current task and retry count while clipboard copy path executes without raising
- [ ] [P3-T3] Add CLI integration test `test_execute_all_aborts_with_exit_code_5_on_persistent_failure` in `tests/scripts/dev_tools/test_atomic_executor_cli.py` confirming plan remains unchecked and exit code equals 5 when retries are exhausted
  - Acceptance: Integration harness observes exit code 5 and no checkbox flip for failing task
- [ ] [P3-T4] Update documentation in `README.md` CLI section to document `execute-all`, retry semantics, and logging outputs
  - Acceptance: README diff includes new subsection with example commands and explanation of retry policy

## Test Plan

- Unit: Run `poetry run pytest tests/scripts/dev_tools/test_atomic_executor_cli.py -k "execute_one_task or execute_all"` and ensure all new scenarios pass.
- Integration: Execute `python -m scripts.dev_tools.atomic_executor.cli execute-all docs/features/fixtures/sample-plan --max-fix-attempts 2` using a mocked Copilot executor to validate orchestration and logging without network dependency.
- Manual/CLI: Perform live run on an actual feature folder with `poetry run python -m scripts.dev_tools.atomic_executor.cli execute-all docs/features/active/<feature-folder> --max-fix-attempts 3`, verify log files, plan checkbox updates, retry counters, and full QC execution after each phase.

## Open Questions / Notes

- None at this time.
