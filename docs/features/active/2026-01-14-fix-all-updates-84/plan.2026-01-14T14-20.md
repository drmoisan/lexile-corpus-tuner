---
feature_id: 2026-01-14-fix-all-updates-84
issue: 84
repo: drmoisan/lexile-corpus-tuner
default_branch: main
working_branch: development
owner: drmoisan
status: Planned
status_color: blue
last_updated: 2026-01-14
---

# 2026-01-14-fix-all-updates - Plan

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

This plan updates `scripts/dev_tools/fix_all.py` to provide (1) live per-branch status while the parallel branches run, (2) default fail-fast semantics, and (3) an opt-in `--complete-all` flag to restore run-to-completion behavior, while preserving the existing per-branch buffered logs and final “Branch Results” summary.

All tasks are designed to be executable without human interpretation by locating code via exact symbol names and exact string matches (rather than line numbers).

## Required References

- General Coding Standards: [`.github/instructions/general-code-change.instructions.md`](../../../../.github/instructions/general-code-change.instructions.md)
- General Unit Test Policy: [`.github/instructions/general-unit-test.instructions.md`](../../../../.github/instructions/general-unit-test.instructions.md)
- Python Coding Standards: [`.github/instructions/python-code-change.instructions.md`](../../../../.github/instructions/python-code-change.instructions.md)
- Python Unit Test Policy: [`.github/instructions/python-unit-test.instructions.md`](../../../../.github/instructions/python-unit-test.instructions.md)

**All work must comply with these policies; do not duplicate their content here.**

## Requirements Traceability

| REQ-ID | Requirement (deterministic) | Source | Verification Artifact |
| --- | --- | --- | --- |
| REQ-COMPLIANCE-001 | Policy docs are read and acknowledged in the repo-required order before code changes begin. | `.github/agents/atomic_execution.agent.md` “Highest Priority: Repository Policy Compliance” | `artifacts/qa/policy_ack_84.txt` exists and contains the ordered list of policy files read. |
| REQ-BASELINE-001 | Baseline quality-gate outputs are captured before modifications (Ruff, Pyright, Pytest; coverage when used). | `.github/agents/atomic_execution.agent.md` “Plan Ingestion Protocol”; repo policies | `artifacts/qa/fix_all_84_baseline.txt` exists and contains outputs for Black, Ruff, Pyright, Pytest (with coverage). |
| REQ-CLI-001 | Add CLI flag `--complete-all` (default `False`) to disable fail-fast behavior. | `spec.md` “API / CLI Surface”; `user-story.md` Acceptance Criteria | Pytest unit test(s) asserting `parse_args(["--complete-all"])` sets `args.complete_all is True`; help text contains `--complete-all`. |
| REQ-EXEC-001 | Default fail-fast: first branch failure signals cancellation and other branches stop before starting their next step (step-boundary cancellation). | `spec.md` “Fail-fast semantics”; `user-story.md` Acceptance Criteria | Pytest unit test(s) proving other branches do not invoke their next step after cancellation is set. |
| REQ-EXEC-002 | `--complete-all` restores run-to-completion semantics (branches do not stop when another fails). | `spec.md` “Fail-fast semantics”; `user-story.md` Story Statement | Pytest unit test(s) proving a non-failing branch continues to execute subsequent steps after another branch fails when `complete_all=True`. |
| REQ-UI-001 | Interactive terminals: render a fixed-height live status board (one line per branch) with in-place redraw, updating at least at step boundaries, without corrupting subsequent output. | `spec.md` “Status board behavior (interactive vs. non-interactive)”; `user-story.md` Acceptance Criteria | Pytest unit test(s) for pure board rendering functions; non-interactive mode tests ensure no ANSI is emitted when redirecting output. |
| REQ-UI-002 | Non-interactive output (CI/redirect): do not use cursor movement; emit line-oriented status transitions at step boundaries. | `spec.md` “Status board behavior (interactive vs. non-interactive)” | Pytest unit test(s) asserting non-interactive mode emits `STATUS|...` lines via injected status sink, without ANSI sequences. |
| REQ-WIN-001 | On Windows, attempt best-effort VT enablement; if not possible, fall back to non-interactive mode. | `spec.md` “Windows console support” | Pytest unit test(s) for decision logic (pure function) selecting interactive vs fallback. |
| REQ-SHELL-001 | Shell branch: when shell tests are skipped (exit code 0 but output contains skip message), surface `SKIP tests` in the status board rather than reporting a normal “test” pass. | `spec.md` “Shell \“SKIP tests\” behavior”; `user-story.md` Acceptance Criteria | Pytest unit test(s) for `_shell_test_was_skipped(output)` and branch status transitions producing `SKIP tests`. |
| REQ-OUTPUT-001 | Preserve end-of-run output shape: per-branch buffered logs printed after completion and the final “Branch Results” summary with PASS/FAIL and failed step name when applicable. | `spec.md` “Behavior”; `user-story.md` Acceptance Criteria | Pytest regression tests asserting the summary header/footer and branch result line formatting remain present in the main logger stream. |
| REQ-QA-001 | Formatting passes in a clean final toolchain pass. | Repo policy: `.github/instructions/python-code-change.instructions.md` | `poetry run black .` exits 0 with no changes in the final pass. |
| REQ-QA-002 | Linting passes in a clean final toolchain pass. | Repo policy: `.github/instructions/python-code-change.instructions.md` | `poetry run ruff check` exits 0 in the final pass. |
| REQ-QA-003 | Type checking passes in a clean final toolchain pass. | Repo policy: `.github/instructions/python-code-change.instructions.md` | `poetry run pyright` exits 0 in the final pass. |
| REQ-QA-004 | Tests pass (with coverage) in a clean final toolchain pass. | Repo policy: `.github/instructions/python-unit-test.instructions.md` | `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing` exits 0 in the final pass. |

## Implementation Plan (Atomic Tasks)

> **Instructions for this section:**
> - Break work into **Phases** (broad buckets) and **Atomic Tasks** (binary, 5-30 min units).
> - Use `- [ ] [P#-T#]` for every task.
> - Start every task with a **strong verb** (Implement, Create, Update, Verify).
> - No "bucket" tasks like "Refactor module" or "Write tests"; split them into specific, verifiable steps.
> - **Self-Validating Phases:** Include necessary test creation/update tasks *within* the phase that implements the code. Do not defer verification to a final "Testing" phase.

### Phase 0 — Context & Inputs
- [ ] [P0-T1] (REQ-COMPLIANCE-001, TASK-COMPLIANCE-001) Create `artifacts/qa/policy_ack_84.txt` that contains the ordered list of policy files read: `.github/copilot-instructions.md` → `.github/instructions/general-code-change.instructions.md` → `.github/instructions/general-unit-test.instructions.md` → `.github/instructions/python-code-change.instructions.md` → `.github/instructions/python-unit-test.instructions.md`
  - Acceptance: `artifacts/qa/policy_ack_84.txt` exists and contains all five paths in the exact order above.
- [ ] [P0-T2] (REQ-BASELINE-001, TASK-BASELINE-001) Capture baseline quality gate outputs (no code changes) into `artifacts/qa/fix_all_84_baseline.txt` by running commands in this exact order: `poetry run black .` → `poetry run ruff check` → `poetry run pyright` → `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`
  - Acceptance: `artifacts/qa/fix_all_84_baseline.txt` exists and contains each command string verbatim at least once.

### Phase 1 — TDD: CLI flag (`--complete-all`) surface
- [ ] [P1-T1] (REQ-CLI-001, TASK-CLI-TEST-001) Add Pytest unit test `test_help_includes_complete_all_flag` in `tests/scripts/dev_tools/test_fix_all.py` that captures `--help` output and asserts it contains the substring `--complete-all`
  - Acceptance: Running `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_help_includes_complete_all_flag` exits non-zero before any implementation changes.
- [ ] [P1-T2] (REQ-CLI-001, TASK-CLI-TEST-002) Add Pytest unit test `test_parse_args_complete_all_sets_true` in `tests/scripts/dev_tools/test_fix_all.py` asserting `fix_all.parse_args(["--complete-all"]).complete_all is True`
  - Acceptance: Running `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_parse_args_complete_all_sets_true` exits non-zero before any implementation changes.
- [ ] [P1-T3] (REQ-CLI-001, TASK-CLI-IMPL-001) Update `scripts/dev_tools/fix_all.py:parse_args` to add `--complete-all` with `action="store_true"` and default `False`
  - Acceptance: Running both `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_help_includes_complete_all_flag` and `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_parse_args_complete_all_sets_true` exits 0.

### Phase 2 — TDD: Fail-fast cancellation vs `--complete-all`
- [ ] [P2-T1] (REQ-EXEC-001, TASK-EXEC-TEST-001) Add Pytest unit test `test_fail_fast_cancels_json_before_validate` in `tests/scripts/dev_tools/test_fix_all.py`:
  - Test setup requirements (deterministic):
    - Configure `python` responses so it fails at step name `Pyright: type-check`.
    - Configure `json` responses with `JSON: format` success only (omit `JSON: validate` so a call would raise AssertionError).
  - Assertion requirements: `fix_all.run_fix_all(...)` exits non-zero AND json runner calls do not include `JSON: validate`.
  - Acceptance: Running `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_fail_fast_cancels_json_before_validate` exits non-zero before cancellation is implemented.
- [ ] [P2-T2] (REQ-EXEC-002, TASK-EXEC-TEST-002) Add Pytest unit test `test_complete_all_allows_json_validate_after_python_failure` in `tests/scripts/dev_tools/test_fix_all.py`:
  - Test setup requirements (deterministic):
    - Same as [P2-T1] but provide `JSON: validate` success response.
    - Invoke `fix_all.run_fix_all(..., complete_all=True, ...)`.
  - Assertion requirements: json runner calls include `JSON: validate` even though python fails.
  - Acceptance: Running `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_complete_all_allows_json_validate_after_python_failure` exits non-zero before `complete_all` is wired through.

- [ ] [P2-T3] (REQ-EXEC-002, TASK-EXEC-TEST-003) Add Pytest unit test `test_run_fix_all_accepts_complete_all_parameter` in `tests/scripts/dev_tools/test_fix_all.py` that calls `fix_all.run_fix_all(..., complete_all=True, ...)` with a runner factory configured for immediate success in all branches
  - Acceptance: Running `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_run_fix_all_accepts_complete_all_parameter` exits non-zero before `run_fix_all` accepts the keyword argument.

### Phase 3 — TDD: Status formatting + rendering mode selection
- [ ] [P3-T1] (REQ-UI-002, TASK-UI-TEST-001) Add Pytest unit test `test_format_status_transition_line_exact_format` in `tests/scripts/dev_tools/test_fix_all.py` for new function `fix_all.format_status_transition_line(branch: str, status: str) -> str` returning exactly `STATUS|branch=<branch>|status=<status>`
  - Acceptance: Running `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_format_status_transition_line_exact_format` exits non-zero before implementation.
- [ ] [P3-T2] (REQ-UI-001, TASK-UI-TEST-002) Add Pytest unit test `test_render_status_board_line_count_and_trailing_newline` in `tests/scripts/dev_tools/test_fix_all.py` for new function `fix_all.render_status_board(lines: list[str], *, width: int) -> str`:
  - Assertion requirements: output contains `len(lines)` newline characters and endswith `\n`.
  - Acceptance: Running `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_render_status_board_line_count_and_trailing_newline` exits non-zero before implementation.
- [ ] [P3-T3] (REQ-WIN-001, TASK-WIN-TEST-001) Add Pytest unit test `test_should_use_interactive_board_requires_isatty_and_vt` in `tests/scripts/dev_tools/test_fix_all.py` for new function `fix_all.should_use_interactive_board(*, isatty: bool, vt_enabled: bool) -> bool`:
  - Assertion requirements: only returns True when both inputs are True.
  - Acceptance: Running `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_should_use_interactive_board_requires_isatty_and_vt` exits non-zero before implementation.

### Phase 4 — Implement fail-fast cancellation (make Phase 2 tests pass)
- [ ] [P4-T1] (REQ-EXEC-002, TASK-EXEC-IMPL-001) Update `scripts/dev_tools/fix_all.py:run_fix_all` signature to accept `complete_all: bool = False` and create a shared `cancel_event = threading.Event()` inside `run_fix_all`
  - Acceptance: Running `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_run_fix_all_accepts_complete_all_parameter` exits 0.
- [ ] [P4-T2] (REQ-EXEC-001, TASK-EXEC-IMPL-002) Update `scripts/dev_tools/fix_all.py` thread wrapper `_runner(...)` inside `run_fix_all` so that when a branch returns `success=False` and `complete_all is False`, it sets `cancel_event.set()` immediately
  - Acceptance: `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_fail_fast_cancels_json_before_validate` still exits non-zero until json step-boundary check is implemented.
- [ ] [P4-T3] (REQ-EXEC-001, TASK-EXEC-IMPL-003) Add step-boundary cancellation check to `run_json_branch` so `JSON: validate` is not started when `cancel_event.is_set()`
  - Acceptance: Running `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_fail_fast_cancels_json_before_validate` exits 0.
- [ ] [P4-T4] (REQ-EXEC-002, TASK-EXEC-IMPL-004) Gate cancellation behavior behind `complete_all` so that when `complete_all=True`, `run_json_branch` does not stop due to `cancel_event` and `_runner(...)` does not set cancellation
  - Acceptance: Running `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_complete_all_allows_json_validate_after_python_failure` exits 0.
- [ ] [P4-T5] (REQ-CLI-001, TASK-CLI-IMPL-003) Update `scripts/dev_tools/fix_all.py:main` so it passes `complete_all=args.complete_all` into `run_fix_all`
  - Acceptance: Running `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_parse_args_complete_all_sets_true` exits 0 and `poetry run python -m scripts.dev_tools.fix_all --help` output contains `--complete-all`.

### Phase 5 — Implement status formatting + mode selection (make Phase 3 tests pass)
- [ ] [P5-T1] (REQ-UI-002, TASK-UI-IMPL-001) Implement `scripts/dev_tools/fix_all.py:format_status_transition_line(branch: str, status: str) -> str` returning exactly `STATUS|branch=<branch>|status=<status>`
  - Acceptance: Running `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_format_status_transition_line_exact_format` exits 0.
- [ ] [P5-T2] (REQ-UI-001, TASK-UI-IMPL-002) Implement `scripts/dev_tools/fix_all.py:render_status_board(lines: list[str], *, width: int) -> str` so it returns a string with `len(lines)` newline characters and a trailing newline
  - Acceptance: Running `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_render_status_board_line_count_and_trailing_newline` exits 0.
- [ ] [P5-T3] (REQ-WIN-001, TASK-WIN-IMPL-001) Implement `scripts/dev_tools/fix_all.py:should_use_interactive_board(*, isatty: bool, vt_enabled: bool) -> bool` returning True only when both inputs are True
  - Acceptance: Running `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_should_use_interactive_board_requires_isatty_and_vt` exits 0.

### Phase 6 — Implement status emission at step boundaries (non-interactive first)
- [ ] [P6-T1] (REQ-UI-002, TASK-UI-TEST-003) Add Pytest unit test `test_non_interactive_emits_status_transitions_without_ansi` in `tests/scripts/dev_tools/test_fix_all.py`:
  - Test setup requirements: run `fix_all.run_fix_all(...)` with non-interactive detection based on `logger.stream.isatty()` returning False by using `StepLogger(stream=StringIO())`.
  - Assertion requirements: the captured log contains at least one substring starting with `STATUS|branch=` and contains no `\x1b[`.
  - Acceptance: Running `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_non_interactive_emits_status_transitions_without_ansi` exits non-zero before implementation.
- [ ] [P6-T2] (REQ-UI-002, TASK-UI-IMPL-003) Update `scripts/dev_tools/fix_all.py` so each branch emits status transitions at step boundaries when not using interactive board:
  - Exact format: use `format_status_transition_line(branch, status)` and write it via the provided `StepLogger.stream`.
  - Acceptance: Running `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_non_interactive_emits_status_transitions_without_ansi` exits 0.

### Phase 7 — Interactive status board redraw (TTY mode)
- [ ] [P7-T1] (REQ-UI-001, TASK-UI-TEST-004) Add Pytest unit test `test_format_ansi_redraw_contains_only_erase_and_cursor_up` in `tests/scripts/dev_tools/test_fix_all.py` for new pure helper `fix_all.format_ansi_redraw(board: str, *, line_count: int) -> str`:
  - Assertion requirements: output contains `\x1b[2K` and `\x1b[1A`, and does not contain any other `\x1b[` sequence besides those two.
  - Acceptance: Running `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_format_ansi_redraw_contains_only_erase_and_cursor_up` exits non-zero before implementation.
- [ ] [P7-T2] (REQ-UI-001, TASK-UI-IMPL-004) Implement `scripts/dev_tools/fix_all.py:format_ansi_redraw(board: str, *, line_count: int) -> str` using only line erase (`\x1b[2K`) and cursor-up (`\x1b[1A`) sequences
  - Acceptance: Running `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_format_ansi_redraw_contains_only_erase_and_cursor_up` exits 0.
- [ ] [P7-T3] (REQ-WIN-001, TASK-WIN-TEST-002) Add Pytest unit test `test_is_vt_enabled_for_stream_true_on_non_windows` in `tests/scripts/dev_tools/test_fix_all.py` for new helper `fix_all.is_vt_enabled_for_stream(stream: TextIO) -> bool`:
  - Test setup requirements: monkeypatch `sys.platform` to a non-Windows value.
  - Assertion requirements: returns True.
  - Acceptance: Running `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_is_vt_enabled_for_stream_true_on_non_windows` exits non-zero before implementation.
- [ ] [P7-T4] (REQ-WIN-001, TASK-WIN-IMPL-002) Implement `scripts/dev_tools/fix_all.py:is_vt_enabled_for_stream(stream: TextIO) -> bool` so it returns True on non-Windows platforms and attempts best-effort VT enablement on Windows
  - Acceptance: Running `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_is_vt_enabled_for_stream_true_on_non_windows` exits 0.
- [ ] [P7-T5] (REQ-UI-001, TASK-UI-TEST-005) Add Pytest unit test `test_interactive_mode_emits_ansi_redraw_not_status_lines` in `tests/scripts/dev_tools/test_fix_all.py`:
  - Test setup requirements: call `fix_all.run_fix_all(...)` with a logger stream that reports `isatty() == True` and with VT enabled forced True by monkeypatching `fix_all.is_vt_enabled_for_stream`.
  - Assertion requirements: captured output contains `\x1b[2K` and does not contain the substring `STATUS|branch=`.
  - Acceptance: Running `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_interactive_mode_emits_ansi_redraw_not_status_lines` exits non-zero before implementation.
- [ ] [P7-T6] (REQ-UI-001, TASK-UI-IMPL-005) Update `scripts/dev_tools/fix_all.py:run_fix_all` so it selects interactive rendering when `should_use_interactive_board(isatty=logger.stream.isatty(), vt_enabled=is_vt_enabled_for_stream(logger.stream))` is True and uses `format_ansi_redraw(...)` for redraw output
  - Acceptance: Running `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_interactive_mode_emits_ansi_redraw_not_status_lines` exits 0.

### Phase 8 — Shell “SKIP tests” detection
- [ ] [P8-T1] (REQ-SHELL-001, TASK-SHELL-TEST-001) Add Pytest unit test `test_shell_test_was_skipped_no_test_dirs_message` in `tests/scripts/dev_tools/test_fix_all.py` for new helper `fix_all._shell_test_was_skipped(output: str) -> bool`:
  - Input: output contains exact substring `No shell test directories found; skipping.`
  - Assertion: returns True.
  - Acceptance: Running `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_shell_test_was_skipped_no_test_dirs_message` exits non-zero before implementation.
- [ ] [P8-T2] (REQ-SHELL-001, TASK-SHELL-TEST-002) Add Pytest unit test `test_shell_test_was_skipped_bats_missing_message` in `tests/scripts/dev_tools/test_fix_all.py`:
  - Input: output contains exact substring `bats not installed; skipping shell tests.`
  - Assertion: returns True.
  - Acceptance: Running `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_shell_test_was_skipped_bats_missing_message` exits non-zero before implementation.
- [ ] [P8-T3] (REQ-SHELL-001, TASK-SHELL-IMPL-001) Implement `scripts/dev_tools/fix_all.py:_shell_test_was_skipped(output: str) -> bool` so it returns True when either skip substring is present
  - Acceptance: Running `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_shell_test_was_skipped_no_test_dirs_message` exits 0.
- [ ] [P8-T4] (REQ-SHELL-001, TASK-SHELL-TEST-003) Add Pytest unit test `test_shell_branch_emits_skip_tests_status_on_skip_output` in `tests/scripts/dev_tools/test_fix_all.py`:
  - Test setup requirements: configure shell runner so `Shell: test` returns exit code 0 and output includes one of the skip substrings.
  - Assertion requirements: captured status output includes `STATUS|branch=shell|status=SKIP tests`.
  - Acceptance: Running `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_shell_branch_emits_skip_tests_status_on_skip_output` exits non-zero before implementation.
- [ ] [P8-T5] (REQ-SHELL-001, TASK-SHELL-IMPL-002) Update `scripts/dev_tools/fix_all.py:run_shell_branch` so that when step `Shell: test` returns exit code 0 and `_shell_test_was_skipped(result.output)` is True, it emits the status `SKIP tests` (while keeping the branch overall `success=True`)
  - Acceptance: Running `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_shell_branch_emits_skip_tests_status_on_skip_output` exits 0.

### Phase 9 — Preserve end-of-run logs + final summary shape
- [ ] [P9-T1] (REQ-OUTPUT-001, TASK-OUTPUT-TEST-001) Add Pytest regression test `test_final_summary_framing_lines_present` in `tests/scripts/dev_tools/test_fix_all.py` asserting the main logger output contains the exact summary framing lines:
  - `========== Branch Results ==========`
  - `====================================`
  - Acceptance: Running `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k test_final_summary_framing_lines_present` exits 0.

### Phase 10 — Final QA toolchain loop (mandatory)
- [ ] [P10-T1] (REQ-QA-001, TASK-QA-001) Run formatter: `poetry run black .` and restart this phase if formatting changes occur
  - Acceptance: Black completes with no changes in the final pass.
- [ ] [P10-T2] (REQ-QA-002, TASK-QA-002) Run linter: `poetry run ruff check` and if Ruff reports issues or applies fixes, restart at [P10-T1]
  - Acceptance: Ruff completes with exit code 0 in the final pass.
- [ ] [P10-T3] (REQ-QA-003, TASK-QA-003) Run type checker: `poetry run pyright` and if it fails, fix and restart at [P10-T1]
  - Acceptance: Pyright completes with exit code 0 in the final pass.
- [ ] [P10-T4] (REQ-QA-004, TASK-QA-004) Run tests with coverage: `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing` and if tests fail, fix and restart at [P10-T1]
  - Acceptance: Pytest completes with exit code 0 in the final pass.

## Test Plan

- Unit (Pytest):
  - `tests/scripts/dev_tools/test_fix_all.py`:
    - `parse_args` recognizes `--complete-all`.
    - Fail-fast stops other branches at step boundaries after first failure.
    - `--complete-all` disables cancellation.
    - Pure rendering functions return deterministic output.
    - Shell skip detection returns `SKIP tests` status when appropriate.
- Integration (CLI, non-gating notes):
  - `poetry run python -m scripts.dev_tools.fix_all`
  - `poetry run python -m scripts.dev_tools.fix_all --complete-all`
  - Redirected output: `poetry run python -m scripts.dev_tools.fix_all > artifacts/qa/fix_all_84_redirected.txt`
- Manual/CLI (Windows terminals, non-gating notes):
  - Run in PowerShell and Windows Terminal.

## Open Questions / Notes

- Implementation note: do not add new runtime dependencies (e.g., `rich`) for this feature; keep status rendering standard-library-only.
