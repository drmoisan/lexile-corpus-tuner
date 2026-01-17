---
title: "2026-01-17-refactor-atomic-executor-cli-length - Refactor Plan"
issue: "#91"
owner: "drmoisan"
status: "Planned"
status_color: "blue"
last_updated: "2026-01-17"
---

# 2026-01-17-refactor-atomic-executor-cli-length - Refactor Plan

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

- Issue: #91
- Parent Initiative (optional): None
- Owner: drmoisan
- Last Updated: 2026-01-17

## Required References (read, do not restate)

- Repo instructions: [`.github/copilot-instructions.md`](../../../.github/copilot-instructions.md)
- Code change policy: [`.github/instructions/general-code-change.instructions.md`](../../../.github/instructions/general-code-change.instructions.md)
- Unit test policy: [`.github/instructions/general-unit-test.instructions.md`](../../../.github/instructions/general-unit-test.instructions.md)
- Python code policy: [`.github/instructions/python-code-change.instructions.md`](../../../.github/instructions/python-code-change.instructions.md)
- Python unit test policy: [`.github/instructions/python-unit-test.instructions.md`](../../../.github/instructions/python-unit-test.instructions.md)
- Python suppression policy: [`.github/instructions/python-suppressions.instructions.md`](../../../.github/instructions/python-suppressions.instructions.md)
- Python docstring/comment policy: [`.github/instructions/self-explanatory-code-commenting.instructions.md`](../../../.github/instructions/self-explanatory-code-commenting.instructions.md)

## Strategy

Split `scripts/dev_tools/atomic_executor/cli.py` into cohesive modules per `spec.md`, move tests to 1:1 module-aligned files in a TDD order (tests before code), and update all imports/patch targets with no re-export shims.

### Requirements

| REQ-ID | Description | Source |
| --- | --- | --- |
| REQ-01 | Split `scripts/dev_tools/atomic_executor/cli.py` into cohesive modules; each module <= 500 lines. | `spec.md` Scope + DoD |
| REQ-02 | Preserve CLI behavior, exit codes, output strings, and lock/QC semantics. | `spec.md` Invariants |
| REQ-03 | No re-export shims; update all imports/patch targets and entry points to new module paths. | `spec.md` Invariants + Scope |
| REQ-04 | Unit tests must be 1:1 with production modules; integration tests only when explicitly labeled. | `spec.md` Scope |
| REQ-05 | Keep CLI entry point at `scripts.dev_tools.atomic_executor.cli:main` unless explicitly changed. | `spec.md` Dependencies |
| REQ-06 | Update documentation references only if module paths or entry points change. | `spec.md` Dependencies |
| REQ-07 | No new dependencies; keep argparse. | `spec.md` Non-Goals |
| REQ-08 | Run full toolchain loop: Black → Ruff → Pyright → Pytest. | `general-code-change.instructions.md` |

## Work Breakdown

### Phase 0 — Context & Inputs
- [ ] [P0-T1] Create `artifacts/plan-notes/phase0.md` with sections `Policy Reads` and `Baseline Results`.
	- Acceptance: `artifacts/plan-notes/phase0.md` exists with both section headings.
- [ ] [P0-T2] Read `.github/copilot-instructions.md` to confirm repo-level policies (REQ-02, REQ-08).
	- Acceptance: `artifacts/plan-notes/phase0.md` contains a line `P0-T2: .github/copilot-instructions.md` under `Policy Reads`.
- [ ] [P0-T3] Read `.github/instructions/general-code-change.instructions.md` to confirm plan-first + toolchain loop requirements (REQ-08).
	- Acceptance: `artifacts/plan-notes/phase0.md` contains a line `P0-T3: .github/instructions/general-code-change.instructions.md` under `Policy Reads`.
- [ ] [P0-T4] Read `.github/instructions/general-unit-test.instructions.md` to confirm unit test constraints (REQ-04).
	- Acceptance: `artifacts/plan-notes/phase0.md` contains a line `P0-T4: .github/instructions/general-unit-test.instructions.md` under `Policy Reads`.
- [ ] [P0-T5] Read `.github/instructions/python-code-change.instructions.md` to confirm Python typing + tooling rules (REQ-01, REQ-08).
	- Acceptance: `artifacts/plan-notes/phase0.md` contains a line `P0-T5: .github/instructions/python-code-change.instructions.md` under `Policy Reads`.
- [ ] [P0-T6] Read `.github/instructions/python-unit-test.instructions.md` to confirm Pytest + coverage commands (REQ-04, REQ-08).
	- Acceptance: `artifacts/plan-notes/phase0.md` contains a line `P0-T6: .github/instructions/python-unit-test.instructions.md` under `Policy Reads`.
- [ ] [P0-T7] Read `.github/instructions/python-suppressions.instructions.md` to confirm suppression rules (REQ-02, REQ-08).
	- Acceptance: `artifacts/plan-notes/phase0.md` contains a line `P0-T7: .github/instructions/python-suppressions.instructions.md` under `Policy Reads`.
- [ ] [P0-T8] Read `.github/instructions/self-explanatory-code-commenting.instructions.md` to confirm docstring/comment requirements (REQ-02).
	- Acceptance: `artifacts/plan-notes/phase0.md` contains a line `P0-T8: .github/instructions/self-explanatory-code-commenting.instructions.md` under `Policy Reads`.
- [ ] [P0-T9] Read `docs/features/active/2026-01-17-refactor-atomic-executor-cli-length-91/spec.md` for invariants and scope (REQ-01..REQ-07).
	- Acceptance: `artifacts/plan-notes/phase0.md` contains a line `P0-T9: spec.md` under `Policy Reads`.
- [ ] [P0-T10] Read `docs/features/active/2026-01-17-refactor-atomic-executor-cli-length-91/20260117-refactor-atomic-executor-cli-length-implementation-research.md` for public surface findings.
	- Acceptance: `artifacts/plan-notes/phase0.md` contains a line `P0-T10: implementation-research.md` under `Policy Reads`.
- [ ] [P0-T11] Capture baseline Ruff results by running `poetry run ruff check` (REQ-08).
	- Acceptance: `artifacts/plan-notes/phase0.md` contains a line starting with `P0-T11: ruff check exit=` under `Baseline Results`.
- [ ] [P0-T12] Capture baseline Pyright results by running `poetry run pyright` (REQ-08).
	- Acceptance: `artifacts/plan-notes/phase0.md` contains a line starting with `P0-T12: pyright exit=` under `Baseline Results`.
- [ ] [P0-T13] Capture baseline Pytest results by running `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing` (REQ-08).
	- Acceptance: `artifacts/plan-notes/phase0.md` contains a line starting with `P0-T13: pytest exit=` and containing `coverage=` under `Baseline Results`.

### Phase 1 — TDD: `cli_args.py` unit tests
- [ ] [P1-T1] Create `tests/scripts/dev_tools/atomic_executor/test_cli_args.py` with module docstring, imports `parse_args` from `scripts.dev_tools.atomic_executor.cli_args`, and class `TestParseArgs` containing `test_parse_execute_subcommand_with_path` (from `tests/scripts/dev_tools/atomic_executor/test_cli.py`).
  - Acceptance: `tests/scripts/dev_tools/atomic_executor/test_cli_args.py` exists and imports `parse_args` from `scripts.dev_tools.atomic_executor.cli_args`.
- [ ] [P1-T2] Move `test_parse_resume_subcommand` into `tests/scripts/dev_tools/atomic_executor/test_cli_args.py::TestParseArgs` (depends on [P1-T1]).
	- Acceptance: Test removed from original file and appears in new file; import path targets `scripts.dev_tools.atomic_executor.cli_args.parse_args`.
- [ ] [P1-T3] Move `test_parse_with_all_optional_args` into `tests/scripts/dev_tools/atomic_executor/test_cli_args.py::TestParseArgs` (depends on [P1-T1]).
	- Acceptance: Test moved with `parse_args` import updated to `cli_args`.
- [ ] [P1-T4] Move `test_parse_copy_prompt_flag` into `tests/scripts/dev_tools/atomic_executor/test_cli_args.py::TestParseArgs` (depends on [P1-T1]).
	- Acceptance: Test moved with `parse_args` import updated to `cli_args`.
- [ ] [P1-T5] Move `test_parse_raises_for_missing_subcommand` into `tests/scripts/dev_tools/atomic_executor/test_cli_args.py::TestParseArgs` (depends on [P1-T1]).
	- Acceptance: Test moved with `parse_args` import updated to `cli_args`.
- [ ] [P1-T6] Move `test_parse_raises_for_missing_path` into `tests/scripts/dev_tools/atomic_executor/test_cli_args.py::TestParseArgs` (depends on [P1-T1]).
	- Acceptance: Test moved with `parse_args` import updated to `cli_args`.

### Phase 2 — TDD: `workspace_checks.py` unit tests
- [ ] [P2-T1] Create `tests/scripts/dev_tools/atomic_executor/test_workspace_checks.py` with module docstring, imports `ensure_clean_tree` and `refuse_protected_branch` from `scripts.dev_tools.atomic_executor.workspace_checks`, and class `TestEnsureCleanTree` containing `test_ensure_clean_tree_passes_for_clean_tree` (from `test_cli.py`).
  - Acceptance: `tests/scripts/dev_tools/atomic_executor/test_workspace_checks.py` exists and imports from `scripts.dev_tools.atomic_executor.workspace_checks`.
- [ ] [P2-T2] Move `test_ensure_clean_tree_raises_for_dirty_tree` into `tests/scripts/dev_tools/atomic_executor/test_workspace_checks.py::TestEnsureCleanTree` (depends on [P2-T1]).
	- Acceptance: Test moved with updated import path to `workspace_checks.ensure_clean_tree`.
- [ ] [P2-T3] Move `test_refuse_raises_for_main_branch` into `tests/scripts/dev_tools/atomic_executor/test_workspace_checks.py::TestRefuseProtectedBranch` (depends on [P2-T1]).
	- Acceptance: Test moved with updated import path to `workspace_checks.refuse_protected_branch`.
- [ ] [P2-T4] Move `test_refuse_raises_for_master_branch` into `tests/scripts/dev_tools/atomic_executor/test_workspace_checks.py::TestRefuseProtectedBranch` (depends on [P2-T1]).
	- Acceptance: Test moved with updated import path to `workspace_checks.refuse_protected_branch`.
- [ ] [P2-T5] Move `test_refuse_raises_for_development_branch` into `tests/scripts/dev_tools/atomic_executor/test_workspace_checks.py::TestRefuseProtectedBranch` (depends on [P2-T1]).
	- Acceptance: Test moved with updated import path to `workspace_checks.refuse_protected_branch`.
- [ ] [P2-T6] Move `test_refuse_passes_for_feature_branch` into `tests/scripts/dev_tools/atomic_executor/test_workspace_checks.py::TestRefuseProtectedBranch` (depends on [P2-T1]).
	- Acceptance: Test moved with updated import path to `workspace_checks.refuse_protected_branch`.
- [ ] [P2-T7] Move `test_refuse_handles_git_error` into `tests/scripts/dev_tools/atomic_executor/test_workspace_checks.py::TestRefuseProtectedBranch` (depends on [P2-T1]).
	- Acceptance: Test moved with updated import path to `workspace_checks.refuse_protected_branch`.

### Phase 3 — TDD: `clipboard.py` unit tests
- [ ] [P3-T1] Create `tests/scripts/dev_tools/atomic_executor/test_clipboard.py` with module docstring, imports `copy_to_clipboard` and `get_clipboard_command` from `scripts.dev_tools.atomic_executor.clipboard`, and class `TestCopyToClipboard` containing `test_copy_uses_pyperclip_when_available` (from `test_cli.py`).
  - Acceptance: `tests/scripts/dev_tools/atomic_executor/test_clipboard.py` exists and imports from `scripts.dev_tools.atomic_executor.clipboard`.
- [ ] [P3-T2] Move `test_copy_falls_back_to_command_when_pyperclip_unavailable` into `tests/scripts/dev_tools/atomic_executor/test_clipboard.py::TestCopyToClipboard` (depends on [P3-T1]).
	- Acceptance: Test moved with updated import path to `clipboard.copy_to_clipboard`.
- [ ] [P3-T3] Move `test_copy_returns_false_when_all_methods_fail` into `tests/scripts/dev_tools/atomic_executor/test_clipboard.py::TestCopyToClipboard` (depends on [P3-T1]).
	- Acceptance: Test moved with updated import path to `clipboard.copy_to_clipboard`.
- [ ] [P3-T4] Move `test_copy_tries_multiple_fallback_commands` into `tests/scripts/dev_tools/atomic_executor/test_clipboard.py::TestCopyToClipboard` (depends on [P3-T1]).
	- Acceptance: Test moved with updated import path to `clipboard.copy_to_clipboard`.

### Phase 4 — TDD: `copilot_executor.py` unit tests
- [ ] [P4-T1] Create `tests/scripts/dev_tools/atomic_executor/test_copilot_executor.py` with module docstring, imports `run_copilot` and `CopilotPermissionDeniedError` from `scripts.dev_tools.atomic_executor.copilot_executor`, and class `TestRunCopilot` containing `test_run_copilot_raises_when_executable_not_found` (from `test_cli.py`).
  - Acceptance: `tests/scripts/dev_tools/atomic_executor/test_copilot_executor.py` exists and imports from `scripts.dev_tools.atomic_executor.copilot_executor`.
- [ ] [P4-T2] Move `test_run_copilot_rejects_vscode_shim` into `tests/scripts/dev_tools/atomic_executor/test_copilot_executor.py::TestRunCopilot` (depends on [P4-T1]).
	- Acceptance: Test moved with updated import path to `copilot_executor.run_copilot`.
- [ ] [P4-T3] Move `test_run_copilot_rejects_vscode_shim_remote_paths` into `tests/scripts/dev_tools/atomic_executor/test_copilot_executor.py::TestRunCopilot` (depends on [P4-T1]).
	- Acceptance: Test moved with updated import path to `copilot_executor.run_copilot`.
- [ ] [P4-T4] Move `test_run_copilot_creates_log_directory` into `tests/scripts/dev_tools/atomic_executor/test_copilot_executor.py::TestRunCopilot` (depends on [P4-T1]).
	- Acceptance: Test moved with updated import path to `copilot_executor.run_copilot`.
- [ ] [P4-T5] Move `test_run_copilot_invokes_with_correct_arguments` into `tests/scripts/dev_tools/atomic_executor/test_copilot_executor.py::TestRunCopilot` (depends on [P4-T1]).
	- Acceptance: Test moved with updated import path to `copilot_executor.run_copilot`.
- [ ] [P4-T6] Move `test_run_copilot_permission_denied_fails_fast_with_actionable_error` into `tests/scripts/dev_tools/atomic_executor/test_copilot_executor.py::TestRunCopilot` (depends on [P4-T1]).
	- Acceptance: Test moved with updated import path to `copilot_executor.run_copilot`.
- [ ] [P4-T7] Move `test_run_copilot_reuses_session_when_requested` into `tests/scripts/dev_tools/atomic_executor/test_copilot_executor.py::TestRunCopilot` (depends on [P4-T1]).
	- Acceptance: Test moved with updated import path to `copilot_executor.run_copilot`.
- [ ] [P4-T8] Move `test_run_copilot_skips_session_when_not_supported` into `tests/scripts/dev_tools/atomic_executor/test_copilot_executor.py::TestRunCopilot` (depends on [P4-T1]).
	- Acceptance: Test moved with updated import path to `copilot_executor.run_copilot`.
- [ ] [P4-T9] Move `test_run_copilot_times_out_when_cli_is_idle` into `tests/scripts/dev_tools/atomic_executor/test_copilot_executor.py::TestRunCopilot` (depends on [P4-T1]).
	- Acceptance: Test moved with updated import path to `copilot_executor.run_copilot`.

### Phase 5 — TDD: `cli.py` unit tests (resolve_workspace + main)
- [ ] [P5-T1] Update `tests/scripts/dev_tools/atomic_executor/test_cli.py` imports to remove `parse_args`, `ensure_clean_tree`, `refuse_protected_branch`, and `copy_to_clipboard` (now in other modules) and keep `resolve_workspace` imported from `scripts.dev_tools.atomic_executor.cli` (REQ-03).
	- Acceptance: `test_cli.py` imports `resolve_workspace` from `scripts.dev_tools.atomic_executor.cli` and contains no imports of `parse_args`, `ensure_clean_tree`, `refuse_protected_branch`, or `copy_to_clipboard` from `cli`.
- [ ] [P5-T2] Keep `test_resolve_uses_explicit_workspace` in `tests/scripts/dev_tools/atomic_executor/test_cli.py::TestResolveWorkspace` with import of `resolve_workspace` from `scripts.dev_tools.atomic_executor.cli`.
	- Acceptance: `test_cli.py` contains `def test_resolve_uses_explicit_workspace` and imports `resolve_workspace` from `scripts.dev_tools.atomic_executor.cli`.
- [ ] [P5-T3] Keep `test_resolve_infers_from_file_location` in `tests/scripts/dev_tools/atomic_executor/test_cli.py::TestResolveWorkspace` with updated monkeypatch target `scripts.dev_tools.atomic_executor.cli.__file__`.
	- Acceptance: `test_cli.py` contains `scripts.dev_tools.atomic_executor.cli.__file__` inside `test_resolve_infers_from_file_location`.
- [ ] [P5-T4] Keep `test_main_exits_early_with_print_prompt` in `tests/scripts/dev_tools/atomic_executor/test_cli.py::TestMainEdgeCases` (REQ-03).
	- Acceptance: `test_cli.py` contains `def test_main_exits_early_with_print_prompt` and imports `main` from `scripts.dev_tools.atomic_executor.cli` inside the test.
- [ ] [P5-T5] Keep `test_main_exits_early_with_copy_prompt` in `tests/scripts/dev_tools/atomic_executor/test_cli.py::TestMainEdgeCases` and update patch targets to new module paths (REQ-03).
	- Acceptance: `test_cli.py` contains `def test_main_exits_early_with_copy_prompt` and monkeypatches `scripts.dev_tools.atomic_executor.clipboard.copy_to_clipboard`.
- [ ] [P5-T6] Keep `test_main_returns_error_for_missing_plan` in `tests/scripts/dev_tools/atomic_executor/test_cli.py::TestMainEdgeCases` (REQ-03).
	- Acceptance: `test_cli.py` contains `def test_main_returns_error_for_missing_plan` and imports `main` from `scripts.dev_tools.atomic_executor.cli` inside the test.
- [ ] [P5-T7] Keep `test_main_returns_zero_when_plan_already_complete` in `tests/scripts/dev_tools/atomic_executor/test_cli.py::TestMainEdgeCases` (REQ-03).
	- Acceptance: `test_cli.py` contains `def test_main_returns_zero_when_plan_already_complete` and imports `main` from `scripts.dev_tools.atomic_executor.cli` inside the test.
- [ ] [P5-T8] Keep `test_main_returns_error_for_missing_template` in `tests/scripts/dev_tools/atomic_executor/test_cli.py::TestMainEdgeCases` (REQ-03).
	- Acceptance: `test_cli.py` contains `def test_main_returns_error_for_missing_template` and imports `main` from `scripts.dev_tools.atomic_executor.cli` inside the test.
- [ ] [P5-T9] Keep `test_main_with_copy_prompt_fallback_when_clipboard_fails` in `tests/scripts/dev_tools/atomic_executor/test_cli.py::TestMainEdgeCases` with updated patch target to `scripts.dev_tools.atomic_executor.clipboard.copy_to_clipboard` where applicable (REQ-03).
	- Acceptance: `test_cli.py` contains `def test_main_with_copy_prompt_fallback_when_clipboard_fails` and monkeypatches `scripts.dev_tools.atomic_executor.clipboard.copy_to_clipboard`.
- [ ] [P5-T10] Keep `test_main_execute_with_start_flag` in `tests/scripts/dev_tools/atomic_executor/test_cli.py::TestMainEdgeCases` (REQ-03).
	- Acceptance: `test_cli.py` contains `def test_main_execute_with_start_flag` and imports `main` from `scripts.dev_tools.atomic_executor.cli` inside the test.
- [ ] [P5-T11] Keep `test_main_execute_when_all_tasks_complete` in `tests/scripts/dev_tools/atomic_executor/test_cli.py::TestMainEdgeCases` (REQ-03).
	- Acceptance: `test_cli.py` contains `def test_main_execute_when_all_tasks_complete` and imports `main` from `scripts.dev_tools.atomic_executor.cli` inside the test.
- [ ] [P5-T12] Keep `test_main_successful_execution_with_scoped_qc` in `tests/scripts/dev_tools/atomic_executor/test_cli.py::TestMainEdgeCases` with patch targets updated to new module paths (e.g., `copilot_executor._copilot_supports_session`) (REQ-03).
	- Acceptance: `test_cli.py` contains `def test_main_successful_execution_with_scoped_qc` and monkeypatches `scripts.dev_tools.atomic_executor.copilot_executor._copilot_supports_session`.

### Phase 6 — TDD: `executor_lock.py` unit tests
- [ ] [P6-T1] Create `tests/scripts/dev_tools/atomic_executor/test_executor_lock.py` with module docstring, imports `acquire_executor_lock` and `release_executor_lock` from `scripts.dev_tools.atomic_executor.executor_lock`, and include `test_single_run_lock_acquired_on_start` (moved from `tests/scripts/dev_tools/test_atomic_executor_cli.py`).
  - Acceptance: `tests/scripts/dev_tools/atomic_executor/test_executor_lock.py` exists and imports from `scripts.dev_tools.atomic_executor.executor_lock`.
- [ ] [P6-T2] Move `test_single_run_lock_blocks_concurrent_run` into `tests/scripts/dev_tools/atomic_executor/test_executor_lock.py` (depends on [P6-T1]).
	- Acceptance: Test moved with updated import path to `executor_lock.acquire_executor_lock`.
- [ ] [P6-T3] Move `test_single_run_lock_released_on_completion` into `tests/scripts/dev_tools/atomic_executor/test_executor_lock.py` (depends on [P6-T1]).
	- Acceptance: Test moved with updated import path to `executor_lock.release_executor_lock`.

### Phase 7 — TDD: `task_executor.py` unit tests
- [ ] [P7-T1] Create `tests/scripts/dev_tools/atomic_executor/test_task_executor.py` with module docstring and imports `execute_one_task` from `scripts.dev_tools.atomic_executor.task_executor`.
  - Acceptance: `tests/scripts/dev_tools/atomic_executor/test_task_executor.py` exists and imports `execute_one_task` from `scripts.dev_tools.atomic_executor.task_executor`.
- [ ] [P7-T2] Refactor `test_execute_one_task_retries_until_success` from `tests/scripts/dev_tools/test_atomic_executor_cli.py` to call `task_executor.execute_one_task` directly with mocked `PlanParser`, `QCRunner`, and `run_copilot` (depends on [P7-T1]).
	- Acceptance: Test name `test_execute_one_task_retries_until_success` exists in `test_task_executor.py` and references `task_executor.execute_one_task`.

### Phase 8 — TDD: Move remaining CLI integration tests
- [ ] [P8-T1] Move `test_execute_all_runs_full_qc_after_phase` into `tests/scripts/dev_tools/atomic_executor/test_cli.py` and update imports/patch targets to new module paths (REQ-03).
	- Acceptance: Test exists in `test_cli.py` and imports `main` from `scripts.dev_tools.atomic_executor.cli`.
- [ ] [P8-T2] Move `test_execute_all_respects_infinite_retry` into `tests/scripts/dev_tools/atomic_executor/test_cli.py` and update patch targets to new module paths (REQ-03).
	- Acceptance: Test exists in `test_cli.py` and imports `main` from `cli`.
- [ ] [P8-T3] Move `test_execute_all_aborts_with_exit_code_5_on_persistent_failure` into `tests/scripts/dev_tools/atomic_executor/test_cli.py` and update patch targets (REQ-03).
	- Acceptance: Test exists in `test_cli.py` and uses updated patch targets.
- [ ] [P8-T4] Move `test_copilot_argv_includes_agent_flag` into `tests/scripts/dev_tools/atomic_executor/test_copilot_executor.py` and update import to `copilot_executor.run_copilot` (REQ-03).
	- Acceptance: Test exists in `test_copilot_executor.py` with import from `copilot_executor`.
- [ ] [P8-T5] Move `test_first_task_omits_continue_flag` into `tests/scripts/dev_tools/atomic_executor/test_copilot_executor.py` and update import to `copilot_executor.run_copilot` (REQ-03).
	- Acceptance: Test exists in `test_copilot_executor.py` with import from `copilot_executor`.
- [ ] [P8-T6] Move `test_subsequent_task_includes_continue_flag` into `tests/scripts/dev_tools/atomic_executor/test_copilot_executor.py` and update import to `copilot_executor.run_copilot` (REQ-03).
	- Acceptance: Test exists in `test_copilot_executor.py` with import from `copilot_executor`.
- [ ] [P8-T7] Remove `tests/scripts/dev_tools/test_atomic_executor_cli.py` after all tests have been migrated (REQ-04).
	- Acceptance: File deleted and no references remain in the repo.

### Phase 9 — Production code split (create modules)
- [ ] [P9-T1] Create `scripts/dev_tools/atomic_executor/cli_args.py` and move `add_common` (line ~84) from `cli.py`, preserving signature and docstring (REQ-01, REQ-02).
	- Acceptance: `cli_args.py` defines `add_common(sp: argparse.ArgumentParser) -> None` with identical logic.
- [ ] [P9-T2] Move `parse_args` (line ~71) into `scripts/dev_tools/atomic_executor/cli_args.py`, preserving signature and docstring (REQ-01, REQ-02).
	- Acceptance: `cli_args.py` defines `parse_args(argv: list[str]) -> argparse.Namespace` with identical logic.
- [ ] [P9-T3] Create `scripts/dev_tools/atomic_executor/workspace_checks.py` and move `_current_branch` (line ~289) from `cli.py` with unchanged behavior (REQ-01, REQ-02).
	- Acceptance: `workspace_checks.py` defines `_current_branch(repo: Repo) -> str` with identical logic.
- [ ] [P9-T4] Move `ensure_clean_tree` (line ~244) into `workspace_checks.py` with unchanged behavior (REQ-01, REQ-02).
	- Acceptance: `workspace_checks.py` defines `ensure_clean_tree` with identical logic.
- [ ] [P9-T5] Move `refuse_protected_branch` (line ~274) into `workspace_checks.py` with unchanged behavior (REQ-01, REQ-02).
	- Acceptance: `workspace_checks.py` defines `refuse_protected_branch` with identical logic.
- [ ] [P9-T6] Create `scripts/dev_tools/atomic_executor/executor_lock.py` and move `acquire_executor_lock` (line ~203) from `cli.py` with unchanged behavior (REQ-01, REQ-02).
	- Acceptance: `executor_lock.py` defines `acquire_executor_lock` with identical logic.
- [ ] [P9-T7] Move `release_executor_lock` (line ~232) into `executor_lock.py` with unchanged behavior (REQ-01, REQ-02).
	- Acceptance: `executor_lock.py` defines `release_executor_lock` with identical logic.
- [ ] [P9-T8] Create `scripts/dev_tools/atomic_executor/clipboard.py` and move `get_clipboard_command` (line ~310) from `cli.py` with unchanged behavior (REQ-01, REQ-02).
	- Acceptance: `clipboard.py` defines `get_clipboard_command() -> str | None` with identical logic.
- [ ] [P9-T9] Move `copy_to_clipboard` (line ~362) into `clipboard.py` with unchanged behavior (REQ-01, REQ-02).
	- Acceptance: `clipboard.py` defines `copy_to_clipboard(text: str) -> bool` with identical logic.
- [ ] [P9-T10] Create `scripts/dev_tools/atomic_executor/copilot_executor.py` and move `CopilotPermissionDeniedError` (line ~61) from `cli.py` with unchanged behavior (REQ-01, REQ-02).
	- Acceptance: `copilot_executor.py` defines `CopilotPermissionDeniedError` class with identical behavior.
- [ ] [P9-T11] Move `_resolve_idle_timeout_seconds` (line ~700) into `copilot_executor.py` with unchanged behavior (REQ-01, REQ-02).
	- Acceptance: `copilot_executor.py` defines `_resolve_idle_timeout_seconds` with identical logic.
- [ ] [P9-T12] Move `_stream_copilot_output` (line ~728) into `copilot_executor.py` with unchanged behavior (REQ-01, REQ-02).
	- Acceptance: `copilot_executor.py` defines `_stream_copilot_output` with identical logic.
- [ ] [P9-T13] Move `_copilot_supports_session` (line ~891) into `copilot_executor.py` with unchanged behavior (REQ-01, REQ-02).
	- Acceptance: `copilot_executor.py` defines `_copilot_supports_session` with identical logic.
- [ ] [P9-T14] Move `_clean_session_file` (line ~911) into `copilot_executor.py` with unchanged behavior (REQ-01, REQ-02).
	- Acceptance: `copilot_executor.py` defines `_clean_session_file` with identical logic.
- [ ] [P9-T15] Move `run_copilot` (line ~424) into `copilot_executor.py` with unchanged behavior (REQ-01, REQ-02).
	- Acceptance: `copilot_executor.py` defines `run_copilot` with identical logic.
- [ ] [P9-T16] Create `scripts/dev_tools/atomic_executor/task_executor.py` and move `execute_one_task` (line ~948) from `cli.py` with unchanged logic (REQ-01, REQ-02).
	- Acceptance: `task_executor.py` defines `execute_one_task` with the same signature and behavior.

### Phase 10 — Wire-up imports and entry points
- [ ] [P10-T1] Update `scripts/dev_tools/atomic_executor/cli.py` to import `parse_args` from `cli_args`, `ensure_clean_tree/refuse_protected_branch` from `workspace_checks`, `acquire_executor_lock/release_executor_lock` from `executor_lock`, `copy_to_clipboard` from `clipboard`, `run_copilot` and helpers from `copilot_executor`, and `execute_one_task` from `task_executor` (REQ-03, REQ-05).
	- Acceptance: `cli.py` no longer defines moved functions and imports them from new modules.
- [ ] [P10-T2] Run `rg "atomic_executor\.cli\.(parse_args|ensure_clean_tree|refuse_protected_branch|get_clipboard_command|copy_to_clipboard|run_copilot|execute_one_task)" tests` and record output in `artifacts/plan-notes/phase10.md` (REQ-03).
	- Acceptance: `artifacts/plan-notes/phase10.md` exists and indicates zero matches.
- [ ] [P10-T3] Ensure `scripts/dev_tools/atomic_executor/__init__.py` exports only `main`, `PlanParser`, `PromptBuilder`, `QCRunner`, and `FeatureResolver`, and does not import moved helpers (REQ-05).
	- Acceptance: `__init__.py` contains no imports of `cli_args`, `workspace_checks`, `executor_lock`, `clipboard`, `copilot_executor`, or `task_executor`.
- [ ] [P10-T4] Confirm `pyproject.toml` console scripts reference `scripts.dev_tools.atomic_executor.cli:main` (REQ-05).
	- Acceptance: `pyproject.toml` contains the exact entry point string `scripts.dev_tools.atomic_executor.cli:main`.

### Phase 11 — QA verification loop
- [ ] [P11-T1] Run `poetry run black .`; if any files change, re-run from [P11-T1] after applying changes (REQ-08).
	- Acceptance: Black exits with code 0 and no file changes in the final pass.
- [ ] [P11-T2] Run `poetry run ruff check`; if any errors or fixes, resolve and re-run from [P11-T1] (REQ-08).
	- Acceptance: Ruff exits with code 0 and no findings in the final pass.
- [ ] [P11-T3] Run `poetry run pyright`; if any errors, resolve and re-run from [P11-T1] (REQ-08).
	- Acceptance: Pyright exits with code 0 and no errors in the final pass.
- [ ] [P11-T4] Run `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`; if any failures, resolve and re-run from [P11-T1] (REQ-08).
	- Acceptance: Pytest exits with code 0 in the final pass.

### Phase 12 — Documentation and cleanup
- [ ] [P12-T1] Run `grep -R "python -m scripts.dev_tools.atomic_executor.cli" docs` and confirm all matches still reference the active CLI entry point (REQ-06).
  - Acceptance: Grep output shows only `python -m scripts.dev_tools.atomic_executor.cli` or is empty if no docs reference exists.
- [ ] [P12-T2] Confirm all new modules are <= 500 lines via `wc -l scripts/dev_tools/atomic_executor/*.py` and record results in `artifacts/plan-notes/phase12.md` (REQ-01).
  - Acceptance: `artifacts/plan-notes/phase12.md` lists each module file with line count <= 500.
- [ ] [P12-T3] Run `rg "subprocess\.(run|Popen)" scripts/dev_tools/atomic_executor/*.py` and confirm call sites exist only in `clipboard.py`, `workspace_checks.py`, and `copilot_executor.py` (security check).
	- Acceptance: Ripgrep output lists only `clipboard.py`, `workspace_checks.py`, and `copilot_executor.py` file paths.
- [ ] [P12-T4] Run `rg "time\.sleep|\bsleep\(" scripts/dev_tools/atomic_executor/*.py` to confirm no new explicit delays were introduced (performance check).
	- Acceptance: Ripgrep returns no matches.

## Test Plan

- Unit: new module-aligned tests in `tests/scripts/dev_tools/atomic_executor/` for `cli_args.py`, `workspace_checks.py`, `clipboard.py`, `copilot_executor.py`, `executor_lock.py`, `task_executor.py`, and `cli.py`.
- Integration: only if explicitly labeled; otherwise all tests remain unit-scoped per REQ-04.
- Tooling: run Black, Ruff, Pyright, Pytest in the QA loop (Phase 11).

## Rollback / Contingency

If failures occur after module split, revert by restoring `scripts/dev_tools/atomic_executor/cli.py` from the pre-split commit and delete newly created module files; re-run QA loop to confirm rollback consistency.

## Open Questions / Notes

None.
