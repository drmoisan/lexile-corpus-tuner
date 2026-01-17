<!-- markdownlint-disable-file -->

# Task Research Notes: refactor-atomic-executor-cli-length (Issue #91)

## Research Executed

### File Analysis

- /workspaces/lexile-corpus-tuner/docs/features/active/2026-01-17-refactor-atomic-executor-cli-length-91/issue.md
  - Confirms the CLI module size (1321 lines) violates the 500-line policy; highlights impacted module and low severity.
- /workspaces/lexile-corpus-tuner/docs/features/active/2026-01-17-refactor-atomic-executor-cli-length-91/plan.md
  - Plan template exists but is not filled in; no concrete tasks listed yet.
- /workspaces/lexile-corpus-tuner/docs/features/active/2026-01-17-refactor-atomic-executor-cli-length-91/spec.md
  - Spec template exists but is not filled in; no explicit invariants or scope listed.
- /workspaces/lexile-corpus-tuner/scripts/dev_tools/atomic_executor/cli.py
  - Monolithic CLI orchestrates parsing, workspace resolution, lock handling, clipboard, Copilot streaming, QC orchestration, and run loop; defines core helper functions and the main CLI entry point.
- /workspaces/lexile-corpus-tuner/scripts/dev_tools/atomic_executor/plan_parser.py
  - Encapsulates plan parsing, task lookup, checkbox flipping, and preflight validation; already cohesive.
- /workspaces/lexile-corpus-tuner/scripts/dev_tools/atomic_executor/prompt_builder.py
  - Handles prompt assembly and filesystem abstraction; includes logging and plan resolution.
- /workspaces/lexile-corpus-tuner/scripts/dev_tools/atomic_executor/plan_discovery.py
  - Resolves plan file location (legacy plan.md or timestamped); encapsulates selection logic.
- /workspaces/lexile-corpus-tuner/scripts/dev_tools/atomic_executor/copilot_throttling.py
  - Provides deterministic rate limiting and backoff primitives with injected clock/sleeper/random.
- /workspaces/lexile-corpus-tuner/scripts/dev_tools/atomic_executor/feature_resolver.py
  - Resolves feature directory from path/flag/branch heuristics; depends on plan discovery.
- /workspaces/lexile-corpus-tuner/scripts/dev_tools/atomic_executor/qc_runner.py
  - Encapsulates scoped and full QC toolchain execution.
- /workspaces/lexile-corpus-tuner/scripts/dev_tools/atomic_executor/__init__.py
  - Re-exports CLI main and helper classes, indicating public surface.
- /workspaces/lexile-corpus-tuner/pyproject.toml
  - Defines console scripts `atomic-executor` and `dev.atomic-executor` pointing at `scripts.dev_tools.atomic_executor.cli:main`.
- /workspaces/lexile-corpus-tuner/tests/scripts/dev_tools/atomic_executor/test_cli.py
  - Extensive unit coverage for CLI parsing, clipboard, run_copilot, and main flow; refactor must preserve import paths or update tests.
- /workspaces/lexile-corpus-tuner/tests/scripts/dev_tools/test_atomic_executor_cli.py
  - CLI integration tests patch `scripts.dev_tools.atomic_executor.cli` for parser/QC/copilot/clipboard and call `main` directly.

### Code Search Results

- atomic_executor\.cli|execute_one_task|run_copilot
  - Matches in `scripts/dev_tools/atomic_executor/cli.py`, `scripts/dev_tools/atomic_executor/__init__.py`, and `tests/scripts/dev_tools/atomic_executor/test_cli.py` (primary call sites).
- scripts\.dev_tools\.atomic_executor\.cli
  - Matches in `tests/scripts/dev_tools/atomic_executor/test_cli.py` and `tests/scripts/dev_tools/test_atomic_executor_cli.py` (imports and patch targets).
- python -m scripts\.dev_tools\.atomic_executor\.cli
  - Matches across `docs/developer-tooling.md` and multiple feature specs/plans referencing CLI invocation.

### External Research

- #fetch:https://github.com/drmoisan/lexile-corpus-tuner/issues/91
  - Issue body is an auto-promoted template (no extra acceptance criteria); confirms scope and origin.
- #fetch:https://docs.python.org/3/library/argparse.html
  - Confirms argparse subcommand structure and usage consistency; supports refactoring argument setup into sub-parser helpers.
- #fetch:https://docs.python.org/3/tutorial/modules.html
  - Recommends splitting large scripts into modules for maintainability; supports packaging separation for CLI helpers.
- #fetch:https://docs.python.org/3/howto/argparse.html
  - Provides canonical patterns for subcommands and argument parsing flow; useful for preserving CLI behavior when reorganizing.

### Project Conventions

- Standards referenced: `general-code-change.instructions.md` (500-line limit, docstrings, plan-first workflow), `python-code-change.instructions.md` (typing, Black/Ruff/Pyright), `general-unit-test.instructions.md` and `python-unit-test.instructions.md` (Pytest only, no temp files), `self-explanatory-code-commenting.instructions.md` (mandatory docstrings + intent comments).
- Instructions followed: `research-issue.prompt.md`, repository policy attachments (copilot + code change + unit test policies).

## Key Discoveries

### Project Structure

- `scripts/dev_tools/atomic_executor/cli.py` combines CLI parsing, lock management, clipboard operations, Copilot subprocess streaming, retry/backoff logic, and the main execution loop.
- The package already has cohesive submodules (`plan_parser`, `prompt_builder`, `qc_runner`, `copilot_throttling`, `feature_resolver`) that can be leveraged to extract CLI concerns without new dependencies.
- Tests for CLI behaviors are centralized in `tests/scripts/dev_tools/atomic_executor/test_cli.py` and expect import paths from `scripts.dev_tools.atomic_executor.cli`.
- Integration tests in `tests/scripts/dev_tools/test_atomic_executor_cli.py` patch `scripts.dev_tools.atomic_executor.cli.*` and call `main` directly.

### Implementation Patterns

- CLI uses `argparse` with shared subcommand argument definition (`add_common`) and returns `argparse.Namespace`.
- `run_copilot` includes subprocess invocation, log file management, output streaming with a background thread, and permission-denied detection; tests mock `subprocess.Popen` and `subprocess.run` to assert CLI behavior.
- Locking is handled via `.agent_logs/executor.lock` with a guard that only applies to `execute-all` and a `finally` release.

### Complete Examples

```python
# Source: https://docs.python.org/3/library/argparse.html (Subcommands example)
import argparse

parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--foo', action='store_true', help='foo help')
subparsers = parser.add_subparsers(help='subcommand help')

parser_a = subparsers.add_parser('a', help='a help')
parser_a.add_argument('bar', type=int, help='bar help')

parser_b = subparsers.add_parser('b', help='b help')
parser_b.add_argument('--baz', choices=('X', 'Y', 'Z'), help='baz help')

args = parser.parse_args(['a', '12'])
```

### API and Schema Documentation

- The CLI’s public API surface is implicitly defined by `scripts/dev_tools/atomic_executor/__init__.py` re-exporting `main`; refactor must preserve the entry point semantics and module-level import paths used by tests.
- Console scripts in `pyproject.toml` expose `atomic-executor` and `dev.atomic-executor` as `scripts.dev_tools.atomic_executor.cli:main`.

### Configuration Examples

```text
# No new configuration formats introduced for this refactor.
# Existing CLI flags are defined in scripts/dev_tools/atomic_executor/cli.py::parse_args.
```

### Technical Requirements

- The refactor must reduce each module to <= 500 lines, maintain fully typed Python (Pyright strict), and keep docstrings/comments per repository policy.
- Tests cannot create temporary files at runtime (policy restriction); any helper interfaces for I/O should maintain the existing in-memory/testable design patterns used by `PromptBuilder`.
- Preserve CLI behavior: subcommands, arguments, lock file behavior, Copilot invocation, QC gating logic, and output messages used by tests.
- Update all test imports/patch targets that reference `scripts.dev_tools.atomic_executor.cli` when functions move to new modules (no re-export shims).

**Mandatory unachievable objective callout**:
- None identified. The refactor is feasible within current constraints.

## Recommended Approach

**Recommendation:** Split `scripts/dev_tools/atomic_executor/cli.py` into small cohesive modules under `scripts/dev_tools/atomic_executor/` and turn `cli.py` into a thin orchestration layer that wires dependencies together. This keeps public entry points stable while respecting the 500-line limit and preserving test behavior with targeted import updates.

**Proposed module split (single approach):**

- `cli.py` (thin entry point)
  - `parse_args`, `resolve_workspace`, `main`, and minimal dependency wiring.
- `cli_args.py` (argument parsing)
  - `parse_args` and helper for shared argument definitions.
- `workspace_checks.py` (git + branch guards)
  - `ensure_clean_tree`, `refuse_protected_branch`, `_current_branch`.
- `executor_lock.py` (single-run guard)
  - `acquire_executor_lock`, `release_executor_lock`, constants for lock path.
- `clipboard.py` (clipboard detection and copy)
  - `get_clipboard_command`, `copy_to_clipboard`.
- `copilot_executor.py` (Copilot invocation + stream handling)
  - `run_copilot`, `_stream_copilot_output`, `_resolve_idle_timeout_seconds`, `_copilot_supports_session`, `_clean_session_file`, `CopilotPermissionDeniedError`.
- `task_executor.py` (task loop)
  - `execute_one_task` and related orchestration helpers (no CLI parsing).

**State model (proposed):**

- **States:** `Ready` → `RunningTask` → (`TaskFailed` | `TaskSucceeded`) → `PhaseGate` → `Complete`.
- **Transitions:**
  - `Ready` → `RunningTask`: after plan parsing + task selection.
  - `RunningTask` → `TaskFailed`: non-throttle copilot error, QC failure exceeding attempts.
  - `RunningTask` → `TaskSucceeded`: copilot success + scoped QC pass + checkbox flip.
  - `TaskSucceeded` → `PhaseGate`: when `parser.phase_complete(phase)` returns true.
  - `PhaseGate` → `Complete`: full QC succeeds and no remaining tasks.

**Where updates occur:**

- Checkbox updates happen in `PlanParser.flip_checkbox()` only after scoped QC passes.
- Full QC runs after phase completion, not per task.
- Lock file is created at `execute-all` start and removed in `finally` block.

**Rendering loop pseudocode:**

```text
parse args → resolve workspace → resolve feature → resolve plan
if execute-all: acquire lock
parse plan → choose current task (start/resume/next)
while task exists:
  result = execute_one_task(...)
  if result != 0: exit
  if print/copy prompt: exit
  if phase complete: run full QC
  if not execute-all: exit
  task = next_unchecked
release lock
```

**Implementation hooks (current locations to split):**

- `cli.py` functions to relocate:
  - `parse_args` → `cli_args.py`
  - `ensure_clean_tree`, `refuse_protected_branch`, `_current_branch` → `workspace_checks.py`
  - `acquire_executor_lock`, `release_executor_lock` → `executor_lock.py`
  - `get_clipboard_command`, `copy_to_clipboard` → `clipboard.py`
  - `run_copilot`, `_stream_copilot_output`, `_resolve_idle_timeout_seconds`, `_copilot_supports_session`, `_clean_session_file`, `CopilotPermissionDeniedError` → `copilot_executor.py`
  - `execute_one_task` → `task_executor.py`
  - `main` remains in `cli.py` but imports the new module functions.

**Risks and mitigations:**

- **Risk:** Tests expect `scripts.dev_tools.atomic_executor.cli` to export functions currently imported by tests.
  - **Mitigation:** Re-export relocated functions from `cli.py` or update tests to import new module paths.
- **Risk:** Breaking CLI output or lock behavior via refactor.
  - **Mitigation:** Keep exact function signatures and reuse existing logic; add focused regression tests for any new modules.
- **Risk:** New files exceed 500 lines if split incorrectly.
  - **Mitigation:** Split by responsibility and keep helper functions close to their domain.

**Rejected alternatives (brief):**

- **Single-file internal reordering:** Rejected because it cannot satisfy the 500-line policy and offers no structural reduction.
- **New dependency-based CLI framework (e.g., Typer/Click):** Rejected due to dependency policy and higher behavioral risk; argparse already satisfies needs.

## Implementation Guidance

- **Objectives**: Reduce CLI module size below 500 lines while preserving behavior, CLI arguments, and test expectations; maintain strict typing and docstring/comment policies.
- **Key Tasks**: Create new cohesive modules; move functions; update imports and `__init__.py` re-exports; adjust tests to new module paths or re-export in `cli.py`; ensure plan/spec placeholders are filled if required by process.
- **Dependencies**: No new dependencies; rely on stdlib and existing project modules.
- **Success Criteria**: All CLI functionality behaves identically; module sizes <= 500 lines; Pytest, Ruff, Black, Pyright pass; updated docs/plan as required by repo process.