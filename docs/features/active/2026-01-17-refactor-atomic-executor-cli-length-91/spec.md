# 2026-01-17-refactor-atomic-executor-cli-length - Refactor Spec

- Issue: #91
- Parent Initiative (optional): #<parent-id>
- Owner: drmoisan
- Last Updated: 2026-01-17

## Intent & Outcomes

Reduce the size of the atomic executor CLI module by splitting it into cohesive, testable modules while preserving all existing behavior and public surfaces. The end-state is a small, readable `cli.py` that wires dependencies together, with domain-specific helpers in dedicated modules and each module staying within the 500-line policy limit.

## Invariants (must not change)

- CLI subcommands and flags remain identical (`execute`, `resume`, `execute-all` and all existing options).
- Exit codes and user-visible stdout/stderr messages used by tests remain stable.
- Lock-file behavior for `execute-all` remains unchanged (path `.agent_logs/executor.lock`, create-once, release in `finally`).
- Copilot invocation semantics remain unchanged (programmatic prompt file, session reuse rules, allow-tool list, log/session paths).
- Scoped/full QC behavior, order, and toolchain commands remain unchanged.
- Plan parsing, checkbox flipping, and plan discovery behavior remain unchanged.
- No re-export shims: all call sites and imports are updated to the refactored module paths.

## Scope (structural changes)

- Split `scripts/dev_tools/atomic_executor/cli.py` into cohesive modules by responsibility.
- Keep `cli.py` as the CLI entry point that delegates to new modules.
- Update all imports/patch targets to point at new module locations (no re-exports).
- Refactor unit tests into dedicated files so each production module has a single, matching test module.
- Update console script entry points and any documentation references if module paths change.

Target layout (proposed):
- `cli.py`: `main`, `resolve_workspace`, and imports/wiring for other helpers.
- `cli_args.py`: `parse_args` and shared subcommand argument wiring.
- `workspace_checks.py`: `ensure_clean_tree`, `refuse_protected_branch`, `_current_branch`.
- `executor_lock.py`: `acquire_executor_lock`, `release_executor_lock`, lock constants.
- `clipboard.py`: `get_clipboard_command`, `copy_to_clipboard`.
- `copilot_executor.py`: `run_copilot`, `_stream_copilot_output`, `_resolve_idle_timeout_seconds`, `_copilot_supports_session`, `_clean_session_file`, `CopilotPermissionDeniedError`.
- `task_executor.py`: `execute_one_task` and task-level orchestration helpers.

Target test layout (proposed):
- `tests/scripts/dev_tools/atomic_executor/test_cli_args.py` → `cli_args.py`
- `tests/scripts/dev_tools/atomic_executor/test_workspace_checks.py` → `workspace_checks.py`
- `tests/scripts/dev_tools/atomic_executor/test_executor_lock.py` → `executor_lock.py`
- `tests/scripts/dev_tools/atomic_executor/test_clipboard.py` → `clipboard.py`
- `tests/scripts/dev_tools/atomic_executor/test_copilot_executor.py` → `copilot_executor.py`
- `tests/scripts/dev_tools/atomic_executor/test_task_executor.py` → `task_executor.py`
- `tests/scripts/dev_tools/atomic_executor/test_cli.py` → `cli.py`

Cleanup/removals:
- Remove duplicate inline helpers from `cli.py` once relocated.
- Consolidate constants into the owning modules where they are used.

## Non-Goals

- No new CLI flags, no changes to existing CLI defaults or semantics.
- No behavior changes to Copilot throttling, logging, or QC gating.
- No new dependencies or framework swaps (remain on `argparse`).
- No functional refactors beyond structural separation.
- No changes to integration-test coverage other than relocating tests to match production modules.

## Dependencies / Touchpoints

- `scripts/dev_tools/atomic_executor/__init__.py` exports (`main`, `PlanParser`, `PromptBuilder`, `QCRunner`, `FeatureResolver`) and must reflect new module paths.
- `pyproject.toml` console scripts: `atomic-executor` and `dev.atomic-executor` point to `scripts.dev_tools.atomic_executor.cli:main`.
- Tests: `tests/scripts/dev_tools/atomic_executor/test_cli.py` imports functions from `scripts.dev_tools.atomic_executor.cli`.
- Integration tests: `tests/scripts/dev_tools/test_atomic_executor_cli.py` patches `scripts.dev_tools.atomic_executor.cli` symbols and calls `main` directly.
- All unit tests must map 1:1 to production modules (one test file per production file; one production file per test file), except integration tests.
- Downstream tooling: CLI entry points invoked via `python -m scripts.dev_tools.atomic_executor.cli` or package main import.
- Log/artifact paths: `.agent_logs/atomic_executor_*.log`, `.agent_logs/copilot_sessions`, `.agent_logs/prompts`.
- Documentation references in `docs/developer-tooling.md` and feature specs/plans that mention `python -m scripts.dev_tools.atomic_executor.cli` must be updated if the module path changes.

## Risks & Mitigations

- Risk: Breaking tests and external imports that reference `scripts.dev_tools.atomic_executor.cli`.
	- Mitigation: Update all imports/patch targets and entry points; do not add re-export shims.
- Risk: Violating the 1:1 unit test mapping requirement after splitting production modules.
	- Mitigation: Create dedicated test modules for each production module and move relevant tests accordingly; keep any multi-module coverage explicitly labeled as integration tests.
- Risk: Subtle behavior drift from moving logic (stdout/stderr ordering, retries, session flags).
	- Mitigation: Keep function bodies unchanged during the move; add targeted regression tests where new modules are introduced.
- Risk: Circular imports after splitting modules.
	- Mitigation: Define dependency direction explicitly (e.g., `cli.py` depends on helper modules; helpers do not import `cli.py`).
- Risk: New modules exceed 500 lines.
	- Mitigation: Split by responsibility and keep helpers localized to each module.

## Definition of Done

- [ ] Structure matches this spec; legacy paths retired or redirected
- [ ] Behavior unchanged (validated against invariants)
- [ ] Imports/tooling/entry points updated
- [ ] Tests and type checks clean
- [ ] Docs updated (initiative/README/tasks as needed)
- [ ] Unit tests follow 1:1 production-to-test mapping (excluding integration tests)

Evidence expectations for completion:
- `cli.py` and all new modules are <= 500 lines.
- Existing CLI tests pass with no behavioral changes.
- Each production module has exactly one unit test file dedicated to it, and each unit test file covers only its paired production module (integration tests explicitly separated).
- All imports/patch targets/entry points reference the new module paths (no re-exports or shims).
- Documentation updated where module paths or entry points are referenced.
