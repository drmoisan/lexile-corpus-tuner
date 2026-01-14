# PR Notes — Issue #83 (Copilot CLI permission failure)

## Summary

This change updates the atomic executor’s Copilot CLI invocation to be headless-safe and deterministic:

- Always use Copilot programmatic mode (`-p/--prompt`).
- Keep the full prompt in an on-disk file and reference it via `@<prompt_file>` from the `-p` value (avoids Windows argv-length failures).
- Stop feeding the prompt via stdin.
- Add `--add-dir <workspace>` and extend the explicit tool allowlist for QC.
- Fail fast with actionable context when Copilot output contains the known headless permission failure string.

## Changes

- `scripts/dev_tools/atomic_executor/cli.py`
  - `run_copilot()` now uses `-p "Follow these instructions exactly: @<prompt_file>"`.
  - Prompt content is written to `.agent_logs/prompts/prompt_<run_id>_<task_id>.md` and is not provided via stdin.
  - Copilot argv includes:
    - `--add-dir <workspace>`
    - `--allow-tool write`
    - `--allow-tool shell(poetry)`
    - `--allow-tool shell(python)`
    - `--allow-tool shell(python3)`
    - `--allow-tool shell(git)`
  - Streaming output detection fails fast when output contains:
    - `Permission denied and could not request permission from user`

- `tests/scripts/dev_tools/atomic_executor/test_cli.py`
  - Updated regression coverage to enforce the new Copilot invocation contract.
  - Added regression coverage for the fail-fast permission-denied error path.

## Verification (deterministic checklist)

### Targeted regression tests

- `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_cli.py::TestRunCopilot::test_run_copilot_invokes_with_correct_arguments -q`
- `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_cli.py::TestRunCopilot::test_run_copilot_permission_denied_fails_fast_with_actionable_error -q`

### End-to-end repro

- `poetry run python -m scripts.dev_tools.atomic_executor.cli execute-all docs/features/active/2026-01-06-populate-open-stax-ck-12-manifest-73/v4/ --workspace /workspaces/lexile-corpus-tuner --preferred-model gpt-5.1-codex-max --max-fix-attempts 10`

Validation evidence excerpt (post-fix) is recorded in:

- `.agent_logs/atomic_executor_2026-01-11_230622.log`

### Full toolchain loop (repo policy)

Final clean pass (format → lint → type-check → tests):

- `poetry run black .`
- `poetry run ruff check`
- `poetry run pyright`
- `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`

## Risk / Security Statement

This change broadens Copilot’s allowed capabilities (explicit tool allowlist plus workspace directory allowlisting) to enable headless QC gates.

Mitigations:

- Scope remains least-privilege: only `write`, `shell(poetry)`, `shell(python)`, `shell(python3)`, `shell(git)` are allowed.
- The allowed directory is limited to the workspace via `--add-dir <workspace>`.
- The implementation fails fast with actionable context if Copilot hits the known permission-denied condition, rather than hanging until idle-timeout.
