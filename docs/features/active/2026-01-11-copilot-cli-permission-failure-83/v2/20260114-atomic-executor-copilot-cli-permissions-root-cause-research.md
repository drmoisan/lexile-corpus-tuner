# Task Research Notes: Atomic executor Copilot CLI still shows “interactive-mode” behaviors after #83

## Research Executed

### File Analysis

- `/workspaces/lexile-corpus-tuner/scripts/dev_tools/atomic_executor/cli.py`
  - Verified the atomic executor is invoking Copilot CLI in **programmatic mode** using `-p` and a prompt-file `@/abs/path` reference.
  - Verified the executor passes `--model <normalized_model>` when a preferred model is supplied.
  - Verified the executor passes `--add-dir <workspace>` and tool approvals: `--allow-tool write`, `--allow-tool shell(poetry)`, `--allow-tool shell(python)`, `--allow-tool shell(python3)`, `--allow-tool shell(git)`.
  - Verified the executor fail-fast detection is **stdout-stream substring scanning** for:
    - `COPILOT_PERMISSION_DENIED_SUBSTRING = "Permission denied and could not request permission from user"`

- `/workspaces/lexile-corpus-tuner/.agent_logs/atomic_executor_2026-01-14_180844.log`
  - Confirms `P2-T3` was marked gated after Copilot reported running QC via `python -m poetry run ...` (exit 0).
  - Confirms `P2-T4` aborted after Copilot output contained the exact substring:
    - `"Permission denied and could not request permission from user"` (log line is truncated mid-quote, but the prefix matches the executor’s detector).
  - Confirms `share_path` for `P2-T4` was intended, but the transcript file was not produced (consistent with the executor killing the process immediately upon detecting the substring).

- `/workspaces/lexile-corpus-tuner/.agent_logs/copilot_sessions/copilot_session_2026-01-14_180844_P2-T3.md`
  - Confirms Copilot CLI was started with a single “User” message:
    - `Follow these instructions exactly: @...prompt_..._P2-T3.md`
    - This is consistent with **programmatic mode** (`-p`) rather than interactive mode.
  - Confirms multiple `bash` tool invocations of `poetry run ...` failed with:
    - `Permission denied and could not request permission from user`
  - Confirms the same session later succeeds when switching to:
    - `python -m poetry run black/ruff/pyright/pytest` (exit 0)

- `/workspaces/lexile-corpus-tuner/.agent_logs/prompts/prompt_2026-01-14_180844_P2-T3.md`
- `/workspaces/lexile-corpus-tuner/.agent_logs/prompts/prompt_2026-01-14_180844_P2-T4.md`
  - Confirms both prompts contain template language claiming an “interactive session” and instructing use of `/model`.
  - Confirms both prompts instruct QC via `poetry run ...`, which is transcript-proven to be the failing command form in this environment.

- `/workspaces/lexile-corpus-tuner/scripts/dev_tools/atomic_executor/prompt_builder.py`
  - Confirms prompts include a “Model Selection” section that is **incompatible with programmatic mode** (it tells the agent to type `/model`).
  - Confirms the prompt builder hardcodes QC instructions to `poetry run ...`.
  - Confirms template placeholder substitution sets `<feature>` to `feature_dir.name` (e.g., `v4`), producing incorrect “Authoritative Documents” paths like `docs/features/active/v4/plan.md`.

- `/workspaces/lexile-corpus-tuner/.venv/bin/poetry`
  - Confirmed the Poetry script entrypoint uses an absolute-path shebang:
    - `#!/workspaces/lexile-corpus-tuner/.venv/bin/python`
  - Confirmed from the session transcript that `.venv/bin/python` is a symlink to `/usr/local/bin/python3.13`.

### Code Search Results

- `COPILOT_PERMISSION_DENIED_SUBSTRING`
  - Found in `scripts/dev_tools/atomic_executor/cli.py`.

- `poetry run black --check`
  - Found as hardcoded text in `scripts/dev_tools/atomic_executor/prompt_builder.py`.

- `Enter "/model" in the Copilot CLI interactive session`
  - Found in `scripts/dev_tools/atomic_executor/prompt_builder.py` and in per-task prompt files.

### External Research

- #fetch:https://docs.github.com/en/copilot/concepts/agents/about-copilot-cli
  - Confirms “Modes of use”:
    - Interactive mode: `copilot`
    - Programmatic mode: `copilot -p/--prompt "..."`
  - Confirms tool approval flags exist for both modes:
    - `--allow-tool 'shell(COMMAND)'`, `--allow-tool 'shell'`, `--allow-tool 'write'`, `--allow-all-tools`, `--deny-tool ...`

- #fetch:https://docs.github.com/en/copilot/how-tos/use-copilot-agents/use-copilot-cli
  - Confirms Copilot CLI permission layers beyond tool approvals:
    - **Path permissions** (default: CWD + subdirs + temp)
    - **URL permissions** (default: require approval)
  - Confirms path permissions behavior that matches transcript evidence:
    - For shell commands, paths are extracted heuristically from command text.
    - **Symlinks are resolved for existing files**.
    - `--allow-all-paths` disables path verification.

### Project Conventions

- Standards referenced: `.github/instructions/general-code-change.instructions.md`, `.github/instructions/python-code-change.instructions.md`, `.github/instructions/general-unit-test.instructions.md`, `.github/instructions/python-unit-test.instructions.md`
- Instructions followed: Task Researcher mode (write-only in `artifacts/research/`)

## Key Discoveries

### Project Structure

- Copilot CLI invocation artifacts:
  - Executor logs: `.agent_logs/atomic_executor_<run_id>.log`
  - Per-task prompts: `.agent_logs/prompts/prompt_<run_id>_<task_id>.md`
  - Copilot exports: `.agent_logs/copilot_sessions/copilot_session_<run_id>_<task_id>.md`

### Implementation Patterns

- **#83 fix is present and active**
  - The executor invocation in `cli.py::run_copilot()` includes `-p` and `@prompt_file`.
  - The P2-T3 session transcript begins with a single prompt “User” message and does not show an interactive prompt loop.

- **The “interactive mode” impression is coming from prompt text, not the executor**
  - PromptBuilder injects a “Model Selection” block that explicitly says “interactive session” and instructs `/model`.
  - This instruction is incompatible with headless programmatic mode and is now misleading because the executor already passes `--model`.

### Complete Examples

```text
Evidence: atomic executor uses programmatic mode (-p) and prompt file @mention

File: /workspaces/lexile-corpus-tuner/scripts/dev_tools/atomic_executor/cli.py

argv includes:
  - -p
  - Follow these instructions exactly: @/workspaces/.../.agent_logs/prompts/prompt_<run_id>_<task_id>.md
  - --allow-tool shell(poetry)
  - --allow-tool shell(python)
  - --add-dir /workspaces/lexile-corpus-tuner
```

```text
Evidence: Poetry entrypoint is a shebang script referencing an absolute path

File: /workspaces/lexile-corpus-tuner/.venv/bin/poetry

#!/workspaces/lexile-corpus-tuner/.venv/bin/python
...
```

```text
Evidence: Copilot CLI path permissions resolve symlinks for existing files

Source: GitHub Docs “Using GitHub Copilot CLI”
URL: https://docs.github.com/en/copilot/how-tos/use-copilot-agents/use-copilot-cli

Path detection notes include:
  - "Symlinks are resolved for existing files"
  - "To disable path verification, use --allow-all-paths"
```

### API and Schema Documentation

- Copilot CLI approval options (docs-verified):
  - Programmatic mode: `copilot -p/--prompt "..."`
  - Tool approvals:
    - `--allow-tool 'shell(COMMAND)'`
    - `--allow-tool 'shell'`
    - `--allow-tool 'write'`
  - Path approval override:
    - `--allow-all-paths`

### Configuration Examples

```text
Observed prompt template output (per-task prompt files):

- "Before executing this task, please use the /model command in GitHub Copilot CLI"
- QC toolchain explicitly uses: poetry run black/ruff/pyright/pytest
```

### Technical Requirements

- The executor must run Copilot CLI headlessly.
- Copilot CLI must be able to execute QC-related commands reliably.
- Prompts must not instruct interactive-only slash commands when running under `-p`.

## Recommended Approach

Treat this as **two coupled issues** that together create the apparent regression:

1) **Prompt content is stale and implies interactive mode**, which is confusing and encourages behaviors (`/model`, `poetry run ...`) that are fragile in programmatic mode.

2) **Copilot CLI path permissions + symlink resolution** are blocking execution of the Poetry script entrypoint.

Root cause of `poetry ...` denial (evidence-based):

- `poetry` is a script with an absolute-path shebang to `.venv/bin/python`.
- `.venv/bin/python` is a symlink that resolves to `/usr/local/bin/python3.13`.
- GitHub Docs state Copilot CLI path permissions resolve symlinks for existing files.
- The transcript shows access to `/usr/local/bin` is blocked (listing/executing denied) while `python -c ...` works.
- Therefore, `poetry ...` likely fails because its interpreter chain requires absolute-path execution outside trusted paths, which cannot be approved in `-p` mode.

Recommended fix direction (least privilege, evidence-backed):

- Stop relying on executing the Poetry script entrypoint (`poetry ...`) inside Copilot CLI.
- Prefer `python -m poetry ...` (transcript-proven to succeed) for all toolchain steps executed within Copilot sessions.
- Remove interactive-only `/model` instructions from prompts when the executor is already passing `--model`.
- Fix template placeholder substitution for `<feature>` so “Authoritative Documents” paths resolve correctly.

This keeps the executor in programmatic mode (as #83 intended) and avoids broadening Copilot’s path permissions (e.g., `--allow-all-paths`) unless it becomes unavoidable.

## Implementation Guidance

- **Objectives**:
  - Ensure Copilot tasks do not attempt interactive-only slash commands under programmatic runs.
  - Ensure QC commands succeed reliably under Copilot CLI’s permissions model.
  - Ensure prompt “Authoritative Documents” paths are correct to reduce early tool failures.

- **Key Tasks**:
  - Update prompt generation so:
    - QC commands use `python -m poetry run ...` instead of `poetry run ...`.
    - The model-selection section is removed (or reworded) when `preferred_model` is supplied, because the executor already passes `--model`.
    - `<feature>` substitution uses the correct relative path under `docs/features/active/...` (not just the leaf folder name `v4`).
  - Consider adding a short “If you see permission denied…” note in the prompt instructing the `python -m poetry` fallback.

- **Dependencies**:
  - None new. Uses existing Copilot CLI flags and existing Python/Poetry installation.

- **Success Criteria**:
  - In programmatic mode (`-p`), Copilot can run the task-step toolchain via `python -m poetry run ...` without triggering permission-denied.
  - Prompts no longer instruct `/model` or claim an “interactive session” when run under `-p`.
  - Copilot no longer attempts to open non-existent `docs/features/active/v4/plan.md` due to incorrect placeholder substitution.
