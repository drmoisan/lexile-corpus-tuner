# copilot-cli-permission-failure (Issue #83)

- Date captured: 2026-01-11
- Author: Dan Moisan
- Status: Superceded by v2

> Automation note: Keep the section headings below unchanged; the promotion tooling maps each of them into the GitHub bug issue template.

- Issue: #83
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/83
- Last Updated: 2026-01-12
## Summary

The atomic executor can invoke Copilot CLI, but Copilot CLI fails to run any shell commands due to an execution-permissions restriction that it cannot prompt the user to approve. This blocks atomic executor tasks that require running QC gates (e.g., Ruff/Pyright/Pytest), and the executor may subsequently terminate Copilot after an idle timeout.

Resolution:

- The atomic executor now invokes Copilot CLI in programmatic mode (`-p/--prompt`) and references the full prompt body via an `@<prompt_file>` mention, rather than feeding the prompt via stdin.
- The atomic executor adds `--add-dir <workspace>` and maintains an explicit tool allowlist, and it fails fast (actionable error) if the known permission-denied message appears.

## Environment

- OS/version: Linux (Dev Container) — Debian GNU/Linux 12 (bookworm)
- Python version: 3.13.9
- Command/flags used:
	- From VS Code task: "Atomic Executor: Execute (prompted)" (updated to `execute-all`)
	- Equivalent command observed in terminal output:
		- `poetry run python -m scripts.dev_tools.atomic_executor.cli execute-all docs/features/active/2026-01-06-populate-open-stax-ck-12-manifest-73/v4/ --workspace /workspaces/lexile-corpus-tuner --preferred-model gpt-5.1-codex-max --max-fix-attempts 10`
- Data source or fixture:
	- Feature folder plan: `docs/features/active/2026-01-06-populate-open-stax-ck-12-manifest-73/v4/`
	- The failing step was a scoped QC gate that attempted to run pytest/ruff.

## Steps to Reproduce

1. In VS Code, run the task **Atomic Executor: Execute (prompted)** (or run the equivalent command above) targeting a feature folder with a multi-task plan.
2. Allow the atomic executor to start Copilot CLI for a task that requires running scoped QC (e.g., Ruff/Pyright/Pytest).
3. Observe Copilot CLI reporting it cannot run commands due to execution restrictions and cannot request permission from the user; the atomic executor then fails the task and may hang-detect/terminate Copilot.

## Expected Behavior

Copilot CLI should be able to run required local commands (e.g., `poetry run pytest ...`, `poetry run ruff check ...`) during task execution so that atomic executor QC gates can run and the plan can proceed.

## Actual Behavior

Copilot CLI reports it cannot run required commands due to execution restrictions and cannot request permission from the user, causing the atomic executor to fail the task. In a subsequent retry, the atomic executor terminates Copilot due to 300s of no stdout output.

Post-fix behavior:

- Copilot can execute Poetry/Python commands during atomic executor runs without requiring interactive approval prompts.
- If Copilot emits `Permission denied and could not request permission from user`, the atomic executor fails fast with actionable context rather than waiting for a 300s idle-timeout.

## Logs / Screenshots

- [ ] Attached minimal logs or screenshot
- Snippet:

	```text
	<frozen runpy>:128: RuntimeWarning: 'scripts.dev_tools.atomic_executor.cli' found in sys.modules after import of package 'scripts.dev_tools.atomic_executor', but prior to execution of 'scripts.dev_tools.atomic_executor.cli'; this may result in unpredictable behaviour
	Executing task P1-T5 (attempt 1/10)
	Unable to run the required command due to execution restrictions in the environment. Attempts to run `poetry run pytest --cov=... tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py` (and even `poetry --version`/`python3 --version`) all returned “Permission denied and could not request permission from user,” so I couldn’t execute the coverage gate. Please grant permission to execute these commands (or run them manually) and let me know if I should retry.
  
	...
	Scoped QC failed for task P1-T5: Command '['poetry', 'run', 'ruff', 'check', 'src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_catalog.py', 'tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py']' returned non-zero exit status 1.
	Executing task P1-T5 (attempt 2/10)
	...
	TimeoutError: Copilot CLI produced no output for 300.0 seconds while executing task P1-T5; terminated to avoid hanging.
	```

	Notes:
	- The idle timeout default is 300s and is controlled by `ATOMIC_EXECUTOR_COPILOT_IDLE_TIMEOUT_SECONDS` (see `scripts/dev_tools/atomic_executor/cli.py`).
	- The Ruff failure shown above was a normal lint issue (`E501`) that was fixable; however, the *permission denied* issue is separate and appears to be the primary blocker for Copilot-driven QC.

Validation excerpt (post-fix):

	```text
	=== Copilot invocation ===
	task_id: P1-T5
	preferred_model: gpt-5.1-codex-max
	prompt_file: /workspaces/lexile-corpus-tuner/.agent_logs/prompts/prompt_2026-01-11_230622_P1-T5.md
	(prompt omitted from log for brevity; use --print-prompt to view)
	Ran `python -m poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py` successfully (10 passed). No code changes were needed.
	```

	Source: `.agent_logs/atomic_executor_2026-01-11_230622.log`

## Impact / Severity

- [ ] Blocker
- [ ] High
- [ ] Medium
- [ ] Low

This is a **blocker** for using the atomic executor autonomously, because it cannot run its required QC gates when Copilot CLI is unable to execute commands.

Status update:

- This blocker is resolved by changing the atomic executor’s Copilot CLI invocation contract (see “Resolution” in Summary and “Proposed Fix / Validation Ideas” below).

## Suspected Cause / Notes

- This appears to be a Copilot CLI / agent execution permission issue (possibly VS Code settings, enterprise policy, or a “require user approval for command execution” mode) where:
	- Copilot CLI attempts to execute `poetry`/`python3` commands,
	- the environment denies execution and disallows prompting for permission,
	- so Copilot cannot proceed with required QC gates.

- The atomic executor then retries and may hit hang detection because Copilot produces no output while waiting.

- Evidence that the underlying shell is not broken: running the same commands manually in the terminal succeeds (e.g., `poetry run pytest ...` completed successfully in the dev container).

- Related file(s):
	- `scripts/dev_tools/atomic_executor/cli.py`
		- `_resolve_idle_timeout_seconds` (default 300s)
		- env override: `ATOMIC_EXECUTOR_COPILOT_IDLE_TIMEOUT_SECONDS`

Confirmed root cause (post-fix):

- The atomic executor invoked Copilot CLI without programmatic mode (`-p/--prompt`) and fed the prompt via non-interactive stdin. This prevented Copilot from requesting/receiving approvals in headless execution.

Implemented mitigation:

- Always invoke Copilot CLI using `-p/--prompt` and reference a prompt file via `@<path>` (to avoid Windows argv-length issues).
- Add `--add-dir <workspace>` and keep an explicit tool allowlist.
- Fail fast with actionable context on the known permission-denied output string.

## Proposed Fix / Validation Ideas

- [x] Unit coverage areas
- [x] Integration scenario to retest
- [x] Manual verification notes

Ideas (to research/validate after issue creation):

- Determine where Copilot CLI command execution permissions are configured and document the required setting(s).
- Add a preflight check in atomic executor to detect command-execution restrictions early (e.g., attempt a harmless command through the same pathway Copilot uses), and fail with actionable guidance rather than timing out.
- Consider allowing the idle timeout to be configured via CLI flag (if not already) and/or emitting the current resolved timeout value in logs.

Manual validation:

- Re-ran a representative `execute-all` workflow and observed Copilot successfully executing a Poetry-backed command during a Copilot session.

Commands used during validation:

- End-to-end repro:
	- `poetry run python -m scripts.dev_tools.atomic_executor.cli execute-all docs/features/active/2026-01-06-populate-open-stax-ck-12-manifest-73/v4/ --workspace /workspaces/lexile-corpus-tuner --preferred-model gpt-5.1-codex-max --max-fix-attempts 10`
- Targeted regression tests:
	- `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_cli.py::TestRunCopilot::test_run_copilot_invokes_with_correct_arguments -q`
	- `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_cli.py::TestRunCopilot::test_run_copilot_permission_denied_fails_fast_with_actionable_error -q`

## Next Step

- [x] Promote to GitHub issue (bug-report template)
- [x] Move to active fix folder / branch

Follow-up research needed:

- Identify the exact Copilot CLI / VS Code setting or policy that governs “permission to execute commands,” and how to enable it (or how to detect when it’s blocked).

Optional follow-up:

- If additional Copilot execution permission behaviors appear (unrelated to the original “permission denied” string), consider a separate issue to improve idle-timeout ergonomics (e.g., better progress heartbeat, more actionable timeout errors).