# 2026-01-11-copilot-cli-permission-failure (Spec)

- Issue: #83
- Owner: drmoisan
- Date: 2026-01-11
- Status: Draft

## Context
The atomic executor can invoke Copilot CLI, but Copilot CLI fails to run any shell commands due to an execution-permissions restriction that it cannot prompt the user to approve. This blocks atomic executor tasks that require running QC gates (e.g., Ruff/Pyright/Pytest), and the executor may subsequently terminate Copilot after an idle timeout.

Environment:
- OS/version: Linux (Dev Container) — Debian GNU/Linux 12 (bookworm)
- Python version: 3.13.9
- Command/flags used:
	- From VS Code task: "Atomic Executor: Execute (prompted)" (updated to `execute-all`)
	- Equivalent command observed in terminal output:
		- `poetry run python -m scripts.dev_tools.atomic_executor.cli execute-all docs/features/active/2026-01-06-populate-open-stax-ck-12-manifest-73/v4/ --workspace /workspaces/lexile-corpus-tuner --preferred-model gpt-5.1-codex-max --max-fix-attempts 10`
- Data source or fixture:
	- Feature folder plan: `docs/features/active/2026-01-06-populate-open-stax-ck-12-manifest-73/v4/`
	- The failing step was a scoped QC gate that attempted to run pytest/ruff.

Impact / Severity:
- [ ] Blocker
- [ ] High
- [ ] Medium
- [ ] Low

This is a **blocker** for using the atomic executor autonomously, because it cannot run its required QC gates when Copilot CLI is unable to execute commands.


## Repro & Evidence
Steps to Reproduce:
1. In VS Code, run the task **Atomic Executor: Execute (prompted)** (or run the equivalent command above) targeting a feature folder with a multi-task plan.
2. Allow the atomic executor to start Copilot CLI for a task that requires running scoped QC (e.g., Ruff/Pyright/Pytest).
3. Observe Copilot CLI reporting it cannot run commands due to execution restrictions and cannot request permission from the user; the atomic executor then fails the task and may hang-detect/terminate Copilot.

Expected:
Copilot CLI should be able to run required local commands (e.g., `poetry run pytest ...`, `poetry run ruff check ...`) during task execution so that atomic executor QC gates can run and the plan can proceed.

Actual:
Copilot CLI reports it cannot run required commands due to execution restrictions and cannot request permission from the user, causing the atomic executor to fail the task. In a subsequent retry, the atomic executor terminates Copilot due to 300s of no stdout output.

Logs / Screenshots:
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


## Scope & Non-Goals
- In scope:
- Out of scope / non-goals:

## Root Cause Analysis
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


## Proposed Fix
- [ ] Unit coverage areas
- [ ] Integration scenario to retest
- [ ] Manual verification notes

Ideas (to research/validate after issue creation):

- Determine where Copilot CLI command execution permissions are configured and document the required setting(s).
- Add a preflight check in atomic executor to detect command-execution restrictions early (e.g., attempt a harmless command through the same pathway Copilot uses), and fail with actionable guidance rather than timing out.
- Consider allowing the idle timeout to be configured via CLI flag (if not already) and/or emitting the current resolved timeout value in logs.

Manual validation:

- Once permissions are resolved, re-run the same `execute-all` command against the feature folder and confirm tasks can run scoped QC without permission errors/timeouts.


## Assumptions, Constraints, Dependencies
- Assumptions (environment, data, access):
- Constraints (budget, performance, compatibility):
- External dependencies (services, libraries, releases):

## Data / API / Config Impact
- User-facing or API changes:
- Data or migration considerations:
- Logging/telemetry updates (if any):

## Test Strategy
- [ ] Unit coverage areas
- [ ] Integration scenario to retest
- [ ] Manual verification notes

Ideas (to research/validate after issue creation):

- Determine where Copilot CLI command execution permissions are configured and document the required setting(s).
- Add a preflight check in atomic executor to detect command-execution restrictions early (e.g., attempt a harmless command through the same pathway Copilot uses), and fail with actionable guidance rather than timing out.
- Consider allowing the idle timeout to be configured via CLI flag (if not already) and/or emitting the current resolved timeout value in logs.

Manual validation:

- Once permissions are resolved, re-run the same `execute-all` command against the feature folder and confirm tasks can run scoped QC without permission errors/timeouts.


## Acceptance Criteria
- Conditions that must be true for the bug to be considered fixed (map to repro and edge cases).

## Risks & Mitigations
- Technical or operational risks:
- Mitigations and rollbacks:

## Rollout & Follow-up
- Release/rollout steps:
- Post-fix monitoring or clean-up tasks:
- Links: issue, PRs, related docs
