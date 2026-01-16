# 2026-01-11-copilot-cli-permission-failure (Spec)

- Issue: #83
- Owner: drmoisan
- Date: 2026-01-11
- **Status:** Completed
- **Outcome:** Prompt builder now emits `python -m poetry run` forms for QC commands, removes interactive-only `/model` instructions, and correctly substitutes `<feature>` placeholder with relative path from docs/features/active/. End-to-end repro confirms no permission-denied errors.
- **Root Cause:** Copilot CLI path permission enforcement (including symlink resolution) blocks execution of the Poetry script entrypoint whose shebang resolves outside allowed paths; prompt generation also contained stale "interactive session" instructions (/model) and used incorrect <feature> placeholder substitution, which increased failure likelihood.
- Last Updated: 2026-01-15

## Post-Fix Evidence

### Prompt contains correct `python -m poetry run` forms
```
- python -m poetry run black .
- python -m poetry run ruff check
- python -m poetry run pyright
- python -m poetry run pytest \
    --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools \
    --cov-report=term-missing
```

### Prompt excludes `/model` and `interactive session`
Verified: `grep -i "/model"` returns 0 matches; `grep -i "interactive session"` returns 0 matches.

### End-to-end repro shows 0 permission-denied errors
Log: `atomic_executor_2026-01-15_221048.log`
Result: `grep -c "Permission denied and could not request permission from user"` returns 0.

---

## Context
The atomic executor can invoke Copilot CLI, but Copilot CLI fails to run shell commands during the agent session with the message: “Permission denied and could not request permission from user”. This blocks atomic executor tasks that require running QC gates (e.g., Ruff/Pyright/Pytest), and the executor may subsequently terminate Copilot after an idle timeout.

This spec originally focused on a programmatic-mode mismatch; however, follow-up investigation shows the executor **is** using Copilot CLI programmatic mode (`-p`) and the remaining “permission denied” failures are best explained by Copilot CLI’s **path permission** system interacting with Poetry’s script entrypoint and symlink resolution.

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
- [x] Blocker
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
Copilot CLI should be able to run required local commands during task execution so that atomic executor QC gates can run and the plan can proceed. In this devcontainer environment, the reliable command form is `python -m poetry run ...`.

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

Post-fix evidence:

- The atomic executor now invokes Copilot CLI via programmatic mode (`-p/--prompt`) and references the prompt file via an `@<path>` mention. Prompt content is no longer passed via stdin.
- A representative atomic executor run shows Copilot successfully executing a Poetry-backed command during a Copilot session (no interactive approvals required):

	```text
	=== Copilot invocation ===
	task_id: P1-T5
	preferred_model: gpt-5.1-codex-max
	prompt_file: /workspaces/lexile-corpus-tuner/.agent_logs/prompts/prompt_2026-01-11_230622_P1-T5.md
	(prompt omitted from log for brevity; use --print-prompt to view)
	Ran `python -m poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py` successfully (10 passed). No code changes were needed.
	```

	Source: `.agent_logs/atomic_executor_2026-01-11_230622.log`


## Scope & Non-Goals
- In scope:
	- Update the atomic executor’s Copilot CLI invocation to use **programmatic mode** (`-p/--prompt`) on all platforms.
	- Preserve the Windows-safe “large prompt” behavior without OS-conditional branching by using a prompt file and referencing it from the `-p` prompt with an `@path` mention.
	- Ensure Copilot has the necessary session-level permissions to execute the required QC shell commands and make file edits during atomic tasks.
	- Update/replace the existing unit tests that currently enforce “no `-p`”.
	- Document and validate the new behavior via a manual repro run of `execute-all` against a representative feature folder plan.
- Out of scope / non-goals:
	- Changing enterprise policy, VS Code org settings, or global Copilot policy outside of what the atomic executor controls.
	- Adding new third-party dependencies.
	- Refactoring the atomic executor’s broader architecture, scheduling, or retry/backoff logic beyond what’s needed to fix the Copilot CLI invocation.

## Root Cause Analysis
Observed behavior:

- Copilot CLI is started by the atomic executor in a non-interactive context.
- Within the Copilot session, attempts to run shell commands (even `poetry --version` / `python3 --version`) fail with:
	- `Permission denied and could not request permission from user`

Why this happens:

- The atomic executor does invoke Copilot CLI in **programmatic mode** (`-p`) and supplies the prompt via an on-disk file referenced with an `@/abs/path` mention.

- The remaining “Permission denied and could not request permission from user” failures are consistent with Copilot CLI’s **path permission** enforcement:
	- Copilot CLI heuristically extracts paths from shell commands and enforces an allowlist.
	- GitHub Docs state that **symlinks are resolved for existing files**.
	- In this repo’s devcontainer, `poetry` is a script entrypoint with an absolute-path shebang to `.venv/bin/python`, and `.venv/bin/python` resolves (via symlink) to `/usr/local/bin/python3.13`.
	- The session transcript shows `/usr/local/bin` is blocked, so running `poetry ...` often fails, while `python -m poetry ...` succeeds.

- Prompt generation contributes to the observed failures:
	- The prompt includes stale instructions that imply an “interactive session” and tells the agent to use `/model`, which is not applicable in programmatic mode.
	- Template placeholder substitution currently sets `<feature>` to the leaf folder name (e.g., `v4`), producing incorrect “Authoritative Documents” paths such as `docs/features/active/v4/plan.md`.


## Proposed Fix
Implemented fix (core behavior change):

- Change atomic executor Copilot invocation to always run in **programmatic mode**:
	- Use `copilot -p <short_prompt>` instead of `copilot` with prompt fed via stdin.

- Preserve Windows safety and minimize OS branching:
	- Continue generating the full prompt body into a prompt file.
	- Construct a short `-p` prompt that references the file via `@<path>`.
		- Exact prefix used: `Follow these instructions exactly: `.
		- Example: `Follow these instructions exactly: @/abs/path/to/prompt.txt`.

Concrete invocation contract (as implemented in `scripts/dev_tools/atomic_executor/cli.py::run_copilot()`):

- Prompt delivery:
	- Writes the full prompt body to a prompt file under `.agent_logs/prompts/`.
	- Invokes Copilot with `-p` whose value includes an `@{prompt_file}` reference.
	- Does not pass prompt content via stdin.
- Workspace path allowlist:
	- Adds `--add-dir <workspace>` to reduce headless path trust prompts.
- Tool allowlist:
	- Includes `--allow-tool write`, `--allow-tool shell(poetry)`, `--allow-tool shell(python)`, `--allow-tool shell(python3)`, `--allow-tool shell(git)`.

Additional remediation required (based on post-implementation research):

- Update prompt generation to instruct QC commands using the reliable form:
	- Prefer `python -m poetry run ...` over `poetry run ...`.

- Remove or reword interactive-only instructions in prompts:
	- Do not instruct `/model` when the executor already passes `--model`.

- Fix prompt template substitution so “Authoritative Documents” paths refer to the actual feature folder under `docs/features/active/<feature>/...` rather than the leaf folder name.

- Consider (only if needed) enabling broader path permissions:
	- Use `--allow-all-paths` only if prompt fixes are insufficient, because it increases capability surface.

Permissions/approvals (session-level):

- Continue to pass explicit tool permissions required for atomic tasks, at minimum:
	- `--allow-tool write`
	- `--allow-tool shell(poetry)`
	- `--allow-tool shell(python3)` (or `python`, depending on what the repo uses)
	- `--allow-tool shell(git)`

- Ensure Copilot can read prompt / workspace files without interactive path trust prompts:
	- Prefer explicitly allowing the workspace paths for the session (for example, via Copilot CLI path permission flags).
	- If the Copilot session still prompts for path approvals, add `--allow-all-paths` as a targeted workaround for headless operation.

Guardrails and diagnostics:

- If Copilot returns the specific permission error text again, surface an actionable error pointing to:
	- the exact Copilot CLI argv used,
	- which permissions were granted,
	- and recommended remediations (for example, adding `--add-dir`, expanding `--allow-tool`, or running the command manually if policy blocks headless execution).
	- This detection is implemented as a streaming output substring check for:
		- `Permission denied and could not request permission from user`.

Test updates:

- Replace unit test assertions that enforce “no `-p`”. Add assertions that:
	- `-p/--prompt` is present in the Copilot argv.
	- The `-p` prompt contains an `@<prompt_file>` reference.
	- Prompt content is not provided on stdin (stdin should be left as default unless needed for other reasons).
	- Tool permission flags are still present (write + required shell commands).

Manual validation:

- Re-run the original `execute-all` repro command against a representative multi-task plan.
- Confirm Copilot can execute at least:
	- `poetry --version`
	- `python3 --version`
	- and a scoped QC command (e.g., `poetry run ruff check ...`)
	without producing “Permission denied and could not request permission from user”.

Validation command (used during development):

- End-to-end repro:
	- `poetry run python -m scripts.dev_tools.atomic_executor.cli execute-all docs/features/active/2026-01-06-populate-open-stax-ck-12-manifest-73/v4/ --workspace /workspaces/lexile-corpus-tuner --preferred-model gpt-5.1-codex-max --max-fix-attempts 10`
- Targeted regression tests:
	- `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_cli.py::TestRunCopilot::test_run_copilot_invokes_with_correct_arguments -q`
	- `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_cli.py::TestRunCopilot::test_run_copilot_permission_denied_fails_fast_with_actionable_error -q`


## Assumptions, Constraints, Dependencies
- Assumptions (environment, data, access):
	- Copilot CLI is installed in the devcontainer (currently pinned to `0.0.377`).
	- Copilot CLI is authenticated/configured such that `copilot -p` can run.
	- Atomic executor prompts can be written to a file under the workspace.
- Constraints (budget, performance, compatibility):
	- The Copilot invocation must be cross-platform and minimize OS-specific branching.
	- Must avoid Windows command-line length limitations when supplying long prompts.
	- Security: increasing Copilot permissions (tools/paths/urls) increases risk; the fix should prefer least privilege that still permits required QC.
- External dependencies (services, libraries, releases):
	- GitHub Copilot CLI behavior for `-p`, tool permissions, and `@path` expansion (validated locally for `0.0.377`).

## Data / API / Config Impact
- User-facing or API changes:
	- No new public CLI surface is required.
	- Behavior change is internal to the atomic executor: it invokes Copilot differently.
- Data or migration considerations:
	- No data migrations.
	- Prompt file location may be standardized under a workspace-scoped directory (to support `@path` references reliably).
- Logging/telemetry updates (if any):
	- Log the Copilot CLI argv (already effectively captured today) and explicitly log that `-p` mode is being used.
	- If a permission failure occurs, include the detected failure text and guidance about required flags.

## Test Strategy
- [x] Unit coverage areas
	- Update existing unit tests for `scripts.dev_tools.atomic_executor.cli.run_copilot()` to assert `-p` invocation and `@prompt_file` usage.
	- Add a regression assertion that the previous behavior (prompt via stdin without `-p`) is not used.
- [ ] Integration scenario to retest
	- Run `execute-all` against a multi-task plan that triggers scoped QC.
	- Confirm Copilot runs at least one shell command successfully within the session.
- [ ] Manual verification notes
	- Capture and attach a short excerpt of the atomic executor log showing:
		- Copilot started with `-p`
		- Copilot successfully executed a tool command (e.g., `python3 --version` or `poetry --version`)
		- No “Permission denied and could not request permission from user” in the session

Manual verification notes (captured):

- See `.agent_logs/atomic_executor_2026-01-11_230622.log` excerpt in “Repro & Evidence” above for a representative “Ran `python -m poetry run pytest ...` successfully” entry.


## Acceptance Criteria
- Conditions that must be true for the bug to be considered fixed (map to repro and edge cases).

- [x] Atomic executor invokes Copilot CLI with `-p/--prompt` (programmatic mode) for all platforms.
- [x] Atomic executor passes long prompt content via an on-disk prompt file referenced by `@<path>` in the `-p` prompt, avoiding OS command-line length limits.
- [ ] During a real `execute-all` run, Copilot can execute required QC commands reliably using `python -m poetry run ...` without producing “Permission denied and could not request permission from user”.
- [x] The existing unit tests that previously enforced “no `-p`” are updated to enforce the new invocation contract.
- [x] No new interactive prompts are required to complete the run in the devcontainer environment.
- [x] If Copilot cannot proceed due to permissions, the atomic executor fails fast with an actionable error message (no 300s silent hang leading to idle-timeout termination).

- [ ] Generated prompts do not contain instructions that require interactive mode (e.g., `/model`).
- [ ] Generated prompts do not instruct `poetry run ...` for QC in this environment; they instruct `python -m poetry run ...`.
- [ ] Prompt template placeholder substitution produces correct “Authoritative Documents” paths for the targeted feature folder.

- [ ] If remediation requires expanding Copilot path permissions, the decision is documented in this spec (why required, what flag was added, and the security tradeoff).

## Risks & Mitigations
- Technical or operational risks:
	- Risk: Copilot may request additional permissions (paths/tools/urls) beyond the current allowlist depending on the model’s plan.
	- Risk: Overly broad permissions (e.g., `--allow-all-tools`, `--allow-all-paths`) increase the blast radius of a bad suggestion.
	- Risk: `@path` expansion could fail if the prompt file is outside Copilot’s allowed/trusted directories.
- Mitigations and rollbacks:
	- Prefer least-privilege tool allowlists for known QC commands; expand only if needed and document why.
	- Ensure prompt files are written under the workspace so `@path` expansion is stable.
	- Roll back by restoring the prior Copilot invocation strategy (no `-p` + stdin prompt file), but note this reintroduces the original blocker.

## Rollout & Follow-up
- Release/rollout steps:
	- Implement and merge the atomic executor Copilot invocation change.
	- Re-run the original repro command on the devcontainer to confirm the blocker is resolved.
- Post-fix monitoring or clean-up tasks:
	- Monitor `.agent_logs/copilot_sessions/*` for any remaining permission-denied text.
	- If additional permissions are consistently required, document and tighten the allowlist/available-tools strategy accordingly.
- Links: issue, PRs, related docs
	- Issue #83: https://github.com/drmoisan/lexile-corpus-tuner/issues/83
	- Research notes: `docs/features/active/2026-01-11-copilot-cli-permission-failure-83/20260111-copilot-cli-permission-failure-83-research.md`
