- Issue: #83
# 2026-01-11-copilot-cli-permission-failure (Plan)

- **Issue:** #83
- **Owner:** drmoisan
- **Date:** 2026-01-11T14-45
- **Status:** Planned
- **Outcome:** Copilot sessions invoked by the atomic executor can run required shell commands (Poetry/Python/Git) and perform file edits without interactive approval prompts; the executor no longer hits “Permission denied and could not request permission from user” during Copilot-driven QC work.
- **Root Cause:** `scripts/dev_tools/atomic_executor/cli.py::run_copilot()` invokes `copilot` without programmatic mode (`-p/--prompt`) and feeds the prompt via a non-interactive stdin file (`stdin=prompt_f` at `cli.py:585`), preventing Copilot from requesting/receiving approvals. Existing regression test enforces this old contract (`assert "-p" not in captured_argv` at `tests/scripts/dev_tools/atomic_executor/test_cli.py:1150`).

![Status: Planned](https://img.shields.io/badge/Status-Planned-blue)

| REQ-ID | Requirement (deterministic) | Source |
| --- | --- | --- |
| REQ-01 | Invoke Copilot CLI in programmatic mode (`copilot -p/--prompt`) on all platforms. | `spec.md` “Proposed Fix”, “Acceptance Criteria” |
| REQ-02 | Avoid Windows argv-length issues by storing the full prompt body in a file and referencing it from the `-p` prompt using an `@<path>` mention. | `spec.md` “Proposed Fix”; research confirms `@path` expands in `-p` |
| REQ-03 | Preserve/extend required tool permissions for headless QC (`write`, `shell(poetry)`, `shell(python3)` and/or `shell(python)`, `shell(git)`). | `spec.md` “Proposed Fix” |
| REQ-04 | Update unit tests that currently enforce “no `-p`” to enforce the new invocation contract instead (TDD: test must fail before fix). | `spec.md` “Test Strategy”, “Acceptance Criteria” |
| REQ-05 | Ensure failures due to permissions are non-hanging and actionable (no 300s silent idle-timeout termination). | `spec.md` “Acceptance Criteria” |


**Phase 0 — Context & Inputs**

- [x] [P0-T1] Read repo Copilot instructions (policy precondition)
	- File: `.github/copilot-instructions.md`
	- Acceptance: Key constraints captured (no secrets, follow toolchain, doc expectations).
- [x] [P0-T2] Read general code-change policy (policy precondition)
	- File: `.github/instructions/general-code-change.instructions.md`
	- Acceptance: Confirm the required toolchain loop order (format → lint → type-check → test) and reporting expectations.
- [x] [P0-T3] Read general unit-test policy (policy precondition)
	- File: `.github/instructions/general-unit-test.instructions.md`
	- Acceptance: Confirm determinism/isolation requirements and “no temporary files in tests” constraints.
- [x] [P0-T4] Read Python code-change policy (policy precondition)
	- File: `.github/instructions/python-code-change.instructions.md`
	- Acceptance: Confirm approved commands for Black/Ruff/Pyright and suppression rules.
- [x] [P0-T5] Read Python unit-test policy (policy precondition)
	- File: `.github/instructions/python-unit-test.instructions.md`
	- Acceptance: Confirm approved pytest + coverage command and test organization rules.

- [x] [P0-T6] Link approved spec for Issue #83 (REQ-01..REQ-05)
	- File: `docs/features/active/2026-01-11-copilot-cli-permission-failure-83/spec.md`
	- Acceptance: Spec file exists and matches the planned requirements table above.
- [x] [P0-T7] Record branch/commit baseline in log artifact (REQ-05)
	- Run:
		- `git rev-parse --abbrev-ref HEAD`
		- `git rev-parse HEAD`
	- Write results to: `artifacts/research/20260111-copilot-cli-permission-failure-83-baseline.txt`
	- Acceptance: Artifact file contains both branch name and commit SHA.
- [x] [P0-T8] Capture Copilot CLI runtime baseline (REQ-01)
	- Run:
		- `copilot --version`
		- `copilot --help` (store output excerpt containing `-p, --prompt`)
	- Write results to: `artifacts/research/20260111-copilot-cli-permission-failure-83-copilot-help.txt`
	- Acceptance: Artifact includes the `-p, --prompt` help line.
- [x] [P0-T9] Capture QC baseline without modifying files (REQ-05)
	- Run:
		- `poetry run black . --check`
		- `poetry run ruff check`
		- `poetry run pyright`
		- `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`
	- Acceptance: Baseline pass/fail results are recorded in `artifacts/research/20260111-copilot-cli-permission-failure-83-baseline.txt`.

**Phase 1 — Preparation**
- [x] [P1-T1] Confirm scope is locked for this fix (REQ-01..REQ-05)
	- Read: `docs/features/active/2026-01-11-copilot-cli-permission-failure-83/spec.md`
	- Acceptance: No “TBD”/placeholders remain in the spec sections impacting implementation.
- [x] [P1-T2] Confirm target code locations and current line anchors (REQ-01, REQ-04)
	- Verify definitions exist:
		- `scripts/dev_tools/atomic_executor/cli.py::run_copilot` (starts at `cli.py:365`)
		- `stdin=prompt_f` passed to `subprocess.Popen` (at `cli.py:585`)
		- Regression test assertion `assert "-p" not in captured_argv` (at `test_cli.py:1150`)
	- Acceptance: All anchors are present before edits begin.


**Phase 2 — Regression Test (must fail first)**
- [x] [P2-T1] Update regression test to require programmatic mode (`-p`) and forbid stdin-prompt feeding (REQ-01, REQ-04)
	- File: `tests/scripts/dev_tools/atomic_executor/test_cli.py`
	- Test to modify: `TestRunCopilot.test_run_copilot_invokes_with_correct_arguments` (contains old assertion at `test_cli.py:1150`)
	- Exact new assertions (all must be added, all deterministic):
		- `assert "-p" in captured_argv` OR `assert "--prompt" in captured_argv`
		- Assert the `-p/--prompt` argument value contains an `@` mention pointing at the prompt file path under the log dir, e.g. `@.../prompts/prompt_<run_id>_<task_id>.md`
		- Assert `stdin` is not used to provide the prompt (e.g., captured stdin content is `None` and/or `stdin` kwarg absent)
		- Assert tool approvals remain present: `--allow-tool write`, `--allow-tool shell(poetry)`, `--allow-tool shell(git)`
	- Acceptance: Test file saves cleanly and the updated assertions reflect REQ-01/REQ-04.
- [x] [P2-T2] Add regression assertion that prompt file is still written to disk for `@path` expansion (REQ-02)
	- File: `tests/scripts/dev_tools/atomic_executor/test_cli.py`
	- In the same test (`test_run_copilot_invokes_with_correct_arguments`), assert:
		- The expected prompt file path exists under `log_file.parent / "prompts"` with name format `prompt_<run_id>_<task_id>.md`
		- Its contents equal the provided `prompt_text` passed to `run_copilot`
	- Acceptance: Assertions are present and refer to the exact on-disk path format used by `cli.py`.
- [x] [P2-T3] Run the modified regression test and confirm it fails before the fix (REQ-04)
	- Run:
		- `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_cli.py::TestRunCopilot::test_run_copilot_invokes_with_correct_arguments -q`
	- Acceptance: Test fails with an assertion mismatch that clearly indicates the current code still omits `-p` and/or still uses stdin for the prompt.

- [x] [P2-T4] Add regression test that permission-denied output fails fast with actionable context (REQ-05)
	- File: `tests/scripts/dev_tools/atomic_executor/test_cli.py`
	- Add a new test under `TestRunCopilot`:
		- Name: `test_run_copilot_permission_denied_fails_fast_with_actionable_error`
		- Scenario setup: Mock `subprocess.Popen` such that `stdout.read(...)` yields bytes containing the exact message substring:
			- `Permission denied and could not request permission from user`
		- Assertions (deterministic):
			- `run_copilot(...)` raises a specific exception type used by the implementation (e.g., `RuntimeError`).
			- Exception message includes:
				- the known substring above
				- a sanitized/serialized `argv` summary (must include `copilot` and `-p`/`--prompt`)
				- the allow-tool entries present (must include `write` and `shell(poetry)`)
	- Acceptance: Test is added and fails before implementation changes (either by hanging or by not raising the expected exception).

- [x] [P2-T5] Run the new permission-denied regression test and confirm it fails before the fix (REQ-05)
	- Run:
		- `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_cli.py::TestRunCopilot::test_run_copilot_permission_denied_fails_fast_with_actionable_error -q`
	- Acceptance: Test fails with a clear mismatch (no exception / missing message content / hang-protection failure).

**Phase 3 — Minimal Fix**
- [x] [P3-T1] Switch Copilot invocation to programmatic mode and reference the prompt file via `@path` (REQ-01, REQ-02)
	- File: `scripts/dev_tools/atomic_executor/cli.py`
	- Function: `run_copilot` (starts at `cli.py:365`)
	- Exact change:
		- Keep writing the prompt to `prompt_file` (currently `prompt_file.write_text(...)` at `cli.py:528`).
		- Modify argv to include `-p` (or `--prompt`) with a short prompt string that includes `@{prompt_file}`.
			- Required prompt string prefix: `Follow these instructions exactly: `
			- Required prompt string: `Follow these instructions exactly: @{prompt_file}`
		- Remove passing the prompt via stdin (`stdin=prompt_f` at `cli.py:585`) and do not open the prompt file as stdin.
	- Acceptance: `subprocess.Popen(...)` is called without a stdin prompt file, and `captured_argv` includes `-p/--prompt`.
- [x] [P3-T2] Ensure required tool approvals remain explicit and include both Python command variants (REQ-03)
	- File: `scripts/dev_tools/atomic_executor/cli.py`
	- Location: argv permissions block beginning at `cli.py:548`
	- Exact change:
		- Preserve existing `--allow-tool write`, `shell(poetry)`, `shell(git)`.
		- Add `--allow-tool shell(python3)` in addition to existing `shell(python)` to reduce environment variance.
	- Acceptance: argv contains both `shell(python)` and `shell(python3)`.
- [x] [P3-T3] Add workspace directory allowance to reduce file-access prompts in headless runs (REQ-05)
	- File: `scripts/dev_tools/atomic_executor/cli.py`
	- Exact change:
		- Add `--add-dir` with value `str(workspace)` to Copilot argv before invoking `subprocess.Popen`.
	- Acceptance: argv includes `--add-dir` and the resolved workspace path.

- [x] [P3-T4] Fail fast with actionable error on “permission denied” output (REQ-05)
	- File: `scripts/dev_tools/atomic_executor/cli.py`
	- Function: `run_copilot`
	- Exact change:
		- Detect the substring `Permission denied and could not request permission from user` in Copilot output.
		- Immediately abort the run (raise) with an exception that includes:
			- the detected substring
			- a sanitized argv summary (include `-p/--prompt` and `--allow-tool` entries)
			- explicit guidance to adjust tool/path permissions (e.g., add `--add-dir`, expand `--allow-tool`, or run the command manually if policy blocks headless execution)
		- Ensure the failure occurs promptly and does not wait for the idle-timeout.
	- Acceptance: The new test `test_run_copilot_permission_denied_fails_fast_with_actionable_error` passes and the failure path does not rely on the 300s idle-timeout.

**Phase 4 — Verification Loop**
- [x] [P4-T1] Re-run the regression test and confirm it passes (REQ-01..REQ-04)
	- Run:
		- `poetry run pytest tests/scripts/dev_tools/atomic_executor/test_cli.py::TestRunCopilot::test_run_copilot_invokes_with_correct_arguments -q`
	- Acceptance: Test passes.
- [x] [P4-T2] Run a local end-to-end repro of the original failing workflow (REQ-05)
	- Run (example from issue):
		- `poetry run python -m scripts.dev_tools.atomic_executor.cli execute-all docs/features/active/2026-01-06-populate-open-stax-ck-12-manifest-73/v4/ --workspace /workspaces/lexile-corpus-tuner --preferred-model gpt-5.1-codex-max --max-fix-attempts 10`
	- Acceptance: During the Copilot session, Copilot can run at least one shell tool command (e.g., `python3 --version` or `poetry --version`) and does not emit “Permission denied and could not request permission from user”.
- [x] [P4-T3] Run the full toolchain loop and restart from formatting if any step fails/changes files (REQ-05)
	- Loop until a single clean pass succeeds:
		1) `poetry run black .`
		2) `poetry run ruff check`
		3) `poetry run pyright`
		4) `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`
	- Acceptance: One complete pass succeeds with no failures.

**Phase 5 — Documentation & Status**
- [x] [P5-T1] Update Issue #83 documentation with final behavior and validation evidence (REQ-01..REQ-05)
	- Files:
		- `docs/features/active/2026-01-11-copilot-cli-permission-failure-83/spec.md`
		- `docs/features/active/2026-01-11-copilot-cli-permission-failure-83/issue.md`
	- Exact updates:
		- Record that the fix is `copilot -p` + `@prompt_file` and that stdin prompting is removed.
		- Add the exact validation command used and a short log excerpt showing success.
	- Acceptance: Both docs reflect the implemented invocation contract and validation steps.

**Phase 6 — PR & Handoff**
- [x] [P6-T1] Prepare PR notes with deterministic verification checklist and risk statement (REQ-05)
	- Include:
		- Summary of changes (files + key symbols)
		- Regression test name and command
		- End-to-end repro command and result
		- Risk: permissions broadening (tools/dirs) and mitigations
	- Acceptance: PR description contains all items above.
	- Notes captured in: `docs/features/active/2026-01-11-copilot-cli-permission-failure-83/pr-notes.md`

**Phase 7 — Rollout / Follow-up**
- [x] [P7-T1] Capture deployment/rollout notes and post-fix monitoring items
	- Notes captured in: `docs/features/active/2026-01-11-copilot-cli-permission-failure-83/rollout.md`
- [x] [P7-T2] Record links (issue, PRs, related docs) for traceability
	- Links captured in: `docs/features/active/2026-01-11-copilot-cli-permission-failure-83/rollout.md`

