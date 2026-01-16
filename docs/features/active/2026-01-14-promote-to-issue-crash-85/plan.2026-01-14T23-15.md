- Issue: #85
# 2026-01-14-promote-to-issue-crash (Plan)

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

- **Issue:** #85
- **Owner:** drmoisan
- **Date:** 2026-01-14T23-15
- **Status:** Planned
- **Outcome:** Prevent Windows `UnicodeEncodeError` when promoting potential issues by ensuring the `gh` subprocess stdin is encoded deterministically as UTF-8.
- **Root Cause:** `scripts/dev_tools/potential_to_issue.py::RealGhClient._run()` uses `subprocess.run(..., text=True, input=<str>)` without `encoding=` / `errors=`; on Windows this can default to a non-UTF-8 code page and crash when encoding Unicode (example: `→`).

| Requirement ID | Requirement (deterministic) | Source | Acceptance Validation |
| --- | --- | --- | --- |
| REQ-85-001 | `RealGhClient._run()` MUST call `subprocess.run(..., text=True, encoding="utf-8", errors="replace", ...)` to make stdin encoding deterministic across OSes. | `docs/features/active/2026-01-14-promote-to-issue-crash-85/spec.md` | `poetry run pytest tests/scripts/dev_tools/test_potential_to_issue.py::test_real_gh_client_run_uses_utf8_encoding_for_unicode_body -q` exits 0. |
| REQ-85-002 | Unicode body content (e.g., `→`) MUST be passed unchanged as the `input` argument to `subprocess.run` from `RealGhClient._run()`. | `spec.md` | Same pytest as REQ-85-001 exits 0 (asserts `input == body`). |
| REQ-85-003 | The change MUST NOT alter non-zero exit-code handling: `_run()` MUST still return `GhResult(exit_code=<returncode>)` and include combined stdout+stderr lines. | `spec.md` | `poetry run pytest tests/scripts/dev_tools/test_potential_to_issue.py::test_real_gh_client_invokes_subprocess -q` exits 0. |
| REQ-85-004 | Add a deterministic regression test that fails before the fix and passes after, without spawning real subprocesses or calling `gh`. | `spec.md` | New test exists and fails on pre-fix code; passes post-fix; entire pytest suite exits 0. |

| Traceability ID | Requirement ID | Task IDs |
| --- | --- | --- |
| TRACE-85-001 | REQ-85-001 | P2-T1, P3-T1, P4-T1 |
| TRACE-85-002 | REQ-85-002 | P2-T1, P3-T1, P4-T1 |
| TRACE-85-003 | REQ-85-003 | P4-T2 |
| TRACE-85-004 | REQ-85-004 | P2-T1, P2-T2 |

### Phase 0 — Context & Inputs

- [ ] [P0-T1] Read `docs/features/active/2026-01-14-promote-to-issue-crash-85/spec.md` and confirm the proposed fix is limited to `scripts/dev_tools/potential_to_issue.py::RealGhClient._run()`.
	- Acceptance: The executor can cite the exact target call site: `scripts/dev_tools/potential_to_issue.py` lines 96-102 (the `subprocess.run` call inside `RealGhClient._run`, which starts at line 91).
- [ ] [P0-T2] Record baseline git state by capturing `git rev-parse HEAD` and `git status --porcelain` on branch `fix/fix-all-84`.
	- Acceptance: Both commands exit code 0.
- [ ] [P0-T3] Capture baseline quality gate results (do not change code yet):
	- Run: `poetry run ruff check`
	- Run: `poetry run pyright`
	- Run: `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`
	- Acceptance: Each command exits code 0 OR failures are recorded verbatim for comparison after the fix.

### Phase 1 — Preparation

- [ ] [P1-T1] Confirm test location for the new regression: use `tests/scripts/dev_tools/test_potential_to_issue.py` (append near the existing subprocess tests around line 465).
	- Acceptance: File path exists and is collected by pytest (confirmed by running `poetry run pytest -q tests/scripts/dev_tools/test_potential_to_issue.py` exiting 0 OR failing only due to known baseline failures from P0-T3).
- [ ] [P1-T2] Identify exact edit location for the production fix in `scripts/dev_tools/potential_to_issue.py`:
	- Target: `RealGhClient._run()` (starts at line 91) and the `subprocess.run` call (lines 96-102).
	- Acceptance: The executor can point to the file and line range without ambiguity.

### Phase 2 — Regression Test (must fail first)

- [ ] [P2-T1] Add regression test `test_real_gh_client_run_uses_utf8_encoding_for_unicode_body` to `tests/scripts/dev_tools/test_potential_to_issue.py` immediately after `test_real_gh_client_invokes_subprocess` (starts at line 465).
	- Implementation details (deterministic):
		- Monkeypatch `scripts.dev_tools.potential_to_issue.shutil.which` to return a non-empty string (e.g., `"/usr/bin/gh"`).
		- Monkeypatch `scripts.dev_tools.potential_to_issue.subprocess.run` with a stub that:
			- Captures the `kwargs` passed to `subprocess.run`.
			- Returns an object with `returncode=0`, `stdout=""`, `stderr=""`.
		- Call: `client = mod.RealGhClient()` then `client.issue_create("Title", "Contains unicode →", "feature")`.
		- Assert that the captured kwargs include all of the following exact values:
			- `kwargs["text"] is True`
			- `kwargs["encoding"] == "utf-8"`
			- `kwargs["errors"] == "replace"`
			- `kwargs["input"] == "Contains unicode →"`
	- Acceptance: The new test exists and is discoverable by pytest (collection output includes the test name).
- [ ] [P2-T2] Run the new regression test only (expected to fail before the fix):
	- Command: `poetry run pytest tests/scripts/dev_tools/test_potential_to_issue.py::test_real_gh_client_run_uses_utf8_encoding_for_unicode_body -q`
	- Acceptance: Command exits non-zero and fails due to missing `encoding`/`errors` kwargs (or equivalent assertion failure).

### Phase 3 — Minimal Fix

- [ ] [P3-T1] Update `scripts/dev_tools/potential_to_issue.py::RealGhClient._run()` to pass explicit UTF-8 encoding to `subprocess.run`.
	- Exact edit location: `scripts/dev_tools/potential_to_issue.py` lines 96-102.
	- Exact change: add keyword arguments `encoding="utf-8"` and `errors="replace"` to the `subprocess.run(...)` call.
	- Non-goals (must not change):
		- Do not change `issue_create()` argument list (`--body-file -` must remain).
		- Do not change return-shaping (`GhResult(output=..., exit_code=...)`).
	- Acceptance: The only behavior change in `_run()` is the presence of explicit `encoding`/`errors` kwargs.

### Phase 4 — Verification Loop

- [ ] [P4-T1] Re-run the regression test from Phase 2 and confirm it now passes.
	- Command: `poetry run pytest tests/scripts/dev_tools/test_potential_to_issue.py::test_real_gh_client_run_uses_utf8_encoding_for_unicode_body -q`
	- Acceptance: Command exits 0.
- [ ] [P4-T2] Run the full Python toolchain loop until one clean pass succeeds (restart from Black if any step changes files or fails):
	- Step 1: `poetry run black .`
	- Step 2: `poetry run ruff check`
	- Step 3: `poetry run pyright`
	- Step 4: `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`
	- Acceptance: In the final pass, all four steps exit 0 with no file changes.

### Phase 5 — Documentation & Status

- [ ] [P5-T1] Update `docs/features/active/2026-01-14-promote-to-issue-crash-85/spec.md` Status from `Draft` to `Completed` and add a short “Implementation Notes” subsection describing:
	- The exact code change (`encoding="utf-8"`, `errors="replace"` in `RealGhClient._run`).
	- The regression test name and location.
	- Acceptance: `spec.md` contains the literal string `Status: Completed` and mentions `test_real_gh_client_run_uses_utf8_encoding_for_unicode_body`.

### Phase 6 — PR & Handoff

- [ ] [P6-T1] Prepare PR notes in the PR description including:
	- Root cause summary (Windows default encoding in subprocess text mode).
	- Fix summary (explicit UTF-8 encoding).
	- Verification summary (commands run in Phase 4).
	- Acceptance: PR description includes the literal command lines from P4-T2.

### Phase 7 — Rollout / Follow-up

- [ ] [P7-T1] Record a follow-up item (new backlog row in `docs/features/backlog.md`) to standardize subprocess wrappers across `scripts/dev_tools/*` (no code changes in this task).
	- Acceptance: `docs/features/backlog.md` contains a new row mentioning “standardize subprocess encoding (utf-8 + replace)”.
- [ ] [P7-T2] Update this plan file status to `Completed` after the full toolchain pass succeeds.
	- Acceptance: This file contains `**Status:** Completed` and the badge URL contains `status-Completed`.
