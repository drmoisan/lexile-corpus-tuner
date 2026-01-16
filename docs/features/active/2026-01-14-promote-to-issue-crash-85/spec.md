# Promote Potential Issue Crashes On Windows With Unicode Body (Spec)

- Issue: #85
- Owner: TBD
- Date: 2026-01-14
- Status: Draft

## Context

This bug affects the developer tool `scripts/dev_tools/potential_to_issue.py`, which promotes a
markdown “potential issue” document into a real GitHub issue using the `gh` CLI.

Impact:

- On Windows, promotion can crash before the GitHub issue is created when the markdown body
	contains Unicode characters that are not representable in the active Windows ANSI code page
	(example observed: `→`).
- The failure is disruptive because it blocks the “promote potential” workflow and forces users
	into manual workarounds.

Observed environment(s):

- Windows (PowerShell / VS Code terminal).
- Python running with default locale encoding behavior (UTF-8 Mode not assumed).

Related research:

- `20260115-issue-85-windows-unicode-encoding-research.md`

## Repro & Evidence

Steps to reproduce:

1. Create or edit a potential issue markdown file containing non-ASCII Unicode, for example:
	- A line that contains `→`.
2. Run the promote tool in Windows, for example via the script entry point (exact invocation may
	vary based on how the tool is called in the repo):
	- Promote the file as a bug/feature so it calls `RealGhClient.issue_create(...)`.
3. Observe the tool crash during issue creation.

Expected behavior:

- The tool creates the GitHub issue successfully.
- The GitHub issue body preserves the original Unicode content (including `→`).
- The tool updates the potential file metadata and moves the file to the promoted folder.

Actual behavior:

- The tool can raise a `UnicodeEncodeError` while attempting to pass the issue body over stdin to
  `gh` via `subprocess.run(..., text=True, input=body, ...)`.
- The issue is not created.

Evidence (code-level):

- `scripts/dev_tools/potential_to_issue.py::RealGhClient._run()` uses `subprocess.run(...,
  text=True, input=body, capture_output=True)` without specifying `encoding=` / `errors=`.
- Repo precedent demonstrates this exact class of Windows failure and uses `encoding="utf-8",
  errors="replace"` in subprocess wrappers:
  - `scripts/dev_tools/pr_context/git.py`
  - `scripts/dev_tools/collect_commit_context.py`

## Scope & Non-Goals

In scope:

- Fix Windows Unicode promotion crashes by making subprocess text-mode encoding deterministic.
- Preserve Unicode issue title/body content end-to-end.
- Add a deterministic unit test that prevents regressions by asserting the subprocess call uses
	explicit UTF-8 encoding.

Out of scope / non-goals:

- Changing the on-disk encoding of potential markdown files (already UTF-8 in `RealFileSystem`).
- Converting issue bodies to ASCII or stripping Unicode beyond the existing
	`normalize_smart_punctuation()` behavior.
- Requiring users to enable Python UTF-8 Mode, change Windows system locale, or change terminal
	code pages.
- Adding new dependencies.

## Root Cause Analysis

Confirmed root cause:

- The tool passes the GitHub issue body to the `gh` CLI via stdin using `subprocess.run` in text
	mode: `text=True` + `input=<str>`.
- In text mode, `subprocess.run` uses an `io.TextIOWrapper` internally. If the caller does not
	provide `encoding=` / `errors=`, the wrapper uses platform-dependent defaults.
- On Windows, those defaults can be a non-UTF-8 ANSI code page. When Python attempts to encode a
	Unicode character that is not representable in that code page (e.g. `→`), it raises
	`UnicodeEncodeError` before the subprocess even runs.

Signals/evidence:

- Code inspection shows `RealGhClient._run()` does not provide `encoding` or `errors`.
- The repo already solved similar Windows encoding pitfalls in other dev tooling by specifying
	`encoding="utf-8"` and `errors="replace"` on subprocess calls.
- Python documentation and PEPs describe platform-dependent default encodings and warn that this
	behavior frequently causes Windows-specific bugs when UTF-8 is assumed.

## Proposed Fix

Design summary:

- Update `scripts/dev_tools/potential_to_issue.py::RealGhClient._run()` to call
	`subprocess.run(..., text=True, encoding="utf-8", errors="replace", ...)`.
- Keep using `gh issue create --body-file -` and stdin piping; this is the correct `gh` interface
	for multiline bodies and avoids Windows argument quoting issues.

Boundaries and invariants to preserve:

- Do not change file I/O encoding behavior (already UTF-8).
- Do not change the `GhClient` protocol surface.
- Keep subprocess security posture unchanged:
	- Continue to use argument lists (no `shell=True`).
	- Continue to validate `gh` availability via `shutil.which("gh")` in `__post_init__`.
- Preserve existing normalization behavior:
	- `normalize_smart_punctuation()` remains a display/consistency improvement, not a general
		Unicode sanitization layer.

Dependencies or blocked work:

- None expected. This is a standard-library-only change.

## Assumptions, Constraints, Dependencies

Assumptions:

- `gh` is installed and authenticated (existing precondition).
- Potential markdown files may contain arbitrary Unicode.

Constraints:

- The fix must be deterministic across Windows and non-Windows platforms.
- Unit tests must not call `gh` or touch the network.
- Maintain compatibility with the repo’s supported Python versions.

External dependencies:

- GitHub CLI (`gh`) is required at runtime for real promotion (already a dependency).

## Data / API / Config Impact

User-facing or API changes:

- No new CLI flags or API changes.
- Users should simply observe that promotion no longer crashes for Unicode content.

Data or migration considerations:

- None.

Logging/telemetry updates:

- None required for the fix.

## Test Strategy

Regression tests to add or update:

- Add a unit test that exercises the real subprocess boundary *without spawning* a subprocess:
	- Monkeypatch `subprocess.run`.
	- Instantiate `RealGhClient` with a known `gh_path` (e.g., `"gh"` placeholder string).
	- Call `RealGhClient._run([...], body="Contains unicode →")`.
	- Assert `subprocess.run` was invoked with:
		- `text=True`
		- `encoding="utf-8"`
		- `errors="replace"`
		- `input` equal to the provided Unicode body (unchanged)

Edge cases and negative scenarios:

- Body contains arbitrary Unicode beyond smart punctuation (e.g., arrows, emoji, non-Latin text).
- `gh` subprocess returns non-zero exit codes: behavior should remain unchanged (tool reports
	output and exits non-zero).
- `gh_path` is missing/unresolved: behavior should remain unchanged (precondition error).

Manual validation steps (optional but recommended):

- On a Windows machine, run the promote flow against a potential markdown containing `→`.
- Confirm the resulting GitHub issue body preserves the Unicode characters.

## Acceptance Criteria

The bug is considered fixed when all of the following are true:

1. **No Windows UnicodeEncodeError during promotion**
	 - Given a potential issue body containing `→` (and other non-ASCII Unicode), running the
		 promotion flow does not raise `UnicodeEncodeError`.

2. **Unicode content is preserved in the created issue**
	 - The created GitHub issue body contains the original Unicode characters (not stripped or
		 replaced by ASCII normalization).

3. **Subprocess encoding is explicitly UTF-8 at the subprocess boundary**
	 - `RealGhClient._run()` passes `encoding="utf-8"` and `errors="replace"` to
		 `subprocess.run()`.
	 - This is validated by a deterministic unit test (monkeypatching `subprocess.run`) that fails
		 before the fix and passes after.

4. **No breaking changes to existing success/failure flows**
	 - Non-zero `gh` exit codes are still surfaced as tool failures with captured output.
	 - Successful promotions still update metadata and move the potential file as before.

## Risks & Mitigations

Technical/operational risks:

- Using `errors="replace"` can mask undecodable output from `gh` by replacing characters.
	- This is already an accepted repo tradeoff in existing subprocess wrappers.
	- Input encoding is forced to UTF-8, so the primary risk is only in decoding subprocess output.

Mitigations and rollbacks:

- Mitigation: align with established repo precedent (same encoding strategy used elsewhere).
- Rollback: revert the subprocess wrapper change if unexpected behavior occurs.

## Rollout & Follow-up

Release/rollout steps:

- Land fix + unit test.
- Run the full repo toolchain (format → lint → type-check → test) per project policy.

Post-fix monitoring / follow-up:

- Consider standardizing subprocess helpers for all `scripts/dev_tools/*` to reduce duplication
	(follow-up item; not required for this fix).

Links:

- Issue: #85
- Research: `20260115-issue-85-windows-unicode-encoding-research.md`
