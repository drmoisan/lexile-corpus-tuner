# promote-to-issue-crash (Issue 85)

- Date captured: 2026-01-14
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/2026-01-14-promote-to-issue-crash-85/ (Issue #85)

> Automation note: Keep the section headings below unchanged; the promotion tooling maps each of them into the GitHub bug issue template.

## Summary

Promoting a potential feature file to a GitHub issue via `scripts.dev_tools.potential_to_issue` crashes on Windows when the generated issue body contains Unicode characters that are not representable in the system ANSI code page (commonly CP1252).

This prevents the promotion workflow from creating the issue, even though the `gh` CLI is installed and authenticated.

## Environment

- OS/version: Windows (observed on Windows 11; likely affects any Windows where the preferred encoding is CP1252)
- Python version: 3.13.x
- Command/flags used:
	- `poetry run python -m scripts.dev_tools.potential_to_issue --potential-path docs/features/potential/2026-01-14-fix-all-updates.md --promotion-type feature`
- Data source or fixture:
	- Potential markdown file contains the Unicode right arrow character `→` (U+2192) in the body text.
	- Example line from the potential file:
		- `formatting → validation ("linting") → PASS/FAIL`

## Steps to Reproduce

1. On Windows, ensure `gh` is installed and authenticated (for example, `gh auth status` succeeds).
2. Create (or use an existing) potential file whose content includes the right arrow `→` character (U+2192). For example, add a line like:
	- `formatting → validation ("linting") → PASS/FAIL`
3. Run the promotion tool with that file, e.g.:
	- `poetry run python -m scripts.dev_tools.potential_to_issue --potential-path <path-to-potential.md> --promotion-type feature`
4. Observe that the tool crashes before `gh issue create` completes.

## Expected Behavior

The tool should successfully create the GitHub issue (via `gh issue create`) and then complete the promotion workflow (e.g., move the potential file to the promoted/archive location and/or write back any links depending on the promotion type).

## Actual Behavior

The tool crashes with a `UnicodeEncodeError` while attempting to pipe the issue body into the `gh` subprocess.

Key error text:

`UnicodeEncodeError: 'charmap' codec can't encode character '\u2192' ...`

## Logs / Screenshots

- [ ] Attached minimal logs or screenshot
- Snippet:

	```
	UnicodeEncodeError: 'charmap' codec can't encode character '\u2192' in position 1418: character maps to <undefined>
  
	...
	File "...\scripts\dev_tools\potential_to_issue.py", line 96, in _run
		proc = subprocess.run(
				[gh_exe, *args],
				input=body,
				text=True,
				capture_output=True,
				check=False,
		)
	...
	```

## Impact / Severity

- [x] Blocker
- [ ] High
- [ ] Medium
- [ ] Low

## Suspected Cause / Notes

Root cause is the interaction between:

- The potential file is read as UTF-8 (good): `Path.read_text(encoding="utf-8")`.
- The issue body is passed to `gh` via stdin using `subprocess.run(..., input=body, text=True, ...)`.

On Windows, when `text=True` and the subprocess input is a Python `str`, `subprocess` encodes the string using the process “preferred encoding” unless an explicit `encoding=` is provided.

Commonly, that encoding is CP1252 (a.k.a. “charmap” in the exception), which cannot represent many Unicode characters including `→` (U+2192). This causes the crash before the `gh` CLI ever receives the full body.

Related code to inspect:

- `scripts/dev_tools/potential_to_issue.py`
	- `RealGhClient._run()` (stdin piping uses `text=True` but does not specify `encoding=`)
	- `RealFileSystem.read_text()` (reads UTF-8 correctly)

## Proposed Fix / Validation Ideas

### Durable fix (recommended)

- In `RealGhClient._run()`, pass an explicit UTF-8 encoding when running the subprocess:
	- `subprocess.run(..., text=True, encoding="utf-8", ...)`
	- Consider also setting `errors="strict"` to fail fast on unexpected encoding issues.

This ensures the stdin encoding is deterministic and supports Unicode issue bodies across platforms.

### Alternative durable fix

- Avoid `text=True` for stdin entirely and pass bytes:
	- `subprocess.run(..., input=body.encode("utf-8"), text=False, ...)`

### Workarounds (not ideal)

- Replace/normalize unsupported characters in the markdown (e.g., replace `→` with `->`).
- Run Python in UTF-8 mode (e.g., `PYTHONUTF8=1` / `-X utf8`). This is more environment-dependent than fixing the tool.

### Validation ideas

- Unit coverage areas:
	- Add a unit test that calls the `GhClient` path (ideally via a stubbed `subprocess.run`) using a body containing `→` and asserts no encode crash occurs.
	- Test that promotion still works for ASCII-only content.
- Integration scenario to retest:
	- Run `potential_to_issue` on Windows against a potential file that includes `→`, smart quotes, and/or em-dashes.
- Manual verification notes:
	- Confirm `gh issue create` succeeds and the created issue body renders those characters correctly on GitHub.

## Next Step

- [ ] Promote to GitHub issue (bug-report template)
- [ ] Move to active fix folder / branch
