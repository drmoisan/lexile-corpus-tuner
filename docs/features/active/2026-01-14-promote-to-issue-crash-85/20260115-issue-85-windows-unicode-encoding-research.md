<!-- markdownlint-disable-file -->

# Task Research Notes: Issue #85 — Windows Unicode crash when promoting potential issue (gh CLI)

## Research Executed

### File Analysis

- `scripts/dev_tools/potential_to_issue.py`
  - `RealGhClient._run()` calls `subprocess.run(..., text=True, input=body, capture_output=True)` without explicitly setting `encoding=` / `errors=`.
  - `issue_create()` uses `gh issue create --body-file -` and passes the issue body via stdin (`input=body`). This is the correct `gh` pattern for preserving Unicode content, but the Python `subprocess.run()` text-mode stdin encoding defaults can break on Windows if not forced to UTF-8.
  - File I/O already uses UTF-8 via `RealFileSystem.read_text(..., encoding="utf-8")` and `write_text(..., encoding="utf-8")`, so the crash is not from reading the markdown.
  - `normalize_smart_punctuation()` only replaces a small set of “smart punctuation” characters and does not (and should not) attempt to ASCII-fy all Unicode; therefore it cannot prevent failures from arbitrary Unicode like `→`.

- `scripts/dev_tools/pr_context/git.py`
  - `SubprocessRunner.run()` uses `subprocess.run(..., text=True, encoding="utf-8", errors="replace")` specifically to avoid Windows code-page failures.

- `scripts/dev_tools/collect_commit_context.py`
  - `run_git()` uses `subprocess.run(..., text=True, encoding="utf-8", errors="replace")` for the same cross-platform reason.

- `tests/scripts/dev_tools/test_potential_to_issue.py`
  - Uses `FakeGhClient` and does not exercise the real subprocess call path; therefore, it cannot catch missing `encoding=` / `errors=` parameters in `RealGhClient._run()`.

### Code Search Results

- `--body-file|def _run\(|input=body|text=True`
  - `scripts/dev_tools/potential_to_issue.py` matches:
    - `def _run(self, args: list[str], body: str | None = None) -> GhResult:`
    - `input=body,`
    - `text=True,`
    - `"--body-file",`

- `subprocess\.run\(|encoding="utf-8"|--body-file`
  - Many matches across `scripts/dev_tools/**` and tests (sampled), including:
    - `scripts/dev_tools/pr_context/git.py` uses `encoding="utf-8"` + `errors="replace"` in `subprocess.run`.
    - `scripts/dev_tools/collect_commit_context.py` uses `encoding="utf-8"` + `errors="replace"` in `subprocess.run`.

### External Research

- #fetch:https://docs.python.org/3/library/subprocess.html
  - Text mode (`text=True` / `universal_newlines`) uses the `encoding` and `errors` specified in the call, or defaults to `io.TextIOWrapper` defaults.
  - `subprocess.run()` explicitly supports `encoding=` and `errors=` (added in Python 3.6).

- #fetch:https://docs.python.org/3/library/io.html
  - The default encoding for `TextIOWrapper` / `open()` is locale-specific.
  - Python docs explicitly warn that “the locale encoding is not UTF-8 for most Windows users” and recommend specifying `encoding="utf-8"` when reading UTF-8 text.
  - This explains why relying on defaults is fragile cross-platform.

- #fetch:https://peps.python.org/pep-0540/
  - UTF-8 Mode exists, is not always enabled, and changes default encoding behavior.
  - PEP explicitly describes platform encoding variance and why “ignore locale, use UTF-8” is useful, but also notes backward compatibility constraints.

- #fetch:https://peps.python.org/pep-0686/
  - UTF-8 Mode is planned to become default in Python 3.15.
  - Motivation explicitly calls out platform-dependent defaults and recommends explicit `encoding="utf-8"` (or `encoding="locale"`) to avoid cross-platform bugs.

### Project Conventions

- Standards referenced: repo-established subprocess wrappers prefer `text=True, encoding="utf-8", errors="replace"` for portability.
- Instructions followed: `general-code-change.instructions.md`, `python-code-change.instructions.md`, `general-unit-test.instructions.md`, `python-unit-test.instructions.md` (for proposed test strategy only; no code changes made in this research note).

## Key Discoveries

### Project Structure

- `scripts/dev_tools/potential_to_issue.py` is responsible for turning a “potential” markdown into a GitHub issue using `gh`.
- The script already isolates file I/O behind `RealFileSystem` and GitHub CLI behind `RealGhClient`, enabling unit tests with fakes.
- The encoding bug is localized to the real subprocess boundary: `RealGhClient._run()`.

### Implementation Patterns

- The repo already contains two strongly relevant precedents that handle Windows encoding robustly:
  - `scripts/dev_tools/pr_context/git.py::SubprocessRunner.run()`
  - `scripts/dev_tools/collect_commit_context.py::run_git()`
- Both set `encoding="utf-8"` and `errors="replace"` when using `text=True`.

### Complete Examples

```python
# Source: scripts/dev_tools/potential_to_issue.py

def _run(self, args: list[str], body: str | None = None) -> GhResult:
    gh_exe = self.gh_path
    if gh_exe is None:
        raise RuntimeError("gh CLI path was not resolved")

    proc: subprocess.CompletedProcess[str] = subprocess.run(  # noqa: S603
        [gh_exe, *args],
        input=body,
        text=True,
        capture_output=True,
        check=False,
    )
    ...
```

```python
# Source: scripts/dev_tools/pr_context/git.py

completed = subprocess.run(  # noqa: S603
    args,
    cwd=str(cwd) if cwd else None,
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
    check=False,
    shell=False,
)
```

### API and Schema Documentation

- `subprocess.run(..., text=True)` uses an `io.TextIOWrapper` internally, so encoding defaults matter unless explicitly set.
- `subprocess.run()` supports `encoding=` and `errors=` which govern stdin/stdout/stderr conversion between `str` and `bytes`.

### Configuration Examples

```text
# gh CLI behavior (from scripts/dev_tools/potential_to_issue.py)
gh issue create --title <title> --body-file - --label <promotion_type>

# Implication:
# The body is passed over stdin ("-") and will be encoded by Python when using subprocess.run(..., text=True, input=body).
```

### Technical Requirements

- On Windows, the default locale encoding may be a legacy ANSI code page rather than UTF-8.
- When `subprocess.run(..., text=True, input=<str>)` attempts to encode characters not representable in that code page (e.g. `→`), Python raises `UnicodeEncodeError` before `gh` even runs.
- The fix must be at the subprocess boundary (explicit UTF-8), not via partial character replacement.

## Recommended Approach

Modify the real subprocess boundary in `RealGhClient._run()` to force UTF-8 text mode:

- Add `encoding="utf-8"` and `errors="replace"` to the existing `subprocess.run(..., text=True, input=body, capture_output=True)` call.
- Keep `--body-file -` usage (stdin) unchanged; it is the correct `gh` usage pattern for multiline issue bodies.

Rationale (evidence-based):

- Python’s own docs state that text-mode subprocess pipes use `io.TextIOWrapper` defaults unless encoding is specified (#fetch:subprocess).
- Python warns that locale default encodings are not UTF-8 for most Windows users (#fetch:io).
- Repo precedent already uses `encoding="utf-8"` + `errors="replace"` for subprocess calls (#file:scripts/dev_tools/pr_context/git.py and #file:scripts/dev_tools/collect_commit_context.py).

Rejected alternatives (non-exhaustive):

- Rely on Python UTF-8 Mode / future default behavior (PEP 540 / PEP 686): not deterministic today; depends on startup flags and Python version.
- Require users to change system code page / console settings: operationally fragile and outside the tool’s control.
- Expand `normalize_smart_punctuation()` into a broad Unicode-to-ASCII filter: loses meaning and still fails for arbitrary Unicode unless it aggressively strips content.

## Implementation Guidance

- **Objectives**:
  - Ensure `scripts/dev_tools/potential_to_issue.py` can create GitHub issues from markdown containing arbitrary Unicode (including `→`) on Windows.
  - Preserve Unicode content rather than replacing it.

- **Key Tasks**:
  - Update `RealGhClient._run()` to pass `encoding="utf-8"` and `errors="replace"` to `subprocess.run()`.
  - Add a unit test that patches `subprocess.run` and asserts `encoding` and `errors` are passed when `RealGhClient._run()` is invoked.
    - Test must not call `gh` or touch the network; it should be a pure unit test via monkeypatch.

- **Dependencies**:
  - No new dependencies expected (standard library only).

- **Success Criteria**:
  - On Windows, issue promotion does not throw `UnicodeEncodeError` when body contains `→`.
  - Unit test fails before fix and passes after fix by verifying subprocess invocation is configured with UTF-8.
