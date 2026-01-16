<!-- markdownlint-disable-file -->

# Task Research Notes: fix-all-updates (Issue #84)

## Research Executed

### File Analysis

- d:\repos\lexile-corpus-tuner-bg\scripts\dev_tools\fix_all.py
  - Current `run_fix_all()` executes 4 branches in parallel via `threading.Thread`, buffers branch logs to `StringIO`, and prints logs + a final “Branch Results” summary after joining threads.
  - Current semantics are explicitly “failure in one branch does not stop others” (docstring), and there is no CLI surface for fail-fast vs run-to-completion.
  - The “shell test” step always runs, but `scripts/dev_tools/shell_qc.py` can return 0 while printing “skipping” messages (meaning a live status board needs a way to represent “SKIP tests” even on exit code 0).

- d:\repos\lexile-corpus-tuner-bg\tests\scripts\dev_tools\test_fix_all.py
  - Test seam exists via `runner_factory` injection (per-branch fake runners), enabling deterministic unit tests for step ordering and failure behavior.
  - Existing tests validate per-branch step ordering and failure-short-circuiting *within* a branch, but do not validate cross-branch fail-fast semantics.

- d:\repos\lexile-corpus-tuner-bg\scripts\dev_tools\shell_qc.py
  - `run_test()` returns `0` when:
    - no test directories exist (“No shell test directories found; skipping.”), or
    - `bats` is missing (“bats not installed; skipping shell tests.”).
  - This creates an “exit code 0 but tests skipped” scenario.

- d:\repos\lexile-corpus-tuner-bg\docs\features\active\2026-01-14-fix-all-updates-84\issue.md
  - Captures requirements: add live “status board”; default fail-fast; add `--complete-all` to restore run-to-completion.

- d:\repos\lexile-corpus-tuner-bg\docs\features\active\2026-01-14-fix-all-updates-84\user-story.md
  - Explicitly targets Windows-first local dev and CI log readability; reiterates “no heavy terminal dependency unless strictly necessary.”

- d:\repos\lexile-corpus-tuner-bg\pyproject.toml
  - Confirms Poetry setup and `dev.fix-all = scripts.dev_tools.fix_all:main` entry point.
  - Confirms `rich` is not currently a dependency.

### Code Search Results

- run_fix_all
  - Found in `scripts/dev_tools/fix_all.py` and referenced by `tests/scripts/dev_tools/test_fix_all.py`.

- threading.Thread
  - Found in `scripts/dev_tools/fix_all.py` where branch functions are launched in parallel.

- complete-all
  - Found in feature docs (`issue.md`, `user-story.md`, `spec.md`); not present in `scripts/dev_tools/fix_all.py`.

### External Research

- #githubRepo:"(not executed)"
  - Tool not used in this research pass.

- #fetch:https://learn.microsoft.com/en-us/windows/console/getstdhandle
  - Confirms standard handle constants (e.g., `STD_OUTPUT_HANDLE` is `((DWORD)-11)`) and that `GetStdHandle` returns a handle to the console screen buffer (or redirected handle).

- #fetch:https://learn.microsoft.com/en-us/windows/console/getconsolemode
  - Confirms output console mode flags include `ENABLE_PROCESSED_OUTPUT` (required when using control sequences) and `ENABLE_VIRTUAL_TERMINAL_PROCESSING` (enables VT100-like processing).

- #fetch:https://learn.microsoft.com/en-us/windows/console/setconsolemode
  - Confirms `SetConsoleMode` sets console modes, and reiterates that `ENABLE_PROCESSED_OUTPUT` should be set when using `ENABLE_VIRTUAL_TERMINAL_PROCESSING`.

- #fetch:https://learn.microsoft.com/en-us/windows/console/console-virtual-terminal-sequences
  - Documents specific VT sequences relevant to a “status board” implementation:
    - Cursor movement: `ESC [ <n> A` (cursor up), etc.
    - Erase in line: `ESC [ <n> K` (e.g., `ESC[2K` to clear entire line).
    - Save/restore cursor: `ESC 7` and `ESC 8`.
  - Provides sample C code enabling VT processing by reading existing mode via `GetConsoleMode` and OR-ing `ENABLE_VIRTUAL_TERMINAL_PROCESSING`.

- #fetch:https://rich.readthedocs.io/en/stable/live.html
  - Confirms Rich provides a `Live` display with refresh and `update()` support, optional alternate screen (`screen=True`) and transient behavior (`transient=True`).
  - Confirms Rich can redirect stdout/stderr by default to avoid breaking live display output.

- #fetch:https://rich.readthedocs.io/en/stable/reference/live.html
  - Confirms the `Live` constructor supports `screen`, `auto_refresh`, `refresh_per_second`, `transient`, `redirect_stdout`, and `redirect_stderr`.

### Project Conventions

- Standards referenced: `.github/instructions/general-code-change.instructions.md`, `.github/instructions/python-code-change.instructions.md`, `.github/instructions/general-unit-test.instructions.md`, `.github/instructions/python-unit-test.instructions.md`
- Instructions followed: Research-only mode constraints (write only to `artifacts/research/`; document only verified findings from tool output)

## Key Discoveries

### Project Structure

- `fix_all.py` already *centralizes* step boundaries and labels via `step_name` (e.g., `Black: format`, `Pyright: type-check`). This is an ideal seam for:
  - driving per-branch status updates, and
  - reporting failures in a deterministic, testable way.

- Branch output is already buffered per branch (`StringIO`), which significantly reduces the complexity of terminal UI:
  - while the status board is active, the only thing that should write to the terminal is the board renderer;
  - per-branch command output is still preserved and printed after all branches complete.

### Implementation Patterns

- `runner_factory` already makes `fix_all.py` testable with deterministic fake runners.
- `shell_qc.py` uses “exit code 0 + informational stdout” to represent skipped shell tests; a live status board needs to be able to represent “SKIP tests” even if the step succeeds.

### Complete Examples

```python
from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from typing import Final


CSI: Final[str] = "\x1b["
EL_CLEAR: Final[str] = f"{CSI}2K"  # erase entire line


@dataclass
class BranchStatus:
    name: str
    message: str


def render_status_board(lines: list[BranchStatus]) -> str:
    """Render a fixed-height status board as plain text lines.

    This function intentionally renders without any side effects so it can be
    unit-tested independently of terminal capabilities.
    """

    # Keep output stable and predictable: one line per branch.
    return "\n".join(f"{line.name:<11} {line.message}" for line in lines)


def write_status_board_over_previous(
    *,
    stream: object,
    board_text: str,
    num_lines: int,
) -> None:
    """Rewrite an N-line status board in-place using VT sequences.

    Requires VT processing support. The caller is responsible for ensuring the
    stream is a TTY and VT processing is enabled.
    """

    # Move the cursor back up over the previous board so we can rewrite it.
    if num_lines:
        sys.stdout.write(f"{CSI}{num_lines}A")

    for line in board_text.splitlines():
        sys.stdout.write(f"{EL_CLEAR}\r{line}\n")
    sys.stdout.flush()

```

### API and Schema Documentation

- Windows VT support (Microsoft Learn):
  - VT sequences only work when `ENABLE_VIRTUAL_TERMINAL_PROCESSING` is set on the console output handle via `SetConsoleMode`.
  - Output mode should include `ENABLE_PROCESSED_OUTPUT` when using control sequences.

### Configuration Examples

```text
No new configuration files are required for the recommended approach.
The feature is expected to be driven via CLI flags (e.g., --complete-all).
```

### Technical Requirements

- Must provide a live “status board” during execution in interactive terminals (Windows Terminal / PowerShell).
- Must degrade gracefully to non-interactive output (CI logs) without spamming redraw artifacts.
- Must preserve existing buffered per-branch logs and final summary.
- Must change default semantics to fail-fast, with `--complete-all` restoring current behavior.

## Recommended Approach

Implement a lightweight, dependency-free status board renderer backed by:

1) **A shared in-memory status model** updated by each branch at step boundaries.
   - Each branch writes status updates (e.g., “Black (format)”, “Ruff (lint)”, “PASS/FAIL”) into a shared map protected by a lock.
   - This keeps the UI logic independent of subprocess execution and makes it unit-testable.

2) **A dedicated renderer loop** that:
   - uses in-place VT redraw *only when* output is interactive and VT mode is available/enabled,
   - otherwise prints simple, line-oriented status transitions (CI-safe).

3) **Windows VT enablement (best-effort)**:
   - Attempt to enable `ENABLE_VIRTUAL_TERMINAL_PROCESSING` on Windows by calling `GetStdHandle` / `GetConsoleMode` / `SetConsoleMode` via `ctypes`.
   - If VT can’t be enabled (or handles are redirected), fall back to non-interactive status printing.

4) **Fail-fast coordination**:
   - Introduce a shared `threading.Event` cancel signal.
   - On first failure (unless `--complete-all`), set the cancel signal.
   - Each branch checks the cancel signal before starting the next step and aborts early if set, yielding an “aborted” result.
   - This matches the requirement that abort may be enforced at step boundaries.

This approach is favored because it meets the “no heavy dependency” constraint in the feature docs, and the existing design (buffered logs + step boundaries) makes VT redraw feasible without garbling subprocess output.

Rejected alternatives (brief, non-exhaustive):

- Using Rich `Live` display
  - Pros: robust cross-platform terminal handling, stdout/stderr redirection, rich rendering.
  - Rejected for now because `rich` is not a current dependency (would require changing Poetry dependencies) and the feature spec explicitly prefers avoiding a new terminal UI dependency unless strictly necessary.

## Implementation Guidance

- **Objectives**:
  - Add live status visibility during parallel execution.
  - Default to fail-fast with `--complete-all` to restore run-to-completion.
  - Preserve final per-branch logs and summary output.

- **Key Tasks**:
  - Add a shared status model + renderer loop.
  - Add Windows VT enablement logic (best-effort) and robust fallback.
  - Add fail-fast cancellation signaling and update branch runners to honor it at step boundaries.
  - Add/extend unit tests using the existing `runner_factory` seam.

- **Dependencies**:
  - No new runtime dependencies recommended.
  - Use standard library only (`threading`, `ctypes` on Windows).

- **Success Criteria**:
  - Feature acceptance criteria in `docs/features/active/2026-01-14-fix-all-updates-84/issue.md` and `user-story.md` are satisfied.
  - Status output remains readable in CI.
  - No regression to buffered per-branch logs and final “Branch Results” summary.
