# `2026-01-14-fix-all-updates` — User Story

- Issue: #84
- Owner: drmoisan
- Status: Draft | In Progress | Complete
- Last Updated: 2026-01-14

## Story Statement

- As a developer running `fix-all`, I want to see real-time status for each parallel branch (json/shell/python/powershell), so that I can tell what is happening and whether the run is stuck.
- As a developer iterating quickly, I want `fix-all` to stop other work when any branch fails by default, so that I get faster feedback and don’t waste time waiting for unrelated branches to finish.
- As a developer or CI maintainer, I want an opt-in flag (`--complete-all`) to allow all branches to run to completion even if another branch fails, so that I can collect comprehensive failure signals in a single run when needed.

## Problem / Why

The `scripts/dev_tools/fix_all.py` workflow now runs branches in parallel, but it behaves like a black box: it produces little/no feedback until all branches complete.

This hurts developer experience (especially when a long-running step is stuck or slow), and it also makes it hard to know which branch/step failed until the very end.

Additionally, the current behavior runs all branches to completion even if one branch fails; for many day-to-day workflows we want a fast “fail-fast” mode that aborts other work when any branch fails.


## Personas & Scenarios

- Persona: Repo contributor (Windows-first)
  - Who: A developer contributing changes locally on Windows using PowerShell / Windows Terminal.
  - Cares about: Fast feedback loops, visibility into “what’s running”, and actionable failures.
  - Constraints: Toolchain steps can be slow (Pytest + coverage, PowerShell Pester, etc.). Parallel output can get noisy or interleaved.
  - Goals/frustrations: Wants to avoid staring at a blank terminal, and wants to know which branch is currently executing which step.

- Persona: CI / quality gate maintainer
  - Who: Maintains the repo’s quality signals and wants clean logs.
  - Cares about: Deterministic outcomes, readable CI logs, and minimizing wasted compute.
  - Constraints: CI logs may not support interactive redraw; excessive redraw output can be noisy.

- Scenario: Local “Fix All” run with a Python failure
  - Trigger: Developer runs `poetry run python -m scripts.dev_tools.fix_all` after making changes.
  - Steps:
    - The command starts all branches in parallel.
    - The terminal immediately shows a per-branch status line (e.g., json: formatting, python: Black (format), powershell: PoshQC format).
    - As each branch advances, the right-side status changes to reflect the current step (e.g., python: Ruff (lint) → Pyright (type-check)).
    - Pyright fails in the python branch.
    - By default (fail-fast), the other branches are signaled to abort as soon as practical (at latest before starting their next step).
  - Outcome:
    - The terminal prints the existing final summary block with the failed step name.
    - The developer can immediately see where the failure occurred and doesn’t wait for unrelated work to finish.

- Scenario: “Complete all” run to gather all failures
  - Trigger: Developer runs `... fix_all --complete-all` to see all issues in one run.
  - Steps:
    - Branches run in parallel with live status updates.
    - A failure in one branch does not stop the others.
  - Outcome:
    - Final summary shows pass/fail for every branch in the same run, plus per-branch logs.


## Acceptance Criteria

- [ ] While `fix_all.py` is running, the terminal shows a live-updating status line per branch (`json`, `shell`, `python`, `powershell`) instead of appearing idle until completion.
- [ ] Each branch’s displayed status updates at least at step boundaries (e.g., `json` shows “formatting”, then “validation”, then “PASS/FAIL”).
- [ ] `python` branch status reflects the toolchain phases: “Black (format)”, “Ruff (lint)”, “Ruff (fix)” when auto-fix is running, “Pyright (type-check)”, “Pytest (test[/coverage])”, then “PASS/FAIL”.
- [ ] `powershell` branch status reflects: “PoshQC format”, “PoshQC analyze”, “PoshQC test”, then “PASS/FAIL”.
- [ ] `shell` branch status reflects: “shell format”, “shell check”, optional “shell test” (or “SKIP tests”), then “PASS/FAIL”.
- [ ] The final summary output remains present and accurate (including the failed step name when applicable).
- [ ] Default behavior is **fail-fast**: when any branch fails, remaining branches abort work as soon as practical and the overall exit code is non-zero.
- [ ] Passing `--complete-all` preserves the current behavior: all branches continue running even if another branch fails.
- [ ] The status updater works in common Windows terminals (PowerShell, Windows Terminal) without leaving the console in a broken state (cursor hidden, corrupted output, etc.).
- [ ] `fix_all.py --help` documents `--complete-all` and any other new flags introduced to support status display.


## Non-Goals

This feature intentionally does not include:

- Rewriting or reordering the underlying toolchain semantics beyond the agreed fail-fast default and `--complete-all` override.
- Per-command progress within a single step (e.g., parsing Pytest progress output). Step-level status updates are sufficient.
- Adding new branches beyond the existing json/shell/python/powershell branches.
- Introducing a heavy terminal UI dependency unless it is strictly necessary; prefer a lightweight, terminal-friendly approach.
- Changing which tools are run (Black/Ruff/Pyright/Pytest/PoshQC/etc.) or their core invocation flags (except what is required to support the status updater and abort behavior).
