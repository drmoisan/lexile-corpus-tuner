# fix-all-updates (Issue #84)

- Date captured: 2026-01-14
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/fix-all-updates/ (Issue #84)

- Issue: #84
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/84
- Last Updated: 2026-01-14
## Problem / Why

The `scripts/dev_tools/fix_all.py` workflow now runs branches in parallel, but it behaves like a black box: it produces little/no feedback until all branches complete.

This hurts developer experience (especially when a long-running step is stuck or slow), and it also makes it hard to know which branch/step failed until the very end.

Additionally, the current behavior runs all branches to completion even if one branch fails; for many day-to-day workflows we want a fast “fail-fast” mode that aborts other work when any branch fails.

## Proposed Behavior

Add a live, terminal-friendly “status board” (or progress updater) to `fix_all.py` that continuously reports each branch’s current step/state while work is in progress.

The status board should:

- Show each branch on its own line, with the right-side status changing as the branch progresses.
- Replace the current end-only feedback loop with real-time updates, without losing the full per-branch logs currently printed at the end.

Default execution semantics should change to **fail-fast**:

- If any branch fails, the remaining branches should be signaled to abort as soon as practical (before starting the next step, or sooner if feasible).
- Add a flag `--complete-all` to restore the existing behavior (all branches run to completion regardless of failures).

Step model per branch:

- `json` branch: formatting -> validation (“linting”) -> PASS/FAIL
- `shell` branch: formatting -> linting -> (optional) testing -> PASS/FAIL (tests may be skipped today if none exist)
- `python` branch: formatting -> linting (+ optional auto-fix loop) -> type checking -> testing -> PASS/FAIL
- `powershell` branch: formatting -> linting -> testing -> PASS/FAIL

End-of-run output should still include the final summary block, e.g.

```
========== Branch Results ==========
Branch json: PASS
Branch shell: PASS
Branch python: FAIL (failed at Pyright: type-check)
Branch powershell: PASS
====================================
```

## Acceptance Criteria (early draft)

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

## Constraints & Risks

- **Terminal compatibility:** The updater must behave well on Windows terminals (PowerShell/Windows Terminal) and within CI logs (where interactive updating may not be supported). A non-interactive fallback may be needed.
- **Concurrency & output interleaving:** Branch logs are currently captured to per-branch buffers and printed at the end; live status updates should avoid garbling output, especially if some commands also emit output.
- **Abort semantics:** Some steps invoke external processes; “abort” may only be enforceable at step boundaries unless subprocess cancellation/termination is added. Define the minimum acceptable abort behavior (e.g., don’t start the next step after abort requested).
- **Dependencies:** Prefer not to introduce new runtime dependencies for terminal UI unless there is a strong reason (keep `fix_all.py` lightweight and reliable).
- **CI behavior:** In CI, frequent terminal redraws can make logs noisy; ensure output remains readable and useful.

## Test Conditions to Consider

- [ ] Unit: status state machine per branch (initial -> step transitions -> PASS/FAIL) is deterministic.
- [ ] Unit: fail-fast coordination signals other branches to stop before starting subsequent steps.
- [ ] Unit: `--complete-all` disables fail-fast coordination and allows other branches to continue.
- [ ] Unit: `shell` branch correctly reports “SKIP tests” when tests are not configured/available.
- [ ] Integration: running `scripts.dev_tools.fix_all` (or the underlying module invocation) shows live status updates and still prints full per-branch logs at the end.
- [ ] CLI: `poetry run python -m scripts.dev_tools.fix_all --help` includes `--complete-all`.
- [ ] CLI: Example behavior when Python branch fails at `Pyright: type-check` still produces final summary with failed step name.

## Next Step

- [ ] Promote to GitHub issue (feature request template)
- [ ] Create `docs/features/active/fix-all-updates/` folder from the template
