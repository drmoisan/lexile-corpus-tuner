# 2026-01-14-fix-all-updates — Spec

- Issue: #84
- Owner: drmoisan
- Last Updated: 2026-01-14

## Overview

The `scripts/dev_tools/fix_all.py` workflow now runs branches in parallel, but it behaves like a black box: it produces little/no feedback until all branches complete.

This hurts developer experience (especially when a long-running step is stuck or slow), and it also makes it hard to know which branch/step failed until the very end.

Additionally, the current behavior runs all branches to completion even if one branch fails; for many day-to-day workflows we want a fast “fail-fast” mode that aborts other work when any branch fails.

This spec defines a lightweight (standard-library-only) implementation that:

- Preserves the existing per-branch buffered logs and final summary output.
- Adds a live status board for interactive terminals.
- Defaults to fail-fast execution semantics, with an opt-in `--complete-all` flag to restore run-to-completion.


## Behavior

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

### Status board behavior (interactive vs. non-interactive)

- Interactive terminals (local dev):
	- Render a fixed-height board with one line per branch.
	- Update the right-side status at least at step boundaries.
	- Prefer in-place redraw (cursor movement + erase line) so output does not “scroll spam”.
	- The live board must not corrupt subsequent terminal output (e.g., avoid leaving the cursor hidden).

- Non-interactive output (CI logs / redirected output):
	- Do not attempt cursor movement redraw.
	- Emit line-oriented “status transitions” at step boundaries (e.g., `python -> Pyright (type-check)`), keeping logs readable.

### Windows console support

- On Windows, VT/ANSI escape sequences require enabling `ENABLE_VIRTUAL_TERMINAL_PROCESSING` on the console output handle.
- The implementation should attempt best-effort enablement on Windows.
- If VT enablement is not possible (e.g., output handle is redirected), fall back to line-oriented status transitions.

### Fail-fast semantics

- Default behavior is fail-fast:
	- On the first branch failure, signal other branches to abort as soon as practical.
	- Minimum acceptable enforcement is step-boundary cancellation: do not start the next step after cancellation is requested.

- `--complete-all` disables fail-fast:
	- All branches run to completion even if another branch fails.

### Shell “SKIP tests” behavior

The shell branch currently runs a “Shell: test” step, but `scripts/dev_tools/shell_qc.py` may return exit code `0` while printing that tests were skipped (no test dirs or `bats` missing). The status board should surface this as:

- `shell: SKIP tests` (or equivalent) rather than `shell: Shell: test` → `PASS` with no context.


## Inputs / Outputs

### Inputs

- CLI flags (existing):
	- `--max-ruff-retries <int>`
	- `--max-black-retries <int>`
	- `--no-coverage`

- CLI flags (new):
	- `--complete-all`
		- When set, branches continue running even if another branch fails.
		- When omitted, default behavior is fail-fast.

- Files
	- None introduced by this feature. `fix_all.py` continues to operate over the repository working tree.

- Environment variables
	- None introduced by this feature.
	- Runtime behavior may depend on whether stdout is a TTY (interactive) vs redirected (CI log).

### Outputs

- Live output (during execution):
	- Interactive: a live-updating per-branch status board.
	- Non-interactive: line-oriented status transitions at step boundaries.

- End-of-run output (unchanged shape, but must remain accurate):
	- Per-branch buffered logs (printed after completion).
	- Final “Branch Results” summary showing PASS/FAIL and failed step when applicable.

- Artifacts
	- No new artifact files.

## API / CLI Surface

This feature updates the CLI surface of:

- `poetry run python -m scripts.dev_tools.fix_all`
- `poetry run dev.fix-all` (Poetry script alias)

### Flags

- `--complete-all`
	- Description: Run all branches to completion even if a branch fails.
	- Default: `False` (fail-fast).

- `--max-ruff-retries <int>` (existing)
- `--max-black-retries <int>` (existing)
- `--no-coverage` (existing)

### Examples

- Default (fail-fast):
	- `poetry run python -m scripts.dev_tools.fix_all`

- Run-to-completion (collect all failures):
	- `poetry run python -m scripts.dev_tools.fix_all --complete-all`

- Skip coverage (faster local runs):
	- `poetry run python -m scripts.dev_tools.fix_all --no-coverage`

## Data & State

This feature introduces only in-memory state during execution:

- Shared status model
	- A per-branch status string/state updated at step boundaries.
	- Protected by a lock to avoid races between worker threads and renderer.

- Cancel signal for fail-fast
	- A shared `threading.Event` (or equivalent) set on first failure.
	- Each branch checks the signal before starting the next step.

- Renderer loop state
	- Tracks whether VT redraw is enabled and the last board height rendered.

No state is persisted to disk. No caches or DBs are introduced.

## Constraints & Risks

- **Terminal compatibility:** The updater must behave well on Windows terminals (PowerShell/Windows Terminal) and within CI logs (where interactive updating may not be supported). A non-interactive fallback may be needed.
- **Concurrency & output interleaving:** Branch logs are currently captured to per-branch buffers and printed at the end; live status updates should avoid garbling output, especially if some commands also emit output.
- **Abort semantics:** Some steps invoke external processes; “abort” may only be enforceable at step boundaries unless subprocess cancellation/termination is added. Define the minimum acceptable abort behavior (e.g., don’t start the next step after abort requested).
- **Dependencies:** Prefer not to introduce new runtime dependencies for terminal UI unless there is a strong reason (keep `fix_all.py` lightweight and reliable).
- **CI behavior:** In CI, frequent terminal redraws can make logs noisy; ensure output remains readable and useful.

Additional constraints/risks derived from research:

- **VT enablement on Windows:** VT sequences require enabling `ENABLE_VIRTUAL_TERMINAL_PROCESSING`. This may fail if stdout is redirected or if the environment doesn’t support it.
	- Mitigation: best-effort enablement; fall back to non-interactive status transitions.

- **Console state hygiene:** Terminal mode changes or cursor control must not leave the console “broken” after the run.
	- Mitigation: keep cursor visibility unchanged (avoid hiding cursor), and if console modes are modified, restore them on exit when practical.

- **Shell tests “skipped” ambiguity:** A successful shell test step may actually represent “skipped” rather than “executed.”
	- Mitigation: detect the skip message pattern in the shell branch output and reflect it in status.


## Definition of Done

- [ ] Behavior matches acceptance criteria, including:
	- live per-branch status updates at step boundaries
	- default fail-fast semantics
	- `--complete-all` run-to-completion behavior
	- shell “SKIP tests” surfaced when appropriate
- [ ] Unit tests updated/added (Pytest) for:
	- fail-fast cancellation behavior across branches
	- `--complete-all` disabling cancellation
	- deterministic status state transitions (pure rendering/state model)
	- shell “SKIP tests” detection
- [ ] CLI help output documents `--complete-all`.
- [ ] No new heavy runtime dependencies introduced for terminal UI.
- [ ] Docs updated if any user-facing CLI behavior changes beyond `--complete-all`.
- [ ] Telemetry/logging: not applicable (no new telemetry introduced).

## Seeded Test Conditions (from potential)
- [ ] Unit: status state machine per branch (initial -> step transitions -> PASS/FAIL) is deterministic.
- [ ] Unit: fail-fast coordination signals other branches to stop before starting subsequent steps.
- [ ] Unit: `--complete-all` disables fail-fast coordination and allows other branches to continue.
- [ ] Unit: `shell` branch correctly reports “SKIP tests” when tests are not configured/available.
- [ ] Integration: running `scripts.dev_tools.fix_all` (or the underlying module invocation) shows live status updates and still prints full per-branch logs at the end.
- [ ] CLI: `poetry run python -m scripts.dev_tools.fix_all --help` includes `--complete-all`.
- [ ] CLI: Example behavior when Python branch fails at `Pyright: type-check` still produces final summary with failed step name.
