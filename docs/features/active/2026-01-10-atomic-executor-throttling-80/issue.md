# atomic-executor-throttling (Issue #80)

- Date captured: 2026-01-10
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/atomic-executor-throttling/ (Issue #80)

> Automation note: Keep the section headings below unchanged; the promotion tooling maps each of them into the GitHub bug issue template.

- Issue: #80
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/80
- Last Updated: 2026-01-10
## Summary

`atomic_executor` intermittently gets throttled while orchestrating atomic tasks via GitHub Copilot CLI.

When throttling occurs, the executor either fails the current atomic task or becomes unreliable due to repeated CLI invocation failures, even when overall token usage is not excessive.

## Environment

- OS/version: Windows (version not captured)
- Python version: Python 3.x (exact version not captured; project supports 3.10+)
- Command/flags used: `poetry run python -m scripts.dev_tools.atomic_executor.cli execute-all --path <feature-folder> --preferred-model <model>`
- Data source or fixture: N/A (throttling is triggered by CLI call frequency, not by a specific corpus fixture)

## Steps to Reproduce

1. Run `atomic_executor` in a mode that prompts one atomic task at a time (or `execute-all`) such that it makes frequent GitHub Copilot CLI calls (multiple calls per minute).
2. Ensure the executor reaches a sequence of tasks that require repeated “prompt -> validate -> prompt next task” cycles.
3. After enough calls in a short time window (varies by model and global load), GitHub Copilot CLI returns a throttling / rate-limit style failure.

Important:
- Do **not** attempt to reproduce this in a way that causes real throttling during development sessions; the throttling behavior is load-dependent and can consume large token budgets.
- Prefer deterministic reproduction via a mocked CLI runner in unit tests that simulates throttling responses after $N$ calls per time window.

## Expected Behavior

When GitHub Copilot CLI throttles, `atomic_executor` should automatically self-regulate:

- Reduce the **number of GitHub Copilot CLI calls per time window** (call-rate based, not token-based).
- Retry the same atomic task prompt after an exponential backoff delay (with jitter).
- Preserve strict “one atomic task at a time” ordering and never advance to the next task until the current one is completed and validated.

## Actual Behavior

The executor experiences repeated CLI call failures that are consistent with throttling/rate limiting (exact error text varies).

Observed failure modes include:

- Immediate failure of the current atomic task due to a CLI error.
- Retry loops without adequate spacing, increasing the chance of continued throttling.
- Loss of productivity because the only reliable workaround is manual waiting / reruns.

## Logs / Screenshots

- [ ] Attached minimal logs or screenshot
- Snippet: (not attached here to avoid encouraging reproduction that causes throttling; capture minimal CLI stderr text when it happens organically)

## Impact / Severity

- [ ] Blocker
- [ ] High
- [ ] Medium
- [ ] Low

## Suspected Cause / Notes

This is not primarily a “token usage” issue. The trigger is the **rate of GitHub Copilot CLI invocations**.

GitHub Copilot service throttling appears to vary by:

- Selected model
- Overall platform load
- Call cadence (many short calls vs fewer longer calls)

`atomic_executor` currently does not implement a robust call-rate limiter or exponential backoff policy that adapts when throttling is observed.

## Proposed Fix / Validation Ideas

- [ ] Implement a call-rate limiter (token bucket or leaky bucket) keyed to “GitHub Copilot CLI calls per time window”
	- Start with a conservative default, e.g. max calls per minute (configurable)
	- Gate every CLI invocation behind the limiter
- [ ] Add exponential backoff on detected throttling
	- Backoff should be per executor run (shared state), not per single function call
	- Use exponential backoff with full jitter; cap to a max delay
	- Reset/decay backoff after a stable period without throttling
- [ ] Detect throttling robustly
	- Treat CLI exit codes and stderr patterns as signals (e.g., “rate limit”, “throttle”, HTTP 429/503 equivalents)
	- Centralize detection logic so it’s testable
- [ ] Configuration surface (defaults must be safe)
	- `--copilot-cli-max-calls-per-window` (or config file setting)
	- `--copilot-cli-window-seconds`
	- `--copilot-cli-backoff-base-seconds`
	- `--copilot-cli-backoff-max-seconds`
	- `--copilot-cli-max-retries` (optional)
- [ ] Unit tests (must be deterministic and must not call the real CLI)
	- Use a fake CLI runner that simulates success for the first $N$ calls, then throttling, then recovery
	- Assert the executor sleeps (or schedules) increasing delays and does not advance tasks out of order
- [ ] Manual verification (non-destructive)
	- Run a small set of atomic tasks at a known low call rate and confirm no regressions
	- If throttling occurs organically, confirm the executor recovers without user intervention

## Next Step

- [ ] Promote to GitHub issue (bug-report template)
- [ ] Move to active fix folder / branch