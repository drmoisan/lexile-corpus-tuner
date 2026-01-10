# 2026-01-10-atomic-executor-throttling (Spec)

- Issue: #80
- Owner: 2026-01-10-atomic-executor-throttling
- Date: 2026-01-10
- Status: Draft

## Context
`atomic_executor` intermittently gets throttled while orchestrating atomic tasks via GitHub Copilot CLI.

When throttling occurs, the executor either fails the current atomic task or becomes unreliable due to repeated CLI invocation failures, even when overall token usage is not excessive.

Environment:
- OS/version: Windows (version not captured)
- Python version: Python 3.x (exact version not captured; project supports 3.10+)
- Command/flags used: `poetry run python -m scripts.dev_tools.atomic_executor.cli execute-all --path <feature-folder> --preferred-model <model>`
- Data source or fixture: N/A (throttling is triggered by CLI call frequency, not by a specific corpus fixture)

Notes:
- GitHub Docs describe Copilot CLI native Windows PowerShell support as experimental; Windows via WSL is listed as supported.
	- This may contribute to instability, but the primary trigger remains call cadence.

Impact / Severity:
- [ ] Blocker
- [x] High
- [ ] Medium
- [ ] Low


## Repro & Evidence
Steps to Reproduce:
1. Run `atomic_executor` in a mode that prompts one atomic task at a time (or `execute-all`) such that it makes frequent GitHub Copilot CLI calls (multiple calls per minute).
2. Ensure the executor reaches a sequence of tasks that require repeated “prompt -> validate -> prompt next task” cycles.
3. After enough calls in a short time window (varies by model and global load), GitHub Copilot CLI returns a throttling / rate-limit style failure.

Important:
- Do **not** attempt to reproduce this in a way that causes real throttling during development sessions; the throttling behavior is load-dependent and can consume large token budgets.
- Prefer deterministic reproduction via a mocked CLI runner in unit tests that simulates throttling responses after $N$ calls per time window.

Expected:
When GitHub Copilot CLI throttles, `atomic_executor` should automatically self-regulate:

- Reduce the **number of GitHub Copilot CLI calls per time window** (call-rate based, not token-based).
- Retry the same atomic task prompt after an exponential backoff delay (with jitter).
- Preserve strict “one atomic task at a time” ordering and never advance to the next task until the current one is completed and validated.

Actual:
The executor experiences repeated CLI call failures that are consistent with throttling/rate limiting (exact error text varies).

Observed failure modes include:

- Immediate failure of the current atomic task due to a CLI error.
- Retry loops without adequate spacing, increasing the chance of continued throttling.
- Loss of productivity because the only reliable workaround is manual waiting / reruns.

Logs / Screenshots:
- [ ] Attached minimal logs or screenshot
- Snippet: (not attached here to avoid encouraging reproduction that causes throttling; capture minimal CLI stderr text when it happens organically)


## Scope & Non-Goals
- In scope:
	- Add a **call-rate-based** self-regulation mechanism to `atomic_executor` for GitHub Copilot CLI invocations.
	- Add exponential backoff behavior when throttling is detected.
	- Ensure the executor maintains strict ordering (never advance to the next atomic task until the current task is completed + validated).
	- Add deterministic unit tests that simulate throttling without calling the real GitHub Copilot CLI.

- Out of scope / non-goals:
	- Any approach that intentionally triggers real throttling in tests/CI (explicitly prohibited).
	- Token-usage-based throttling heuristics (this bug is about **call count per time window**).
	- Changing the atomic task ordering semantics (the “one atomic task at a time” flow remains).
	- Attempting to bypass or evade throttling beyond respectful rate limiting and retries.

## Root Cause Analysis
This is not primarily a “token usage” issue. The trigger is the **rate of GitHub Copilot CLI invocations**.

GitHub Copilot service throttling appears to vary by:

- Selected model
- Overall platform load
- Call cadence (many short calls vs fewer longer calls)

`atomic_executor` currently does not implement a robust call-rate limiter or exponential backoff policy that adapts when throttling is observed.

Verified implementation gaps (from repo inspection):
- Copilot invocation failures abort the run immediately.
	- `run_copilot()` raises `subprocess.CalledProcessError` when Copilot exits non-zero.
	- `_execute_one_task()` retries only when scoped QC fails, and does not catch Copilot failures.
	- Result: any Copilot CLI non-zero exit (including rate limiting) ends the executor run rather than spacing/retrying.
- Throttling detection is not currently implementable using `CalledProcessError.stdout/stderr`.
	- Copilot output is streamed to console + appended to a log file.
	- The raised exception does not include captured output text, so robust throttle classification requires additional capture (in-memory tail) and/or log tail parsing.

External confirmation (GitHub Docs):
- GitHub explicitly states Copilot features can experience rate limits and that frequent/automated requests can trigger rate limiting.
- GitHub does not publish stable numeric thresholds; “wait and try again” is the recommended mitigation.


## Proposed Fix
- [ ] Implement a call-rate limiter (token bucket or leaky bucket) keyed to “GitHub Copilot CLI calls per time window”
	- Start with a conservative default, e.g. max calls per minute (configurable)
	- Gate every CLI invocation behind the limiter
- [ ] Add exponential backoff on detected throttling
	- Backoff should be per executor run (shared state), not per single function call
	- Use exponential backoff with full jitter; cap to a max delay
	- Reset/decay backoff after a stable period without throttling
- [ ] Detect throttling robustly
	- Treat Copilot CLI failures as candidates for classification rather than immediate fatal errors.
	- Classify throttling using:
		- Exit code (non-zero) combined with output text patterns (e.g., “rate limit”, “rate limited”, “throttle”, HTTP 429/503 equivalents when present).
		- A small in-memory “output tail” captured during streaming, and/or parsing the tail of the per-task log file (since output is not otherwise captured).
	- Centralize detection logic so it’s unit-testable.
- [ ] Configuration surface (defaults must be safe)
	- `--copilot-cli-max-calls-per-window` (or config file setting)
	- `--copilot-cli-window-seconds`
	- `--copilot-cli-backoff-base-seconds`
	- `--copilot-cli-backoff-max-seconds`
	- `--copilot-cli-output-tail-bytes` (or equivalent) for deterministic throttle classification without full output capture
	- `--copilot-cli-max-retries` (optional)
- [ ] Unit tests (must be deterministic and must not call the real CLI)
	- Use a fake CLI runner that simulates success for the first $N$ calls, then throttling, then recovery
	- Assert the executor sleeps (or schedules) increasing delays and does not advance tasks out of order
- [ ] Manual verification (non-destructive)
	- Run a small set of atomic tasks at a known low call rate and confirm no regressions
	- If throttling occurs organically, confirm the executor recovers without user intervention

Implementation notes (design-level):
- The limiter should be based on “calls per window”, implemented as a small in-memory queue of recent call timestamps.
  - If the queue already contains `max_calls` entries in the last `window_seconds`, the executor waits until the oldest entry exits the window.
- Exponential backoff should be applied **on top of** the base limiter when throttling is detected.
  - Backoff delay schedule: `min(backoff_max, backoff_base * 2^k)` with full jitter, where `k` increments per throttle event.
  - Backoff should decay back toward the base limiter after a sustained “no throttle” streak.
- Both the limiter and backoff should be centralized inside the GitHub Copilot CLI invocation wrapper used by `atomic_executor`.
	- This wrapper must also retain enough output context (in-memory tail and/or log tail parsing) to make throttle classification deterministic.


## Assumptions, Constraints, Dependencies
- Assumptions (environment, data, access):
	- There are no stable/published GitHub Copilot service rate limits; throttling thresholds vary by model and platform load.
	- Throttling can be inferred from GitHub Copilot CLI failures via exit code and/or stderr text patterns.
	- The executor must remain usable offline (tests must not require network access or a GitHub auth session).

- Constraints (budget, performance, compatibility):
	- Defaults must be conservative to reduce throttling frequency without requiring user tuning.
	- Retrying must not cause infinite loops; retries should be capped or be clearly controllable by configuration.
	- Must preserve current behavior for users who are not being throttled (no regressions, only controlled pacing).
	- Unit tests must be deterministic and must not intentionally trigger real throttling or make any network calls.

- External dependencies (services, libraries, releases):
	- GitHub Copilot CLI (external binary).
	- No new third-party Python dependencies are required; implement with stdlib.

## Data / API / Config Impact
- User-facing or API changes:
	- Add CLI flags (and config-file equivalents if the executor already supports config) to control call rate limiting and backoff.
	- Defaults should be safe; users can opt to loosen/tighten depending on environment.
	- Document environment nuance: Copilot CLI on native Windows PowerShell is experimental per GitHub Docs; WSL is the recommended Windows path when available.

- Data or migration considerations:
	- None (in-memory pacing only; no persisted state required).

- Logging/telemetry updates (if any):
	- Add structured log lines when:
		- The call-rate limiter delays a call (include delay duration and limiter settings).
		- Throttling is detected (include detection signal and backoff state).
		- A retry is scheduled (include attempt count and next delay).
		- A Copilot CLI failure is classified as “non-throttle” and the run aborts (include exit code + minimal captured output tail).

## Test Strategy
- [ ] Unit: throttling detector
	- Given representative output-tail samples + non-zero exit codes, confirm the detector classifies “throttle” vs “non-throttle” failures.
	- Include negative cases (random errors, tool permission errors, auth failures) to avoid false positives.

- [ ] Unit: output-tail capture behavior
	- Ensure the Copilot runner retains a deterministic amount of recent output (bytes/lines) during streaming.
	- Ensure throttle classification uses only the captured tail (not full output) to keep memory bounded.

- [ ] Unit: call-rate limiter behavior (no real sleeping)
  - Inject a fake clock and fake sleeper.
  - Assert the limiter schedules exactly the required delay to stay within `max_calls/window_seconds`.

- [ ] Unit: exponential backoff state machine
  - Given a sequence of “throttle detected” events, confirm backoff grows exponentially up to the max.
  - Given a recovery period, confirm backoff decays/resets as specified.

- [ ] Unit: executor ordering invariants under throttle
  - Use a fake CLI runner that returns success, then throttle, then success.
  - Assert the executor does not mark the task complete or advance to the next task until the throttled call eventually succeeds.
	- Assert that retries happen within the same task loop (no checkbox flipping, no `next_unchecked_task()` advance).

- [ ] Unit: bounded retries on repeated throttle
	- Use a fake CLI runner that always returns throttle.
	- Assert the executor eventually exits non-zero after `--copilot-cli-max-retries` (or equivalent) attempts.

- [ ] Manual verification (non-destructive)
  - Run a small feature folder that triggers only a few CLI calls (well below any likely limit) and confirm:
    - No ordering changes
    - No unexpected sleeps
    - No user-visible regressions


## Acceptance Criteria
- [ ] `atomic_executor` enforces a configurable maximum number of GitHub Copilot CLI calls per time window.
- [ ] When throttling is detected, `atomic_executor` retries the same Copilot CLI invocation with exponential backoff (with jitter) rather than failing the run immediately.
- [ ] Backoff and rate limiting are based on **call frequency**, not token usage.
- [ ] The executor never advances to the next atomic task until the current task is completed and validated, even across retries.
- [ ] Copilot CLI failures are classified as either “throttle” (retryable) or “non-throttle” (fail-fast), and the classification logic is unit-tested with positive and negative samples.
- [ ] Throttling detection does not rely on `CalledProcessError.stdout/stderr` being populated; detection works using captured output tail and/or log tail parsing.
- [ ] Retry behavior is bounded by configuration (no infinite loops by default).
- [ ] Unit tests deterministically simulate throttling and recovery without calling the real GitHub Copilot CLI and without any network access.
- [ ] User-facing configuration is documented via CLI help and/or config schema (if applicable), including safe defaults and a note that Copilot CLI native Windows support is experimental per GitHub Docs.

## Risks & Mitigations
- Technical or operational risks:
	- False positives in throttling detection could slow runs unnecessarily.
	- Overly conservative defaults could make long runs feel slower.
	- Underly conservative defaults may still throttle under heavy global load.
	- Adding output capture could increase memory usage if unbounded.

- Mitigations and rollbacks:
	- Keep detection logic centralized and test it against multiple real-world samples.
	- Make limiter/backoff parameters configurable with documented safe defaults.
	- Allow users to disable the limiter/backoff (not recommended) for troubleshooting.
	- Keep output capture bounded (tail only) and include tests that prove bounded behavior.

## Rollout & Follow-up
- Release/rollout steps:
	- Implement and land with unit tests.
	- Update any CLI docs/help text to reflect new throttling controls.

- Post-fix monitoring or clean-up tasks:
	- Capture a small set of anonymized throttle error samples (stderr snippets) to improve detection robustness over time.

- Links: issue, PRs, related docs
	- Issue: #80
	- Potential (promoted) doc: `docs/features/potential/promoted/2026-01-10-atomic-executor-throttling.md`
