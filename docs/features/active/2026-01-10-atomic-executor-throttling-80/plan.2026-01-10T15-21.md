---
id: 2026-01-10-atomic-executor-throttling-80
status: Planned
status_color: blue
owner: drmoisan
last_updated: 2026-01-10
---

# 2026-01-10-atomic-executor-throttling-80 (Plan)

![status: planned](https://img.shields.io/badge/status-planned-blue)

- **Issue:** #80
- **Spec (authoritative):** `docs/features/active/2026-01-10-atomic-executor-throttling-80/spec.md`
- **Date:** 2026-01-10T15-21

## Implementation plan (atomic tasks)

### Phase 0 — Context & guardrails

**Plan-of-record:** This timestamped file (`plan.2026-01-10T15-21.md`) is authoritative for execution. Any references elsewhere to `plan.md` are not applicable for this feature folder.


- [ ] [P0-T1] Read repo Copilot instructions (policy order)
	- Files:
		- `.github/copilot-instructions.md`
	- Acceptance:
		- Confirm the repo’s policy order is understood and will be followed for this change.

- [ ] [P0-T2] Read general code-change policy (toolchain loop + bugfix workflow)
	- Files:
		- `.github/instructions/general-code-change.instructions.md`
	- Acceptance:
		- Confirm the required toolchain loop (format → lint → type-check → test) must be run until clean.

- [ ] [P0-T3] Read Python code-change policy (Black/Ruff/Pyright expectations)
	- Files:
		- `.github/instructions/python-code-change.instructions.md`
	- Acceptance:
		- Confirm strict typing + suppression rules are understood.

- [ ] [P0-T4] Read unit test policies (no temp files; deterministic; pytest)
	- Files:
		- `.github/instructions/general-unit-test.instructions.md`
		- `.github/instructions/python-unit-test.instructions.md`
	- Acceptance:
		- Confirm tests must not write temporary files and must avoid external dependencies.

- [ ] [P0-T5] Confirm authoritative docs and missing user story
	- Notes:
		- `spec.md` is authoritative.
		- No `user-story.md` exists for this feature folder (expected for this bug).
	- Acceptance:
		- Any “user story” references are treated as “not applicable” for this feature.

- [ ] [P0-T6] Branch hygiene: implement #80 on a dedicated branch
	- Acceptance:
		- Do not implement #80 on an unrelated feature branch (e.g., #73).
		- Work occurs on a dedicated branch for #80 so review/rollback is isolated.

- [ ] [P0-T7] Capture baseline toolchain status (record pass/fail)
	- Commands (repo standard):
		- `poetry run black .`
		- `poetry run ruff check`
		- `poetry run pyright`
		- `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`
		- `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/format-powershell.ps1`
		- `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/run-psscriptanalyzer.ps1`
		- `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/run-pester.ps1`
	- Acceptance:
		- Baseline is captured before changes (even if it’s already passing).

- [ ] [P0-T8] Pin spec and acceptance criteria
	- References:
		- `docs/features/active/2026-01-10-atomic-executor-throttling-80/spec.md`
	- Acceptance:
		- This plan explicitly covers every item in the spec's **Acceptance Criteria** section.

- [ ] [P0-T9] Identify current Copilot invocation seam(s)
	- Files:
		- `scripts/dev_tools/atomic_executor/cli.py`: `run_copilot()`, `_stream_copilot_output()`, `_execute_one_task()`
	- Acceptance:
		- Document (in plan notes or code comments during implementation) that `_execute_one_task()` currently retries only on QC failures, and does not handle Copilot non-zero exits.

- [ ] [P0-T10] Choose safe defaults and config surface (call-rate based)
	- Defaults (initial proposal, adjust only with justification in PR):
		- max calls per window: 6
		- window seconds: 60
		- backoff base seconds: 2
		- backoff max seconds: 60
		- output tail bytes: 4096
		- max throttle retries: 8
	- Acceptance:
		- Defaults are conservative and bounded (no infinite retry by default).
		- All settings are explicitly “calls per time window”, not token based.

### Phase 1 — Regression tests (fail first)

- [ ] [P1-T1] Add failing regression test: throttling does not abort run; retries instead
	- New test module:
		- `tests/scripts/dev_tools/atomic_executor/test_executor_throttle_retry_regression.py`
	- Technique (deterministic; no real Copilot CLI):
		- Simulate a Copilot outcome sequence: throttle → success.
		- Verify the executor retries within the same task, and does not advance ordering until the retry succeeds.
		- This may target the planned runner seam API (added later), so failure-before-fix can be due to missing seam and/or missing retry behavior.
	- Acceptance:
		- This test fails against the current code before implementing throttling retry support.
		- The test does not invoke the real `copilot` binary and makes no network calls.

- [ ] [P1-T2] Add unit tests for throttle classifier (positive + negative cases)
	- New test module (avoid extending large existing files):
		- `tests/scripts/dev_tools/atomic_executor/test_copilot_throttling_classifier.py`
	- Acceptance:
		- This test fails against the current code before implementing the classifier.
		- Tests are deterministic (no network, no subprocess, no filesystem writes).

- [ ] [P1-T3] Add unit tests for call-rate limiter scheduling (no real sleep)
	- New test module:
		- `tests/scripts/dev_tools/atomic_executor/test_copilot_rate_limiter.py`
	- Acceptance:
		- This test fails against the current code before implementing the limiter.
		- Uses a fake clock + fake sleeper and asserts scheduled sleep durations.

- [ ] [P1-T4] Add unit tests for exponential backoff growth + reset/decay
	- New test module:
		- `tests/scripts/dev_tools/atomic_executor/test_copilot_backoff.py`
	- Acceptance:
		- This test fails against the current code before implementing the backoff.
		- Uses deterministic `RandomSource`.
		- Verifies capped growth and reset/decay behavior.

### Phase 2 — Deterministic primitives (unit-testable)

- [ ] [P2-T1] Introduce an injectable Copilot runner seam (for deterministic tests)
	- Files (one of the following approaches; prefer the smallest change that stays typed):
		- Option A: `scripts/dev_tools/atomic_executor/cli.py` (add an injectable runner parameter to `run_copilot()`)
		- Option B: new adapter module `scripts/dev_tools/atomic_executor/copilot_runner.py` used by `run_copilot()`
	- Acceptance:
		- Tests can simulate Copilot outcomes (success/throttle/non-throttle) without invoking the real `copilot` binary.
		- The seam returns enough information to support throttling classification (at minimum: exit code + output tail).
		- The production implementation continues to stream output to console + log file as it does today.

- [ ] [P2-T2] Add a call-rate limiter abstraction (queue-of-timestamps)
	- Add new module (or equivalent cohesive location):
		- `scripts/dev_tools/atomic_executor/copilot_throttling.py`
	- Symbols (names may vary, but keep them explicit + typed):
		- `CallRateLimiter(max_calls: int, window_seconds: float, clock: Clock, sleeper: Sleeper)`
		- `Clock` protocol with `now_monotonic() -> float`
		- `Sleeper` protocol with `sleep(seconds: float) -> None`
	- Acceptance:
		- Limiter enforces at most `max_calls` in the last `window_seconds`.
		- Limiter uses injected `clock`/`sleeper` so tests avoid real sleeping.

- [ ] [P2-T3] Add exponential backoff policy with jitter + bounded cap
	- Location: `scripts/dev_tools/atomic_executor/copilot_throttling.py`
	- Symbols:
		- `ExponentialBackoff(base_seconds: float, max_seconds: float, random_source: RandomSource)`
		- `RandomSource` protocol with `random() -> float` (for deterministic jitter in tests)
	- Acceptance:
		- Backoff delay follows $\min(\text{max}, \text{base}\cdot2^k)$ with full jitter.
		- Backoff state increments on throttle events.
		- Backoff resets/decays after a stable (non-throttle) period per spec intent (define explicitly in code + tests).

- [ ] [P2-T4] Implement throttle classification from output tail (not exception stdout/stderr)
	- Location: `scripts/dev_tools/atomic_executor/copilot_throttling.py`
	- Symbols:
		- `classify_copilot_failure(exit_code: int, output_tail: str) -> FailureKind`
		- `FailureKind` enum includes at least: `THROTTLE`, `NON_THROTTLE`
	- Acceptance:
		- Positive classifications include patterns like “rate limit”, “rate limited”, “throttle”, “429”, “503” (case-insensitive).
		- Negative samples (permission errors, auth errors, generic failures) do not classify as throttle.

### Phase 3 — Wire throttling policy into Copilot invocation

- [ ] [P3-T1] Capture an in-memory output tail during Copilot streaming
	- Files:
		- `scripts/dev_tools/atomic_executor/cli.py`: `_stream_copilot_output()` (or a new helper)
	- Acceptance:
		- Streaming still prints to console + writes full log.
		- A bounded tail buffer (bytes/characters) is retained and returned for classification.
		- Classification does not depend on `CalledProcessError.stdout/stderr`.

- [ ] [P3-T2] Add a throttle-aware retry loop around the Copilot CLI call
	- Files:
		- `scripts/dev_tools/atomic_executor/cli.py`: `run_copilot()` and/or `_execute_one_task()`
	- Behavior:
		- Gate every invocation behind the call-rate limiter.
		- On throttle detection, apply backoff delay (in addition to limiter) and retry.
		- On non-throttle failures, fail fast with actionable context including exit code + output tail snippet.
	- Acceptance:
		- Retries are bounded by a new configuration parameter (default bounded).
		- Task ordering is preserved: the executor does not advance to QC or checkbox flipping until a Copilot invocation succeeds.

- [ ] [P3-T3] Add user-facing CLI flags for throttling controls
	- Files:
		- `scripts/dev_tools/atomic_executor/cli.py`: `parse_args()` / `add_common()`
	- Flags (names may be adjusted, but must exist and be documented in `--help`):
		- `--copilot-cli-max-calls-per-window`
		- `--copilot-cli-window-seconds`
		- `--copilot-cli-backoff-base-seconds`
		- `--copilot-cli-backoff-max-seconds`
		- `--copilot-cli-output-tail-bytes`
		- `--copilot-cli-max-retries`
	- Acceptance:
		- Defaults match Phase 0 decisions.
		- Help text explicitly states this is call-rate based (not token based).

### Phase 4 — Executor ordering invariants + integration tests

- [ ] [P4-T1] Add unit test: executor does not advance tasks under throttle
	- New test module:
		- `tests/scripts/dev_tools/atomic_executor/test_executor_throttle_ordering.py`
	- Acceptance:
		- Verifies the executor retries within the same task loop (using the injectable Copilot runner seam).
		- Verifies no checkbox is flipped and no next-task selection occurs until the throttled call succeeds.

- [ ] [P4-T2] Add unit test: bounded retries terminate when always throttled
	- New test module:
		- `tests/scripts/dev_tools/atomic_executor/test_executor_throttle_bounded_retries.py`
	- Acceptance:
		- Confirms executor returns non-zero after configured max retries.

### Phase 5 — Verification + docs

- [ ] [P5-T1] Run the full repo toolchain loop (format → lint → type-check → tests)
	- Commands:
		- `poetry run black .`
		- `poetry run ruff check`
		- `poetry run pyright`
		- `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`
	- Acceptance:
		- All four steps pass in a single final pass.

- [ ] [P5-T2] Update documentation for new throttling controls
	- Files (choose the smallest correct surface):
		- `docs/ci-documentation.md` and/or `docs/developer-tooling.md` and/or `README.md`
	- Acceptance:
		- Documents new CLI flags and safe defaults.
		- Includes a short note: Copilot CLI native Windows PowerShell is experimental; WSL is recommended when available.

### Phase 6 — PR & follow-up

- [ ] [P6-T1] Prepare PR notes: rationale, risks, and validation
	- Acceptance:
		- Notes explicitly state: no attempt was made to trigger real throttling; behavior is covered via deterministic tests.
		- Includes links to key tests covering classifier/limiter/backoff/order/bounded retries.

### Phase 7 — Release QC gate (must be last)

- [ ] [P7-T1] Run the full repo QC gate for all languages; must pass clean
	- Purpose:
		- This is the final release gate. It cannot be considered successful unless the repo is fully compliant.
	- Preferred:
		- Run the VS Code task: `QC: 5 Run All Checks`
	- If running manually, run the full toolchain for all languages (in this order):
		- `poetry run python -m scripts.dev_tools.format_json`
		- `poetry run python -m scripts.dev_tools.validate_json`
		- `poetry run python -m scripts.dev_tools.shell_qc format`
		- `poetry run python -m scripts.dev_tools.shell_qc check`
		- `poetry run python -m scripts.dev_tools.shell_qc test`
		- `poetry run black .`
		- `poetry run ruff check`
		- `poetry run pyright`
		- `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`
		- `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/format-powershell.ps1`
		- `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/run-psscriptanalyzer.ps1`
		- `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/run-pester.ps1`
	- Acceptance:
		- All checks pass in a single final pass (no lingering formatting or lint diffs).
		- If any step changes files or fails, fix issues and restart the gate from the beginning.
