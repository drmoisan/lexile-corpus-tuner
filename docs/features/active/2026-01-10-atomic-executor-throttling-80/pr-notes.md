# PR notes — Issue #80 (atomic_executor throttling)

## Summary
This change makes `atomic_executor` resilient to GitHub Copilot CLI throttling by:

- Enforcing a configurable **call-rate limit** (calls per time window).
- Retrying detected throttling failures with bounded **exponential backoff + jitter**.
- Classifying throttle vs non-throttle failures using a bounded **captured output tail** (not `CalledProcessError.stdout/stderr`).
- Preserving strict ordering: the executor does not advance tasks until the current task succeeds and scoped QC validates.

## Why
`atomic_executor` can hit Copilot CLI throttling based on **call cadence**, especially during long execute-all runs.

## Risks
- **False positives** in throttle classification could add unnecessary delays.
- **Overly conservative defaults** may slow long runs.

Mitigations:
- Centralized, unit-tested classification logic.
- All pacing parameters are user-configurable.
- Retry behavior is bounded by default.

## Validation
No attempt was made to trigger real throttling. Behavior is covered via deterministic unit tests that simulate throttle → recovery.

Key tests:
- `tests/scripts/dev_tools/atomic_executor/test_copilot_throttling_classifier.py`
- `tests/scripts/dev_tools/atomic_executor/test_copilot_rate_limiter.py`
- `tests/scripts/dev_tools/atomic_executor/test_copilot_backoff.py`
- `tests/scripts/dev_tools/atomic_executor/test_executor_throttle_retry_regression.py`
- `tests/scripts/dev_tools/atomic_executor/test_executor_throttle_ordering.py`
- `tests/scripts/dev_tools/atomic_executor/test_executor_throttle_bounded_retries.py`

Commands run locally:
- `poetry run black .`
- `poetry run ruff check`
- `poetry run pyright`
- `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`
- `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCFormat -Root ."`
- `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCAnalyze -Root ."`
- `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCTest -Root ."`
