# refactor-atomic-executor-cli-length (Issue #91)

- Date captured: 2026-01-17
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/refactor-atomic-executor-cli-length/ (Issue #91)

> Automation note: Keep the section headings below unchanged; the promotion tooling maps each of them into the GitHub bug issue template.

- Issue: #91
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/91
- Last Updated: 2026-01-17
## Summary

The `scripts/dev_tools/atomic_executor/cli.py` module exceeds the 500-line policy limit (1321 lines). This violates repo structure guidance and increases maintenance risk.

## Environment

- OS/version: Debian GNU/Linux 12 (bookworm)
- Python version: 3.13 (devcontainer)
- Command/flags used: `wc -l scripts/dev_tools/atomic_executor/cli.py`
- Data source or fixture: N/A

## Steps to Reproduce

1. Open `scripts/dev_tools/atomic_executor/cli.py`.
2. Count lines (e.g., `wc -l scripts/dev_tools/atomic_executor/cli.py`).
3. Compare the line count to the 500-line limit in `general-code-change.instructions.md`.

## Expected Behavior

The module should be at or below 500 lines, or split into smaller cohesive modules.

## Actual Behavior

The file is 1321 lines long, exceeding the 500-line policy limit.

## Logs / Screenshots

- [ ] Attached minimal logs or screenshot
- Snippet: `scripts/dev_tools/atomic_executor/cli.py: 1321`

## Impact / Severity

- [ ] Blocker
- [ ] High
- [ ] Medium
- [x] Low

## Suspected Cause / Notes

The CLI module accumulated orchestration, subprocess handling, lock management, and logging without being split into smaller modules. The policy audit flagged this as pre-existing (not introduced by the current fix).

## Proposed Fix / Validation Ideas

- [ ] Unit coverage areas: retain existing tests for `run_copilot`, lock handling, and task execution; add new unit tests for any new modules created during refactor.
- [ ] Integration scenario to retest: run `poetry run pytest tests/scripts/dev_tools/test_atomic_executor_cli.py -v` and end-to-end `execute-all` flow in a dev session.
- [ ] Manual verification notes: ensure refactor preserves CLI arguments, session continuity, and lock-file behavior.

## Next Step

- [x] Promote to GitHub issue (bug-report template)
- [ ] Move to active fix folder / branch