# atomic-executor-qc-regression-test (Issue #98)

- Date captured: 2026-01-20
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/atomic-executor-qc-regression-test/ (Issue #98)

> Automation note: Keep the section headings below unchanged; the promotion tooling maps each of them into the GitHub bug issue template.

- Issue: #98
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/98
- Last Updated: 2026-01-21
## Summary

When the atomic executor exits mid-execution after completing TDD "red" regression tests (tasks tagged with `[expect-fail]`), restarting the executor causes pre-flight QC to fail on those intentionally-failing tests. Pre-flight QC is unaware of the plan's `[expect-fail]` semantics.

## Environment

- OS/version: Linux (devcontainer)
- Python version: 3.13
- Command/flags used: `poetry run python -m scripts.dev_tools.atomic_executor.cli execute-all <plan> --workspace <path>`
- Data source or fixture: Any plan with `[expect-fail]` TDD tasks in an early phase

## Steps to Reproduce

1. Create an atomic plan with Phase 1 TDD regression tests tagged with `[expect-fail]`
2. Run the atomic executor (`execute-all`)
3. Allow Phase 1 `[expect-fail]` tasks to complete (tests are created and fail as expected)
4. Interrupt or crash the executor before Phase 2 implementation tasks complete
5. Restart the executor on the same plan (without `--skip-preflight-qc`)
6. Pre-flight QC runs pytest, detects the failing regression tests, and attempts to "fix" them

## Expected Behavior

Pre-flight QC should recognize that failing tests correspond to completed `[expect-fail]` plan tasks with pending implementation tasks, and either:
- Skip those specific tests during pre-flight QC, or
- Skip pre-flight QC entirely when resuming mid-plan execution

## Actual Behavior

Pre-flight QC treats the failing regression tests as unexpected failures and invokes Copilot to fix them, which defeats the TDD workflow.

## Logs / Screenshots

- [x] Attached minimal logs or screenshot
- Snippet: (observed behavior, no specific log captured)

## Impact / Severity

- [ ] Blocker
- [ ] High
- [x] Medium
- [ ] Low

The issue only manifests when the executor is interrupted mid-TDD-workflow and restarted. Workaround exists.

## Suspected Cause / Notes

The `[expect-fail]` tag lives in the plan file and is only interpreted during task execution. Pre-flight QC (`_run_preflight_qc_if_needed` in `cli.py`) runs before any plan parsing and has no awareness of which failing tests are intentional.

Relevant files:
- `scripts/dev_tools/atomic_executor/cli.py` — pre-flight QC logic
- `scripts/dev_tools/atomic_executor/plan_parser.py` — `[expect-fail]` parsing

## Proposed Fix / Validation Ideas

**Immediate workaround:**
- Use `--skip-preflight-qc` when resuming after interruption

**Potential fixes (future enhancement):**

| Approach | Pros | Cons |
|----------|------|------|
| **Plan-aware pre-flight** | Single source of truth | Complex: must parse plan, correlate tests |
| **State file** (`.expect_fail_tests.json`) | Pre-flight can exclude specific tests | State management, can get stale |
| **Auto-skip on resume** | Simple UX | May mask real issues |
| **Pytest `xfail` marker** | Standard pytest mechanism | Must remove marker after fix |

Recommended approach: **Plan-aware pre-flight QC** that:
1. Parses the plan before running QC
2. Finds `[expect-fail]` tasks that are checked (test created) but whose implementation sibling is unchecked
3. Runs pytest with `--deselect` for those specific test functions

- [ ] Unit coverage: test that pre-flight QC respects `[expect-fail]` completed tasks
- [ ] Integration scenario: interrupt executor after `[expect-fail]` task, resume, verify no spurious fix attempts
- [ ] Manual verification: use `--skip-preflight-qc` workaround

## Next Step

- [ ] Promote to GitHub issue (bug-report template)
- [ ] Move to active fix folder / branch