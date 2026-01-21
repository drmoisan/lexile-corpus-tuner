# atomic-executor-tdd-regression-test (Issue #96)

- Date captured: 2026-01-20
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/atomic-executor-tdd-regression-test/ (Issue #96)

> Automation note: Keep the section headings below unchanged; the promotion tooling maps each of them into the GitHub bug issue template.

- Issue: #96
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/96
- Last Updated: 2026-01-21
## Summary

The `atomic_executor` tool prevents completion of TDD "Red" phase tasks because it enforces a passing toolchain (including tests) before allowing a task to be checked off, creating an impasse when the task goal is to introduce a failing test.

## Environment

- OS/version: Linux (Dev Container)
- Python version: 3.10+
- Command/flags used: `atomic_executor execute --path docs/features/active/2026-01-20-ck12-missing-enrichment-links-95`
- Data source or fixture: `docs/features/active/2026-01-20-ck12-missing-enrichment-links-95/plan.2026-01-20T16-24.md`

## Steps to Reproduce

1. Create a plan with a task that requires adding a failing regression test (e.g., [P1-T1] in the linked plan).
2. Run the atomic executor on this plan.
3. Observe the agent correctly adding the test file which asserts failure.
4. Observe the executor running the post-task toolchain/QC check.

## Expected Behavior

The executor should implicitly or explicitly allow the task to be marked complete if the task's stated goal is to reproduce a bug or create a failing test (Acceptance Criteria: "fails before code change").

## Actual Behavior

1. The executor runs the full toolchain (including `pytest`), which fails due to the newly added regression test. Consequently, the executor considers the task incomplete/failed and refuses to mark it as done, causing an infinite loop (it retries until max attempts are exhausted).
2. Additional observation: After failing the task and printing an error/exit message (e.g., exiting with code 1 or 5), the process sometimes does not terminate cleanly. It continues to run background operations or loops indefinitely.
3. In this "zombie" state, the loop iteration counter appears stuck (e.g., repeatedly executing "Attempt 4" without incrementing or exiting).

## Logs / Screenshots

- [ ] Attached minimal logs or screenshot
- Snippet:
  Task `[P1-T1]` implementation correct.
  Running post-task QC...
  Pytest failed.
  Task verification failed. Retrying...
  ...
  Failed to complete task P1-T1 after 3 attempts.
  (Process continues running...)
  Running post-task QC...

## Impact / Severity

- [x] Blocker
- [ ] High
- [ ] Medium
- [ ] Low

## Suspected Cause / Notes

1. **TDD Conflict**: The executor enforces an unconditional "all checks pass" policy.
2. **Zombie Process**: Likely a threading issue (daemon threads not joining) or the `main` loop error handling not strictly causing a process exit when `execute_one_task` returns a failure code. The "stuck counter" suggests the failure return code might be ignored in the caller (`execute-all` loop?), causing it to retry the *same task* again without context update, or simply looping on the same task index.

## Proposed Fix / Validation Ideas

- [ ] Introduce a `[expect-fail]` tag or parsed metadata for plan tasks that allows specific test failures.
- [ ] Allow the agent to explicitly override the QC check if the output matches the acceptance criteria (LLM-based verification of failure).
- [ ] Modify the `atomic_executor` to accept a specific non-zero exit code if asserted by the agent.

## Next Step

- [ ] Promote to GitHub issue (bug-report template)
- [ ] Move to active fix folder / branch