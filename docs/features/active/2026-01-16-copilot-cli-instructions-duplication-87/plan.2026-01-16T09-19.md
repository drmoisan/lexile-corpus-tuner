# 2026-01-16-copilot-cli-instructions-duplication (Plan)

- **Issue:** #87
- **Owner:** drmoisan
- **Date:** 2026-01-16T09-19
- **Status:** Draft
- **Outcome:** Eliminate instruction duplication, use native agent profile, implement session continuity
- **Root Cause:** prompt_builder.py inlines instructions that Copilot CLI already auto-loads; cli.py ignores `--agent` flag and starts new sessions per task

---

## Authoritative References

- **Spec (final authority on acceptance):** [spec.md](spec.md)
- **Research:** [20260116-copilot-cli-instructions-duplication-research.md](20260116-copilot-cli-instructions-duplication-research.md)
- **Issue:** [GitHub Issue #87](https://github.com/drmoisan/lexile-corpus-tuner/issues/87)

---

### Phase 0 — Context & Inputs

- [ ] [P0-T1] Read `.github/copilot-instructions.md` and `.github/instructions/general-code-change.instructions.md` to establish baseline coding rules
- [ ] [P0-T2] Read `.github/instructions/general-unit-test.instructions.md` and `.github/instructions/python-unit-test.instructions.md` to confirm testing standards
- [ ] [P0-T3] Read `.github/instructions/python-code-change.instructions.md` to confirm Python-specific coding rules
- [ ] [P0-T4] Capture baseline Ruff results by running `poetry run ruff check` and recording pass/fail status
- [ ] [P0-T5] Capture baseline Pyright results by running `poetry run pyright` and recording pass/fail status
- [ ] [P0-T6] Capture baseline Pytest results by running `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing` and recording pass/fail status and coverage percentage

### Phase 1 — Regression Tests (Fix 1: Prompt Builder)

- [ ] [P1-T1] Add test `test_prompt_excludes_instructions_md_content` in `tests/scripts/dev_tools/test_resolve_execute_plan_prompt.py` that asserts generated prompt does NOT contain the string `"---- BEGIN repo instructions ----"`
- [ ] [P1-T2] Add test `test_prompt_excludes_copilot_instructions_content` in `tests/scripts/dev_tools/test_resolve_execute_plan_prompt.py` that asserts generated prompt does NOT contain `"copilot-instructions.md"` as inlined section header
- [ ] [P1-T3] Add test `test_prompt_size_under_threshold` in `tests/scripts/dev_tools/test_resolve_execute_plan_prompt.py` that asserts generated prompt byte size is under 15KB (allowing margin above 10KB target)
- [ ] [P1-T4] Run tests added in P1-T1 through P1-T3 to confirm they FAIL against current implementation (regression baseline)

### Phase 2 — Regression Tests (Fix 2: Agent Flag)

- [ ] [P2-T1] Add test `test_copilot_argv_includes_agent_flag` in `tests/scripts/dev_tools/test_atomic_executor_cli.py` that asserts the Copilot CLI argument list contains `["--agent", "atomic_executor"]`
- [ ] [P2-T2] Run test added in P2-T1 to confirm it FAILS against current implementation

### Phase 3 — Regression Tests (Fix 3: Session Continuity)

- [ ] [P3-T1] Add test `test_first_task_omits_continue_flag` in `tests/scripts/dev_tools/test_atomic_executor_cli.py` that asserts `--continue` is NOT present in argv for the first task of a plan run
- [ ] [P3-T2] Add test `test_subsequent_task_includes_continue_flag` in `tests/scripts/dev_tools/test_atomic_executor_cli.py` that asserts `--continue` IS present in argv for tasks after the first
- [ ] [P3-T3] Run tests added in P3-T1 and P3-T2 to confirm they FAIL against current implementation

### Phase 4 — Regression Tests (Fix 4: Single-Run Guard)

- [ ] [P4-T1] Add test `test_single_run_lock_acquired_on_start` in `tests/scripts/dev_tools/test_atomic_executor_cli.py` that asserts a lock file is created under `.agent_logs/` when executor starts
- [ ] [P4-T2] Add test `test_single_run_lock_blocks_concurrent_run` in `tests/scripts/dev_tools/test_atomic_executor_cli.py` that asserts executor raises an error when lock file already exists
- [ ] [P4-T3] Add test `test_single_run_lock_released_on_completion` in `tests/scripts/dev_tools/test_atomic_executor_cli.py` that asserts lock file is removed after executor completes (success or failure)
- [ ] [P4-T4] Run tests added in P4-T1 through P4-T3 to confirm they FAIL against current implementation

### Phase 5 — Implement Fix 1: Remove Instruction Inlining

- [ ] [P5-T1] In `scripts/dev_tools/atomic_executor/prompt_builder.py`, remove the `instructions: list[tuple[str, str]] = []` variable and all code that populates it (lines ~238-260)
- [ ] [P5-T2] In `scripts/dev_tools/atomic_executor/prompt_builder.py`, remove the `_format_instructions` method entirely
- [ ] [P5-T3] In `scripts/dev_tools/atomic_executor/prompt_builder.py`, remove the `---- BEGIN repo instructions ----` / `---- END repo instructions ----` block from the appended prompt string
- [ ] [P5-T4] In `scripts/dev_tools/atomic_executor/prompt_builder.py`, update the Constraints section to remove the line `- Follow all repository policies under .github/instructions/.` (now implicit via CLI)
- [ ] [P5-T5] Run tests from Phase 1 (P1-T1 through P1-T3) to confirm they now PASS

### Phase 6 — Implement Fix 2: Add Agent Flag

- [ ] [P6-T1] In `scripts/dev_tools/atomic_executor/cli.py`, locate the `run_copilot()` function and add `["--agent", "atomic_executor"]` to the `argv` list after the copilot_exe entry
- [ ] [P6-T2] Run test from Phase 2 (P2-T1) to confirm it now PASSES

### Phase 7 — Implement Fix 3: Session Continuity

- [ ] [P7-T1] In `scripts/dev_tools/atomic_executor/cli.py`, add a parameter `is_first_task: bool` to the `run_copilot()` function signature with default `True`
- [ ] [P7-T2] In `scripts/dev_tools/atomic_executor/cli.py`, add logic: if `not is_first_task` and `supports_sessions`, append `["--continue"]` to argv
- [ ] [P7-T3] In `scripts/dev_tools/atomic_executor/cli.py`, update all call sites of `run_copilot()` to pass `is_first_task=False` for subsequent tasks in a plan run (track via loop index or state)
- [ ] [P7-T4] In `scripts/dev_tools/atomic_executor/cli.py`, add log entry indicating whether task uses `--continue` or starts a new session
- [ ] [P7-T5] Run tests from Phase 3 (P3-T1 and P3-T2) to confirm they now PASS

### Phase 8 — Implement Fix 4: Single-Run Guard

- [ ] [P8-T1] In `scripts/dev_tools/atomic_executor/cli.py`, add a constant `EXECUTOR_LOCK_FILE = ".agent_logs/executor.lock"` near module top
- [ ] [P8-T2] In `scripts/dev_tools/atomic_executor/cli.py`, create helper function `acquire_executor_lock(workspace: Path) -> Path` that creates the lock file and returns its path; raises `RuntimeError` if lock already exists
- [ ] [P8-T3] In `scripts/dev_tools/atomic_executor/cli.py`, create helper function `release_executor_lock(lock_path: Path) -> None` that removes the lock file if it exists
- [ ] [P8-T4] In `scripts/dev_tools/atomic_executor/cli.py`, update the `execute_all` entry point to call `acquire_executor_lock()` at start and `release_executor_lock()` in a finally block
- [ ] [P8-T5] Run tests from Phase 4 (P4-T1 through P4-T3) to confirm they now PASS

### Phase 9 — Implement Fix 5: Prompt Template Simplification

- [ ] [P9-T1] In `.github/prompts/execute-plan-template.md`, remove any instruction references that duplicate what the agent profile provides (keep only task execution guidance)
- [ ] [P9-T2] In `scripts/dev_tools/atomic_executor/prompt_builder.py`, add prompt byte-size logging: after building the prompt, log `f"Prompt size: {len(prompt_text)} bytes, {prompt_text.count(chr(10))} lines"`
- [ ] [P9-T3] In `scripts/dev_tools/atomic_executor/prompt_builder.py`, add a warning log if prompt size exceeds 15KB: `"WARNING: Prompt exceeds target threshold (15KB); consider reducing context"`

### Phase 10 — QA Verification Loop

- [ ] [P10-T1] Run `poetry run black .` and confirm no files are reformatted; if files change, commit and restart Phase 10
- [ ] [P10-T2] Run `poetry run ruff check` and confirm zero errors; if errors exist, fix and restart Phase 10 from P10-T1
- [ ] [P10-T3] Run `poetry run pyright` and confirm zero type errors; if errors exist, fix and restart Phase 10 from P10-T1
- [ ] [P10-T4] Run `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing` and confirm all tests pass; if any fail, fix and restart Phase 10 from P10-T1
- [ ] [P10-T5] Confirm coverage has not decreased from baseline captured in P0-T6; if decreased, add tests and restart Phase 10 from P10-T1

### Phase 11 — Documentation & Validation

- [ ] [P11-T1] Update `spec.md` Acceptance Criteria section to mark each criterion as verified with date
- [ ] [P11-T2] Add a "Validation Results" section to this plan documenting: prompt size before/after, session continuity evidence, lock file behavior observed
- [ ] [P11-T3] Update `docs/developer-tooling.md` Atomic Execution Agent section if any CLI flags or behaviors changed

### Phase 12 — PR & Handoff

- [ ] [P12-T1] Run `poetry run python -m scripts.dev_tools.collect_commit_context --output artifacts/commit_context.txt` to capture commit context
- [ ] [P12-T2] Create PR with title `fix(atomic-executor): remove instruction duplication and add session continuity (#87)` including summary of changes, test evidence, and link to spec
- [ ] [P12-T3] Add PR description section listing: files changed, tests added, behavioral changes, rollback instructions

### Phase 13 — Rollout & Follow-up

- [ ] [P13-T1] After merge, run atomic executor on a sample plan and verify prompt size is under 15KB
- [ ] [P13-T2] Verify session continuity by checking logs for `--continue` usage on tasks after the first
- [ ] [P13-T3] Update issue #87 with resolution summary and close

---

## Validation Results

| Metric | Baseline | After Fix | Target |
|--------|----------|-----------|--------|
| Prompt size (bytes) | ~179KB | 27,694 bytes | ≤15KB |
| Prompt lines | ~3,423 | 476 | <500 |
| Instructions inlined | Yes | No | No |
| `--agent` flag used | No | Yes | Yes |
| `--continue` for subsequent tasks | No | Yes (unit test) | Yes |
| Single-run lock enforced | No | Yes (unit test) | Yes |

Session continuity evidence:
- Unit tests assert `--continue` is omitted for the first task and included for subsequent tasks.

Single-run lock evidence:
- Unit tests assert lock acquisition, blocking, and release behavior.
