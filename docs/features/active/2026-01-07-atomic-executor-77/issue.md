# Feature: Atomic Executor Script for Copilot Plan Execution (Issue #77)

- Date captured: 2026-01-07
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/Potential_Feature_Atomic_Executor_Script_for_Copilot_Plan_Execution/ (Issue #77)

- Issue: #77
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/77
- Last Updated: 2026-01-07
## Problem / Why

Copilot Agent Mode cannot execute multi-step plans autonomously across multiple turns due to enforced user confirmation between actions. This limits automation and requires manual intervention after each step, even when a deterministic plan is pre-approved. There is a need for a master script that can enforce atomic, stepwise execution of a Copilot-generated plan, ensuring no replanning, strict sequential gating, and toolchain validation after each step, with minimal human friction.

## Proposed Behavior

Implement a Python-based master script (`atomic_executor.py`) that:
- Loads a plan-of-record from a feature folder (with `plan.md`, `spec.md`, etc.)
- Validates plan structure and preconditions (Phase 0, task IDs, QA/toolchain phase)
- Executes **exactly one task per Copilot call** (no replanning, no scope drift)
- After each task:
	- Runs a scoped toolchain (Black, Ruff, Pyright on changed files; Pytest only if tests changed)
- After each phase:
	- Runs the full toolchain (Black, Ruff, Pyright, Pytest)
- Updates `plan.md` by flipping only the current task checkbox when gates pass
- Supports `execute`, `resume`, and `--start P#-T#` for flexible entry
- Integrates with existing prompt templates and feature folder resolution logic

## Acceptance Criteria (early draft)

- [ ] Script loads and validates a Copilot-generated plan from a feature folder
- [ ] Executes one task per invocation, strictly in plan order
- [ ] Runs scoped QC after each task, full QC after each phase
- [ ] Updates plan.md checkboxes only when all gates pass
- [ ] Supports resume and start-at-task options
- [ ] No replanning or scope drift allowed
- [ ] Works cross-platform (Windows, Linux, Mac)
- [ ] Logs actions and errors for traceability

## Constraints & Risks

- Handling of toolchain failures or partial state
- Plan format drift or nonstandard plans
- Integration with Copilot prompt templates and feature folder conventions
- Performance on large plans or repos

## Test Conditions to Consider

- [ ] Unit coverage areas (plan parsing, task execution, QC gating)
- [ ] Integration scenarios (feature folder with various plan structures)
- [ ] CLI/API examples (execute, resume, start-at-task)

## Next Step

- [ ] Promote to GitHub issue (feature request template)
- [ ] Create `docs/features/active/atomic-executor/` folder from the template

## References

- See chat: 260107-01 Copilot-Python-Automation.md (artifacts/chats/260107-01 Copilot-Python-Automation.md)
- Related: resolve_execute_plan_prompt.py, execute-atomic-plan.prompt.md
