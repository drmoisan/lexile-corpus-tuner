# `2026-01-07-atomic-executor` — User Story

- Issue: #77
- Owner: drmoisan
- Status: Draft | In Progress | Complete
- Last Updated: 2026-01-07

## Story Statement

- As a Copilot automation engineer responsible for enforcing repo guardrails, I want a deterministic executor that can run every remaining atomic task in one `execute-all` session so that I do not have to re-trigger Copilot manually between steps.
- As a repository maintainer who reviews Copilot-generated changes, I want the execution flow to loop through each task with QC gating and plan updates so that the final plan.md reflects all completed work without gaps.

## Problem / Why

Copilot Agent Mode cannot execute multi-step plans autonomously across multiple turns due to enforced user confirmation between actions. This limits automation and requires manual intervention after each step, even when a deterministic plan is pre-approved. There is a need for a master script that can enforce atomic, stepwise execution of a Copilot-generated plan, ensuring no replanning, strict sequential gating, and toolchain validation after each step, with minimal human friction.


## Personas & Scenarios

- Persona: Alex (Automation Engineer)
  - Owns Copilot agent configuration for the repo and must keep execution compliant with repo policies.
  - Cares about deterministic task execution, quick recovery on failure, and accurate plan status.
  - Works under the constraint that Copilot Agent Mode pauses after each turn and cannot replan mid-run.
  - Goals: minimize manual babysitting, ensure QC toolchain is honored, provide auditable evidence of completion.
  - Frustrations: repeated "Continue" prompts, drift when Copilot improvises, time lost rerunning full toolchains unnecessarily.
- Scenario: Alex receives a ready-to-execute plan in docs/features/active/2026-01-07-atomic-executor-77. They run `python -m scripts.dev_tools.atomic_executor.cli execute-all docs/features/active/2026-01-07-atomic-executor-77`, which resolves the feature folder, selects the next unchecked task (P1-T2), and copies the pre-approved prompt. Copilot implements code for the task, but scoped QC surfaces Ruff and Pyright failures. Within the same session, the executor re-prompts Copilot to fix issues, reapplies formatting/linting/type checks, and repeats until the task passes or the configured retry limit is hit. After the task clears all gates, the loop checks the box, reloads the plan, and proceeds to P1-T3. When Phase 1 finishes, the script runs the full toolchain before continuing into Phase 2. The expected outcome is that one command processes all remaining tasks in order, iterating on failures until they are resolved and stopping only when the plan completes or the retry budget is exhausted.


## Acceptance Criteria

- [ ] Script loads and validates a Copilot-generated plan from a feature folder
- [ ] `execute-all` loops through remaining tasks sequentially until the plan completes or a gate fails
- [ ] Each loop iteration invokes Copilot exactly once for the current task and runs scoped QC on changed files
- [ ] After each completed phase, the script runs the full QC toolchain automatically
- [ ] Plan checkboxes update only when the corresponding task passes QC; failures trigger iterative Copilot fixes within the same run and only abort after exhausting the configured retry budget
- [ ] `execute`, `resume`, and `--start` flows remain available for targeted recovery after a failure
- [ ] Works cross-platform (Windows, Linux, Mac) and writes audit logs per run
- [ ] Loop termination conditions and errors are clearly reported to the operator


## Non-Goals

- The script will not attempt to bypass Copilot’s enforced pause between turns or automate UI interactions.
- The solution does not introduce new planning heuristics or auto-generate plans; it consumes existing plan.md content only.
- No attempt will be made to infer or reroute dependencies to run partial pytest matrices beyond the changed-test heuristic.
- The script will not manage Git operations (commit, push, branch creation) beyond reading repo status.
- The loop will not silently skip failed tasks; it continues working the current task (via Copilot retries) until the gate passes or the retry budget is exhausted. Manual intervention is required only when retries are exhausted.
