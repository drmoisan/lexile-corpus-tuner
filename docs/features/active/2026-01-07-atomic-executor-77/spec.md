# 2026-01-07-atomic-executor — Spec

- Issue: #77
- Owner: drmoisan
- Last Updated: 2026-01-07

## Overview

Copilot Agent Mode cannot execute multi-step plans autonomously across multiple turns due to enforced user confirmation between actions. This limits automation and requires manual intervention after each step, even when a deterministic plan is pre-approved. There is a need for a master script that can enforce atomic, stepwise execution of a Copilot-generated plan, ensuring no replanning, strict sequential gating, and toolchain validation after each step, with minimal human friction.


## Behavior

Implement a Python-based master script (`atomic_executor.py`) that can:
- Load a plan-of-record from a feature folder (plan/spec/story).
- Validate preconditions (clean git status, non-protected branch, Phase 0 section, QA/toolchain phase, task IDs).
- Loop through every remaining task via the `execute-all` command, invoking Copilot once per task while relentlessly iterating until the task passes its gates.
- Run scoped QC after each task and full QC after each completed phase before advancing.
- Update `plan.md` checkboxes only after QC success; on failure, keep the current task in focus, keep prompting for fixes, and only stop after exhausting the configured retry policy.
- Support recovery through `resume`, `execute`, and `--start` to rerun individual tasks when needed.
- Integrate the existing prompt template so each Copilot call stays bound to the active `[P#-T#]` task.

### Loop flow (`execute-all`)
1. Operator runs `python -m scripts.dev_tools.atomic_executor.cli execute-all <feature-folder>` (with optional overrides such as `--feature`, `--start`, `--prompt-template`, `--max-fix-attempts`).
2. Script resolves the feature folder, parses `plan.md`, and runs preflight validation.
3. Picks the first unchecked task (or the explicit `--start` target) and hydrates the Copilot prompt.
4. Invokes Copilot exactly once for that task, then inspects `git status` to capture changed files.
5. Runs scoped QC on touched Python files/tests; if any gate fails, immediately re-prompts Copilot to repair the failure, reapplies formatting/linting/type checking, and repeats until the task clears the gate or the retry budget is exhausted (`max-fix-attempts = 0` can be treated as infinite retry mode).
6. On QC success, ensures the checkbox is marked, writes the update atomically, and logs the run (including how many retry cycles were required).
7. If the finished task completes its phase, runs the full toolchain (`black --check`, `ruff check`, `pyright`, full `pytest`) and, when failures occur, loops in the same "fix → re-run" pattern until the phase passes or retry budget is spent.
8. Re-parses `plan.md` and repeats until no unchecked tasks remain; prints summary and exits 0 with a tally of retry loops executed.
9. If the retry budget is exhausted without clearing the gate, logs the failure, leaves the checkbox unchecked, emits exit code 5, and surfaces explicit guidance for resuming the same task after manual fixes.

### Alternative flows
- `execute`: single-task execution remains available for dry runs or manual checkpoints.
- `resume`: re-enters the loop after a failure, continuing from the first unchecked task and honoring the same "never give up" retry process.
- `--start`: allows targeted execution of a specific pending task within `execute`, `resume`, or `execute-all`.
- `--print-prompt` / `--copy-prompt`: optional flags to preview or copy the prompt without invoking Copilot (useful for verification when no tasks remain).
- Validation failures (dirty git tree, missing files, ambiguous feature folder) abort before the loop begins with descriptive errors.


## Inputs / Outputs

- **Inputs**
  - CLI verbs: `execute`, `resume`, `execute-all`.
  - Common options: `--feature <name>`, `--start P#-T#`, `--prompt-template <path>`, `--max-fix-attempts <int>` (0 = no limit), `--print-prompt`, `--copy-prompt`.
  - Required files: `plan.md`, `spec.md`; optional `user-story.md` for richer prompt context.
  - Environment assumptions: Poetry tooling on PATH, Copilot agent available, clipboard utility (pyperclip or OS-specific command).
- **Outputs**
  - Updated `plan.md` checkboxes for successfully gated tasks (atomic file replace).
  - Console stream showing each task, QC commands, retry counts, and phase-level gates.
  - Structured log in `.agent_logs/<timestamp>.log` containing command history, outcomes, and failure diagnostics.
  - Clipboard contents set to the prepared prompt for each task (best-effort; failures logged but not fatal).


## API / CLI Surface

- `python -m scripts.dev_tools.atomic_executor.cli execute-all docs/features/active/<feature-folder> --max-fix-attempts 2`
- `python -m scripts.dev_tools.atomic_executor.cli resume docs/features/active/<feature-folder>`
- `python -m scripts.dev_tools.atomic_executor.cli execute docs/features/active/<feature-folder> --start P1-T3`
- Optional Poetry alias: `poetry run atomic-exec execute-all <feature-folder>` (if configured in `pyproject.toml`).
- Exit codes: `0` success, `1` validation/preflight failure, `2` usage error, `5` gate failure during execution.
- Output includes per-task status lines (e.g., `P1-T2 ✅ scoped QC (black, ruff, pyright)`), retry counters, and phase-completion summaries that prove no step was skipped.


## Data & State

- Parses `plan.md` into an ordered task list, reloading the file after every loop iteration to stay synchronized with checkbox state.
- Uses `git status --porcelain` to detect changed files and scope QC runs.
- Writes to `plan.md` using a temporary file + replace pattern to avoid corruption.
- Generates log files under `.agent_logs/` capturing timestamps, selected tasks, every retry attempt, commands issued, and exit codes so reviewers can confirm persistence.
- Clipboard writes per task; if clipboard interaction fails, the prompt is still printed so the operator can copy manually.


## Constraints & Risks

- Handling of toolchain failures or partial state remains critical; the loop must keep focus on the failing gate, iterating Copilot prompts and QC cycles until success or an explicit stop condition is reached.
- Plan format drift (e.g., missing `[P#-T#]`) will break task parsing; add early validation to surface these issues.
- Integration with Copilot prompt templates must remain stable; template renames require corresponding CLI flag updates.
- Performance considerations on large plans: long-running loops may bump against Copilot rate limits; include delays or guidance while emphasizing that the loop should prefer persistence over premature exit.
- Cross-platform clipboard tooling may vary; ensure fallback commands for Windows (`clip`), macOS (`pbcopy`), Linux (`wl-copy`, `xclip`, `xsel`).
- Reliance on `poetry` assumes the environment is prepared; provide clear errors when commands are missing.


## Definition of Done

- [ ] Behavior matches acceptance criteria, including looping through all tasks via `execute-all`.
- [ ] Tests updated/added (unit tests for plan parsing, QC scoping, execute-all orchestration, resume logic; integration smoke test with fixture plan).
- [ ] Docs updated (README CLI section, feature docs usage examples, troubleshooting guidance for loop failures).
- [ ] Telemetry/logging (if applicable) — `.agent_logs` captures command transcript, every retry cycle, and failure reasons.
- [ ] Example execute-all session recorded (screencast or transcript) and linked from the issue for reviewer confidence.


## Seeded Test Conditions (from potential)
- [ ] Unit coverage areas (plan parsing, task execution loop, QC gating)
- [ ] Integration scenarios (feature folder with multiple phases and deliberate failures)
- [ ] CLI/API examples (`execute`, `resume`, `execute-all` with `--start` overrides)
