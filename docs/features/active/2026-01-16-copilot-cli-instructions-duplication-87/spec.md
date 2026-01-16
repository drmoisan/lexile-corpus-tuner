# 2026-01-16-copilot-cli-instructions-duplication (Spec)

- Issue: #87
- Owner: drmoisan
- Date: 2026-01-16
- Status: Draft

## Context
The `atomic_executor` has multiple fundamental design flaws that cause excessive token usage, context window overruns, and wasted compute. The prompt builder inlines all repository instruction files into the prompt, even though Copilot CLI **automatically loads them natively**. Additionally, the executor creates a new session for every task, discarding valuable context and causing "bootstrap" steps to execute in sessions that are immediately abandoned.

Environment:
- OS/version: Linux (devcontainer)
- Python version: 3.13
- Command/flags used: `poetry run python -m scripts.dev_tools.atomic_executor.cli execute-all`
- Data source or fixture: Any feature plan under `docs/features/active/`

Impact / Severity:
- [x] Blocker
- [ ] High
- [ ] Medium
- [ ] Low

**Justification**: 
- Token costs are exponentially higher than they should be (instructions loaded twice and no reused sessions)
- Context window is exhausted by duplicated instructions, leaving insufficient room for actual task context
- Session isolation defeats the purpose of multi-step plans
- "File too large to read" warnings occur because prompts are ~179KB when they should be ~10KB


## Repro & Evidence
Steps to Reproduce:
1. Run the atomic executor on any feature plan
2. Observe the generated prompt file in `.agent_logs/prompts/`
3. Note that it contains ~3,400 lines (~179KB) of inlined instruction content
4. Observe that Copilot CLI starts a new session for each task (P0-T1, P0-T2, etc.)
5. Note that "read policy files" tasks execute in a session that is discarded before implementation begins

Expected:
1. Instructions should be loaded once by Copilot CLI's native mechanism
2. A single session should persist across all tasks in a plan until the context window reaches 90%. Only then should a new session be opened.
3. Context from reading files in Phase 0 should carry forward to implementation phases
4. Token usage should be minimal

Actual:
### Problem 1: Instructions Loaded Twice (Duplication)

**Evidence from GitHub documentation** (https://docs.github.com/en/copilot/how-tos/use-copilot-agents/use-copilot-cli):

> "Custom instructions are natural language descriptions saved in Markdown files in the repository. **They are automatically included in prompts you enter while working in that repository.**"
>
> "Copilot CLI supports:
> - Repository-wide instructions in the `.github/copilot-instructions.md` file.
> - Path-specific instructions files: `.github/instructions/**/*.instructions.md`.
> - Agent files such as `AGENTS.md`."

**What atomic_executor does** ([prompt_builder.py#L243-L260](scripts/dev_tools/atomic_executor/prompt_builder.py#L243-L260)):
```python
# Load all repository instruction files to inline in the prompt
instructions: list[tuple[str, str]] = []

# Include copilot-instructions.md
copilot_instructions_path = self.workspace / ".github" / "copilot-instructions.md"
if self._fs.is_file(copilot_instructions_path):
    instructions.append(("copilot-instructions.md", self._read_text(copilot_instructions_path)))

# Auto-discover all *.instructions.md files in .github/instructions/
instructions_dir = self.workspace / ".github" / "instructions"
if self._fs.is_dir(instructions_dir):
    for instruction_file in self._fs.glob(instructions_dir, "*.instructions.md"):
        instructions.append((instruction_file.name, self._read_text(instruction_file)))
```

**Result**: All instruction files (~2,795 lines, ~130KB) are concatenated into the prompt. Since Copilot CLI already loads these files automatically, they are loaded **twice** — once natively and once via the prompt. This doubles the instruction token usage.

### Problem 2: Custom Agent Feature Not Utilized

**Evidence from GitHub documentation**:

> "Custom agents are specialized versions of Copilot coding agent... Custom agents are defined using Markdown files, called agent profiles..."
>
> "Specifying the custom agent you want to use with the command-line option:
> ```
> copilot --agent=refactor-agent --prompt "Refactor this code block"
> ```"

**What the repo already has**: A complete `atomic_execution.agent.md` agent profile at `.github/agents/atomic_execution.agent.md` (182 lines) that contains:
- Role definition ("execution-only agent")
- Policy compliance requirements
- Anti-replanning rules
- Task execution workflow
- Toolchain verification steps

**What atomic_executor does**: Ignores the `--agent` flag entirely and instead:
1. Inlines all instructions into a prompt file
2. Adds task context to the same prompt file
3. Passes the bloated prompt via `-p "Follow these instructions exactly: @{prompt_file}"`

**Result**: The carefully designed agent profile is never used. The executor reinvents agent behavior in the prompt template and prompt_builder.

### Problem 3: New Session for Every Task (Context Destruction)

**Evidence from GitHub documentation**:

> "You can use the `--resume` command line option to cycle through and resume local and remote interactive sessions, allowing you to pick up right where you left off with your existing context."
>
> "You can quickly resume the most recently closed local session by using the `--continue` command line option."

**What atomic_executor does** ([cli.py#L558-L562](scripts/dev_tools/atomic_executor/cli.py#L558-L562)):
```python
supports_sessions = _copilot_supports_session(copilot_exe)
if resume_session and supports_sessions:
    argv.extend(["--session-path", str(share_path)])
```

However, `resume_session` is only used for **retry attempts within the same task** — NOT for carrying context between tasks.

**Actual behavior**: Each task (P0-T1, P0-T2, P1-T1, etc.) starts a **completely new Copilot CLI session**. When a task completes:
1. The session is closed
2. All context (files read, decisions made, understanding gained) is discarded
3. A new session starts for the next task with zero context

**Result**: Extremely wasteful token usage. Every task re-ingests the same codebase context. No learning carries forward.

### Problem 4: Bootstrap Steps Execute in Wrong Session

**Evidence from plan structure**: Atomic plans typically begin with Phase 0 tasks like:
- `[P0-T1] Read and internalize repository policies`
- `[P0-T2] Capture baseline toolchain results`

**What happens**:
1. Executor starts session for P0-T1
2. Copilot reads `.github/instructions/*.md` files (per the prompt instructions)
3. P0-T1 completes → session is **discarded**
4. Executor starts **new session** for P0-T2
5. The context from reading the policy files is **completely lost**
6. Pattern repeats for every subsequent task

**Result**: Phase 0 "bootstrap" tasks are theater. They appear to prepare the agent, but the preparation is immediately destroyed. Implementation tasks (Phase 1+) start with zero context from Phase 0.

Logs / Screenshots:
- [x] Attached minimal logs or screenshot
- Snippet from generated prompt file structure (3,423 lines total):

```
Lines 1-48:    Task template + constraints
Lines 49-100:  copilot-instructions.md (DUPLICATED - already auto-loaded)
Lines 102-531: codexer.instructions.md (DUPLICATED)
Lines 533-825: general-code-change.instructions.md (DUPLICATED)
...
Lines 2893:    ---- END repo instructions ----
Lines 2904-3119: plan.md
Lines 3121-3378: spec.md
Lines 3380-3423: user-story.md
```


## Scope & Non-Goals
- In scope:
- Replace prompt instruction inlining with task-only prompt payloads and rely on Copilot CLI native instruction loading.
- Use the existing `.github/agents/atomic_execution.agent.md` profile via `--agent=atomic_executor` for all executor calls.
- Persist a single Copilot CLI session across tasks using `--continue` within a single run.
- Add rollover heuristics and validation guidance to respect the 90% context target.
- Enforce single-run exclusivity (lock/guard) to prevent `--continue` resuming unrelated sessions.
- Reduce prompt payload size and record prompt metrics for validation.
- Update prompt template and executor docs to match the new runtime flow.
- Add tests that verify CLI arguments, prompt contents, and rollover guards.
- Out of scope / non-goals:
- Implementing a programmatic “start session and return ID” workflow.
- Adding JSON or structured telemetry parsing for `/usage` or `/context`.
- Changing Copilot Memory behavior or introducing new CLI flags.
- Supporting concurrent or parallel executor runs.

## Root Cause Analysis
The `atomic_executor` was designed before understanding Copilot CLI's native capabilities:

1. **Instruction auto-loading**: The prompt builder was created assuming instructions needed to be injected manually. The GitHub docs clearly state they are "automatically included."

2. **Agent profiles**: The `--agent` flag and `.github/agents/` convention were either unknown or ignored during design. The `atomic_execution.agent.md` file exists but is never referenced.

3. **Session persistence**: The executor treats each task as an isolated unit rather than as steps in a continuous workflow. The `--continue` and `--resume` flags are not used between tasks.

4. **Bootstrap misunderstanding**: Phase 0 was designed assuming context would persist, but the implementation discards sessions after each task.

### Files to Inspect

- `scripts/dev_tools/atomic_executor/prompt_builder.py` — removes instruction inlining
- `scripts/dev_tools/atomic_executor/cli.py` — add `--agent` flag, implement session continuity
- `.github/prompts/execute-plan-template.md` — simplify to task-only content
- `.github/agents/atomic_execution.agent.md` — verify/enhance for direct use


## Proposed Fix
### Fix 1: Prompt builder payload redesign

- Remove instruction concatenation in `scripts/dev_tools/atomic_executor/prompt_builder.py`.
- Prompt payload must include only:
    - Task identifier and task text.
    - Feature context excerpts (plan/spec/user-story snippets as currently expected).
    - Retry context (if applicable).
- Add prompt-size guardrails:
    - Emit prompt byte size and line count to logs.
    - Warn when prompt exceeds the target threshold (≈10KB) and include a mitigation note.

### Fix 2: Agent profile usage

- Update `scripts/dev_tools/atomic_executor/cli.py` to pass `--agent=atomic_executor` for all Copilot CLI invocations.
- Remove any prompt-level policy re-statements that duplicate `atomic_execution.agent.md`.

### Fix 3: Session continuity via `--continue`

- Use `--continue` for all tasks after the first task in a plan run.
- Add a single-run lock or guard (e.g., a lock file under `.agent_logs/`) to prevent concurrent executor runs from sharing the same local Copilot session state.
- Record in logs whether the call is “first task” or “continued task”.

### Fix 4: Rollover heuristic for 90% context target

- Document the exact rollover rule in code/config (no implicit behavior):
    - Primary signal: manual `/usage` check in interactive session when diagnosing.
    - Automated fallback: task-count cap or prompt-size-based heuristic (defined in config or constants).
- When rollover triggers, start a new session (no `--continue`) and log the reason.

### Fix 5: Bootstrap handling

- Treat Phase 0 as a session initializer rather than a disposable task.
- Remove any instructions that force “read policy files” as a separate task; rely on Copilot’s native instruction loading and the agent profile.


## Assumptions, Constraints, Dependencies
- Assumptions (environment, data, access):
- Copilot CLI is installed and available on PATH for the atomic executor runtime.
- Repository instructions auto-load in Copilot CLI per GitHub documentation.
- `.github/agents/atomic_execution.agent.md` remains the authoritative execution policy source.
- Constraints (budget, performance, compatibility):
- **Exact automated 90% context rollover is not achievable with current Copilot CLI telemetry.** `/usage` and `/context` are interactive-only and not structured for automation; the spec must allow manual validation or heuristics.
- `--continue` resumes the most recent local session; concurrent runs can collide unless guarded.
- Prompt payloads must stay small (≈10KB target) to avoid view truncation and “file too large” warnings.
- Compatibility must preserve existing CLI entry points and plan formats.
- External dependencies (services, libraries, releases):
- GitHub Copilot CLI session behavior (`--continue`, `--resume`) and instruction auto-loading.

## Data / API / Config Impact
- User-facing or API changes:
- Atomic executor CLI invocation includes `--agent=atomic_executor` and `--continue` when appropriate.
- Prompts no longer include inlined `.github/instructions` content.
- Data or migration considerations:
- No data migrations; prompt layout changes only.
- Logging/telemetry updates (if any):
- Log prompt size (bytes/lines) for each task.
- Log when `--continue` is used and when rollover heuristics trigger a new session.
- Log the single-run guard state (lock acquired/released).

## Test Strategy
### Unit tests
- Mock subprocess invocation to assert `--agent=atomic_executor` and `--continue` are applied per task.
- Assert generated prompts exclude `.github/instructions/**/*.instructions.md` content.
- Assert prompt payload size/line-count guardrails are enforced (log output or returned metadata).
- Assert single-run guard behavior blocks concurrent runs.

### Integration checks
- Run atomic executor on a real plan and confirm prompt size reduction (target ≈10KB, no “file too large” warnings).
- Verify session continuity via CLI output or session logs (manual `/usage` check acceptable).
- Validate rollover heuristic triggers a new session when threshold is reached and logs the reason.


## Acceptance Criteria
- Conditions that must be true for the bug to be considered fixed (map to repro and edge cases).
- Prompt files no longer inline repository instruction files.
- Copilot CLI is invoked with `--agent=atomic_executor` for all tasks.
- `--continue` is used between tasks within a single plan run, with a single-run guard enforced.
- Prompt payloads are reduced to task-only context and stay within size guardrails.
- Session rollover behavior is implemented and documented with a permitted heuristic.
- Manual validation confirms session continuity (e.g., `/usage` or session logs).

## Risks & Mitigations
- Technical or operational risks:
- Risk: `--continue` resumes an unrelated session when concurrent runs exist.
- Risk: Lack of structured `/usage` output makes automated rollover imprecise.
- Risk: Agent/profile usage changes may alter behavior unexpectedly.
- Mitigations and rollbacks:
- Enforce single-run exclusivity (lock file or guard in executor).
- Document fallback rollover heuristics and allow manual `/usage` checks.
- Provide a feature flag or configuration fallback to previous behavior during rollout.

## Rollout & Follow-up
- Release/rollout steps:
- Release/rollout steps:
- Land changes behind a config switch if needed; default to new behavior after validation.
- Update feature docs and executor prompts to reflect agent/continuity flow.
- Post-fix monitoring or clean-up tasks:
- Monitor prompt sizes and session continuity in `.agent_logs/` for at least one full plan.
- Capture token usage before/after for a representative plan.
- Links: issue, PRs, related docs
- Issue: #87
