# copilot-cli-instructions-duplication (Issue #87)

- Date captured: 2026-01-16
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/copilot-cli-instructions-duplication/ (Issue #87)

> Automation note: Keep the section headings below unchanged; the promotion tooling maps each of them into the GitHub bug issue template.

- Issue: #87
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/87
- Last Updated: 2026-01-16
## Summary

The `atomic_executor` has multiple fundamental design flaws that cause excessive token usage, context window overruns, and wasted compute. The prompt builder inlines all repository instruction files into the prompt, even though Copilot CLI **automatically loads them natively**. Additionally, the executor creates a new session for every task, discarding valuable context and causing "bootstrap" steps to execute in sessions that are immediately abandoned.

## Environment

- OS/version: Linux (devcontainer)
- Python version: 3.13
- Command/flags used: `poetry run python -m scripts.dev_tools.atomic_executor.cli execute-all`
- Data source or fixture: Any feature plan under `docs/features/active/`

## Steps to Reproduce

1. Run the atomic executor on any feature plan
2. Observe the generated prompt file in `.agent_logs/prompts/`
3. Note that it contains ~3,400 lines (~179KB) of inlined instruction content
4. Observe that Copilot CLI starts a new session for each task (P0-T1, P0-T2, etc.)
5. Note that "read policy files" tasks execute in a session that is discarded before implementation begins

## Expected Behavior

1. Instructions should be loaded once by Copilot CLI's native mechanism
2. A single session should persist across all tasks in a plan until the context window reaches 90%. Only then should a new session be opened.
3. Context from reading files in Phase 0 should carry forward to implementation phases
4. Token usage should be minimal

## Actual Behavior

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

## Logs / Screenshots

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

## Impact / Severity

- [x] Blocker
- [ ] High
- [ ] Medium
- [ ] Low

**Justification**: 
- Token costs are exponentially higher than they should be (instructions loaded twice and no reused sessions)
- Context window is exhausted by duplicated instructions, leaving insufficient room for actual task context
- Session isolation defeats the purpose of multi-step plans
- "File too large to read" warnings occur because prompts are ~179KB when they should be ~10KB

## Suspected Cause / Notes

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

## Proposed Fix / Validation Ideas

### Fix 1: Stop Inlining Instructions

Remove the instruction concatenation from `prompt_builder.py`. The prompt should contain only:
- Current task ID and description
- Feature-specific context (plan excerpt, spec excerpt)
- Retry context (if applicable)

### Fix 2: Use `--agent=atomic_executor`

Modify `run_copilot()` in `cli.py` to pass:
```bash
copilot --agent=atomic_executor --prompt "Execute task [P1-T1] from @plan.md"
```

The agent profile already contains all execution rules, policy requirements, and behavioral instructions.

### Fix 3: Implement Session Continuity

Instead of starting a new session per task:
```bash
# First task
copilot --agent=atomic_executor --share session.md --prompt "Execute P0-T1"

# Subsequent tasks (resume the session)
copilot --agent=atomic_executor --session-path session.md --prompt "Execute P0-T2"
```

Or use `--continue` to resume the most recent session.

### Fix 4: Merge Bootstrap into Session Start

Remove Phase 0 as separate tasks. Instead, the first invocation should:
1. Start a session with `--agent=atomic_executor`
2. Let Copilot CLI auto-load instructions (native behavior)
3. Begin execution from Phase 1

Policy reading is implicit via Copilot CLI's auto-loading.

### Validation

- [ ] Unit tests: Mock `subprocess.run` to verify `--agent` flag is passed
- [ ] Unit tests: Verify no instruction content in generated prompts
- [ ] Unit tests: Verify `--session-path` or `--continue` used between tasks
- [ ] Integration: Run atomic_executor on a real plan and measure token usage (should drop ~50%+)
- [ ] Integration: Verify `/usage` shows context from previous tasks is retained

## Next Step

- [x] Promote to GitHub issue (bug-report template)
- [ ] Move to active fix folder / branch