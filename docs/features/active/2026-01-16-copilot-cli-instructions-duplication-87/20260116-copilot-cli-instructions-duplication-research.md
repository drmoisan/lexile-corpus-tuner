<!-- markdownlint-disable-file -->

# Task Research Notes: Copilot CLI Instructions Duplication & Session Design

## Research Executed

### File Analysis

- /workspaces/lexile-corpus-tuner/docs/features/active/2026-01-16-copilot-cli-instructions-duplication-87/issue.md
  - Adds spec-level expectations: prompt size (~3,400 lines/~179KB), a 90% context rollover target, and Phase 0 bootstrap retention.
- /workspaces/lexile-corpus-tuner/docs/features/potential/promoted/2026-01-16-copilot-cli-instructions-duplication.md
  - Confirms four primary defects: instruction duplication, ignoring custom agent, no session continuity across tasks, and Phase 0 bootstrap executed in discarded sessions.
- /workspaces/lexile-corpus-tuner/scripts/dev_tools/atomic_executor/prompt_builder.py
  - Inlines `.github/copilot-instructions.md` and all `.github/instructions/*.instructions.md` into the prompt text, bloating each prompt.
- /workspaces/lexile-corpus-tuner/scripts/dev_tools/atomic_executor/cli.py
  - Uses `--session-path` only when retrying the same task. Each task otherwise opens a new Copilot CLI session.
- /workspaces/lexile-corpus-tuner/.github/agents/atomic_execution.agent.md
  - Repository already ships a custom agent profile (`atomic_executor`) with detailed execution rules.
- /workspaces/lexile-corpus-tuner/artifacts/research/20260115-copilot-cli-large-file-handling-research.md
  - Documents prompt file bloat (~3,423 lines) and `view` tool truncation at ~200 lines, showing instruction payloads crowd out task context.

### Code Search Results

- copilot-instructions|instructions\.md|AGENTS\.md|--session-path|--continue|--resume|--agent
  - Matches show prompt inlining in `prompt_builder.py` and session support confined to retry flow in `cli.py`.

### External Research

- #fetch:https://docs.github.com/en/copilot/how-tos/use-copilot-agents/use-copilot-cli
  - Copilot CLI automatically includes repository-wide instructions, path-specific instructions, and AGENTS.md.
  - Documents `--continue` and `--resume` for resuming interactive sessions.
  - Documents `/context` and `/usage` token usage readouts and auto-compression near 95% usage.
- #fetch:https://docs.github.com/en/copilot/concepts/agents/about-copilot-cli
  - Defines programmatic mode (`-p`) as a single-prompt invocation.
- #fetch:https://docs.github.com/en/copilot/how-tos/configure-custom-instructions/add-repository-instructions
  - Defines repo-wide instructions (`.github/copilot-instructions.md`), path-specific instructions (`.github/instructions/**/*.instructions.md`), and agent instructions (`AGENTS.md`, `CLAUDE.md`, `GEMINI.md`).
- #fetch:https://docs.github.com/en/copilot/reference/custom-instructions-support
  - Confirms Copilot CLI support for repository-wide instructions, path-specific instructions, and AGENTS.md.
- #fetch:https://docs.github.com/en/copilot/how-tos/use-copilot-agents/coding-agent/create-custom-agents
  - Confirms custom agents can be invoked in Copilot CLI via `--agent`.
- #fetch:https://docs.github.com/en/copilot/reference/custom-agents-configuration
  - Defines `.agent.md` frontmatter and `tools`, `target`, `infer` behavior for agents.
- #fetch:https://docs.github.com/en/copilot/concepts/agents/coding-agent/about-custom-agents
  - Describes repository-level agent profiles under `.github/agents` and usage in Copilot CLI.
- #fetch:https://docs.github.com/en/copilot/concepts/prompting/response-customization
  - Explains instruction precedence and combined inclusion for repository instructions.
- #fetch:https://docs.github.com/en/copilot/concepts/agents/coding-agent/about-hooks
  - Hooks can run at session start/end and can be used with Copilot CLI.
- #fetch:https://docs.github.com/en/copilot/reference/hooks-configuration
  - Confirms sessionStart/sessionEnd hooks and input/output formats for CLI use.
- #fetch:https://docs.github.com/en/copilot/how-tos/use-copilot-agents/coding-agent/use-hooks
  - Confirms Copilot CLI loads hooks from `.github/hooks/` in the current working directory.
- #fetch:https://docs.github.com/en/copilot/how-tos/use-copilot-agents/coding-agent/track-copilot-sessions
  - Session logs show token usage and session length, useful for manual continuity validation.
- #fetch:https://docs.github.com/en/copilot/concepts/agents/copilot-memory
  - Copilot Memory is used by Copilot CLI and can reduce repeated context injection when enabled.
- #fetch:https://docs.github.com/en/copilot/how-tos/set-up/install-copilot-cli
  - Confirms CLI installation paths and programmatic mode usage.
- #fetch:https://github.com/github/copilot-cli/issues/442
  - Feature request for `--start-session` in programmatic mode; notes lack of programmatic session ID creation.
- #fetch:https://github.com/github/copilot-cli/issues/52
  - JSON output request; discussion notes need for session ID to make `--resume <session-id>` practical in tooling.
- #fetch:https://github.com/github/copilot-cli/issues/51
  - Shows `--additional-mcp-config` exists; helps explain workarounds for session-scope configuration.
- #fetch:https://github.com/github/copilot-cli/issues/287
  - Notes CLI prerelease workarounds; cited by third-party tooling when integrating Copilot CLI.
- #fetch:https://github.com/undoio/addons/pull/79
  - Third-party integration notes: uses temporary state directories and `--resume` to approximate session continuity.
- #fetch:https://github.com/endorhq/rover/pull/233
  - Third-party integration notes: Copilot CLI lacks JSON output; requires prompt discipline/parsing.
- #fetch:https://github.com/gabrypavanello/ralph-wiggum-cli/commit/3ac01c08ff3aba9c959128621bafa3c493788d34
  - Confirms Copilot CLI does not support `--output-format stream-json` yet; uses `--resume` when session ID is known.
- #fetch:https://github.com/openai/agents.md
  - AGENTS.md format concept and examples; aligns with agent instruction files.
- #fetch:https://github.com/agentskills/agentskills
  - Agent Skills standard referenced by GitHub for reusable task-specific context.
- #fetch:https://github.com/anthropics/skills
  - Skills examples and SKILL.md structure that can supplement instructions when needed.
- #fetch:https://github.com/github/awesome-copilot
  - Community repository of agents and instructions; demonstrates standard `.agent.md` usage.
- #fetch:https://gh.io/dpa
  - Copilot CLI preview data protection reference noted in GitHub docs.

### Project Conventions

- Standards referenced: `.github/instructions/*.instructions.md`, `.github/copilot-instructions.md`, `AGENTS.md`.
- Instructions followed: Repository policies require explicit compliance with instruction files and toolchain order.

## Key Discoveries

### Project Structure

- `prompt_builder.py` inlines all instruction files into every prompt.
- `cli.py` uses `-p` with an `@prompt_file`, causing large prompt files per task.
- `atomic_execution.agent.md` already encodes the workflow and policies expected of the executor.
- Prompt files reach ~3,423 lines (~179KB), while the `view` tool truncates around 200 lines, preventing later content from being seen.

### Implementation Patterns

- Prompts are regenerated per task with full instruction concatenation.
- Sessions are isolated per task; `--session-path` is used only for retry attempts.
- Agent profiles exist but are not invoked via `--agent`.
- Copilot CLI programmatic mode (`-p`) is documented as a single prompt invocation; session continuity is only documented for interactive sessions via `--continue`/`--resume`.
- Copilot CLI auto-compresses history near 95% token usage and warns when <20% remains (via `/context` and `/usage`).
- There is no documented programmatic “start session and return session ID” API; issue #442 requests this capability and confirms current limitations.
- `--continue` resumes the most recent local session and can collide with other Copilot CLI runs in the same state directory.
- Copilot CLI does not provide structured JSON output for `/usage` or `/context`, limiting automation-friendly token checks.

### Complete Examples

```python
# prompt_builder.py: inlining instructions into every prompt
instructions: list[tuple[str, str]] = []
copilot_instructions_path = self.workspace / ".github" / "copilot-instructions.md"
if self._fs.is_file(copilot_instructions_path):
    instructions.append(
        ("copilot-instructions.md", self._read_text(copilot_instructions_path))
    )

instructions_dir = self.workspace / ".github" / "instructions"
if self._fs.is_dir(instructions_dir):
    for instruction_file in self._fs.glob(instructions_dir, "*.instructions.md"):
        instructions.append(
            (instruction_file.name, self._read_text(instruction_file))
        )
```

```python
# cli.py: session reuse only for retries, not across tasks
supports_sessions = _copilot_supports_session(copilot_exe)
if resume_session and supports_sessions:
    argv.extend(["--session-path", str(share_path)])

argv.extend(
    [
        "--share",
        str(share_path),
        "--add-dir",
        str(workspace),
        "-p",
        f"Follow these instructions exactly: @{prompt_file}",
    ]
)
```

### API and Schema Documentation

- Copilot CLI automatically includes repository instructions, path-specific instructions, and AGENTS.md.
- Custom agents are defined as `.agent.md` files and can be invoked via `--agent=<name>`.
- Session reuse is documented via `--resume` and `--continue`, with `-p` described as a single-prompt mode.
- No documented mechanism exists to create a new session and programmatically return its session ID.
- Hook system allows sessionStart/sessionEnd scripts for CLI workflows.
- Instruction precedence combines repository-wide and path-specific instructions; avoids duplicating them in prompts.
- `/context` and `/usage` provide token usage readouts in interactive sessions; auto-compression begins near 95% usage.

### Configuration Examples

```bash
# Copilot CLI programmatic mode with a custom agent
copilot --agent=atomic_executor -p "Execute task P1-T1 from plan.md"

# Session resumption behavior
copilot --continue

# Manual context check (interactive mode)
copilot
/context
/usage

# Custom instructions supported by CLI
# .github/copilot-instructions.md
# .github/instructions/**/*.instructions.md
# AGENTS.md
```

### Technical Requirements

- Do not inline repository instructions into prompts when Copilot CLI is used (instructions are auto-included).
- Use the repository-level agent profile via `--agent=atomic_executor`.
- Persist CLI sessions across tasks using `--continue`, which reuses the last active session state without requiring a programmatic session ID.
- Treat Phase 0 “bootstrap” as session-level initialization, not per-task initialization.
- Define a session rollover rule that aligns with the 90% context target using `/context` or a documented fallback heuristic.
- Enforce single-run exclusivity to avoid `--continue` resuming an unrelated session.
- Add prompt-size guardrails (target ≈10KB task payload; remove instruction inlining) to avoid “file too large” warnings.
- Include validation checkpoints using session logs or `/usage` to confirm continuity across tasks.

**Not achievable as stated (requires fallback/approximation):**
- **Exact automated “90% context” rollover is not currently achievable in programmatic mode.** The CLI exposes `/context` and `/usage` only via interactive commands and does not provide structured output or a programmatic session ID. As a result, the spec must permit a fallback (manual `/usage` validation, task-count caps, or prompt-size heuristics) instead of a precise automated 90% threshold.

## Recommended Approach

Adopt Copilot CLI’s native instruction and agent mechanisms, then refactor atomic executor to use `--continue` with explicit rollover and prompt-size guardrails. This eliminates duplication, reduces prompt size, and keeps Phase 0 context available across tasks.

Core actions:
- Stop inlining `.github/copilot-instructions.md` and `.github/instructions/**/*.instructions.md` in prompt generation.
- Invoke the `.github/agents/atomic_execution.agent.md` profile using `--agent=atomic_executor`.
- Reuse session context across tasks using `--continue` so the CLI carries forward the active session state.
- Reduce prompt payloads to task-specific context only (task ID + feature context excerpts).
- Add continuity guardrails (single-run lock, documented rollover heuristic).

## Implementation Guidance

- **Objectives**: Remove instruction duplication, use custom agent, maintain session continuity with safe rollover, and make Phase 0 meaningful.
- **Key Tasks**:
  - Remove instruction concatenation in `prompt_builder.py`.
  - Add `--agent=atomic_executor` in `run_copilot()` invocation.
  - Use `--continue` when running multiple tasks in a single session to keep state without a session ID.
  - Update prompt template to include only task-specific context.
  - Define context-usage measurement or a fallback heuristic to trigger new sessions near 90% usage.
  - Add a single-run guard to avoid `--continue` resuming unrelated sessions.
  - Document the unachievable exact 90% automated rollover and the approved fallback path.
- **Dependencies**: Copilot CLI supports instruction auto-loading, `--agent`, and interactive session resumption via `--continue`. Programmatic session ID creation is not documented.
- **Success Criteria**:
  - Prompt files no longer contain `.github/instructions/*.instructions.md` content.
  - CLI is invoked with `--agent=atomic_executor`.
  - A single session spans all tasks in a plan via `--continue` (verified by session logs).
  - Token usage and prompt size are reduced materially.
  - Session rollover behavior is defined and validated using `/usage` or session logs.