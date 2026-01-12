<!-- markdownlint-disable-file -->

# Task Research Notes: copilot-cli-permission-failure-83

## Research Executed

### File Analysis

- /workspaces/lexile-corpus-tuner/docs/features/active/2026-01-11-copilot-cli-permission-failure-83/issue.md
  - Captures that Copilot CLI (invoked by the atomic executor) cannot execute even basic commands (e.g., `poetry --version`, `python3 --version`) and reports: “Permission denied and could not request permission from user”.
  - Confirms atomic executor later kills Copilot after an idle timeout (default 300s) and cites the env override `ATOMIC_EXECUTOR_COPILOT_IDLE_TIMEOUT_SECONDS`.

- /workspaces/lexile-corpus-tuner/scripts/dev_tools/atomic_executor/cli.py
  - `run_copilot()` constructs the Copilot CLI argv with `--allow-tool write` plus `--allow-tool shell(poetry)`, `shell(python)`, `shell(git)`.
  - The prompt is written to a file and passed as `stdin` to the Copilot process (`stdin=prompt_f`) to avoid Windows command-line length limits.
  - This design implies the Copilot process has *no interactive stdin available* for approval prompts (EOF once the prompt file is consumed).

- /workspaces/lexile-corpus-tuner/.agent_logs/copilot_sessions/copilot_session_2026-01-11_002907_P1-T1.md
  - Confirms Copilot attempted to run shell commands during the session (e.g., `poetry --version`) and the tool execution failed with:
    - `Permission denied and could not request permission from user`

- /workspaces/lexile-corpus-tuner/.devcontainer/local/devcontainer.json
  - Devcontainer runs as `remoteUser: vscode`; default integrated terminal profile is `pwsh`.
  - Workspace is a Docker *volume mount* at `/workspaces/lexile-corpus-tuner`.

- /workspaces/lexile-corpus-tuner/.devcontainer/local/Dockerfile
  - Installs GitHub Copilot CLI using the official install script pinned to `0.0.377` (via `https://gh.io/copilot-install`, installed into `/usr/local`).

### Local Experiments (devcontainer)

- Confirmed Copilot CLI version and prompt-mode behavior (Copilot CLI 0.0.377)
  - `copilot -p` works non-interactively with `--silent`.
  - `copilot -p` can run tools in non-interactive mode when explicitly allowed via `--allow-tool`.
  - Prompt text can include `@relative/or/absolute/path` references that Copilot expands into file contents in `-p` mode.

Evidence (captured terminal output):

- `copilot -p "Reply with exactly the contents of @artifacts/research/copilot-at-mention-sentinel.txt. Do not add any other text." --silent`
  - Output: `COPILOT_AT_MENTION_SENTINEL`

- `copilot -p "Run python3 --version using the shell tool and reply with exactly the command output." --allow-tool 'shell(python3)' --silent`
  - Output: `Python 3.13.9`

- `copilot -p "Reply with exactly: OK" --silent`
  - Output: `OK`

- Negative result: piping what look like argv tokens into `copilot` did not behave like “stdin-as-argv”
  - `"--help" | copilot` and `"--version" | copilot` remained running until forcibly stopped (no output after 2s).
  - This suggests the GitHub Docs “pipe options into copilot” example is either incomplete/ambiguous, version-dependent, or uses different semantics than naive token piping.

### Code Search Results

- --allow-tool
  - Found in `scripts/dev_tools/atomic_executor/cli.py` and asserted in `tests/scripts/dev_tools/atomic_executor/test_cli.py`.

- ATOMIC_EXECUTOR_COPILOT_IDLE_TIMEOUT_SECONDS
  - Found in `scripts/dev_tools/atomic_executor/cli.py` and `tests/scripts/dev_tools/atomic_executor/test_cli.py`.

- Permission denied and could not request permission
  - Found in `issue.md`, `spec.md`, `.agent_logs/atomic_executor_2026-01-11_002907.log`, and `.agent_logs/copilot_sessions/*`.

### External Research

- #githubRepo:"github/copilot-cli script-outputting-options|pipe this to copilot|stdin options|parse stdin as args|non-interactive options"
  - Search results did not surface additional implementation-level documentation for the “pipe options into copilot” feature beyond what is already described in GitHub Docs.
  - Returned excerpts were primarily from `changelog.md` and `README.md`.

- #fetch:https://docs.github.com/en/copilot/concepts/agents/about-copilot-cli
  - Confirms Copilot CLI has two modes:
    - Interactive mode: `copilot` (expects interactive approvals)
    - Programmatic mode: `copilot -p/--prompt "..."`
  - Confirms a documented alternative to passing options directly via argv:
    - “use a script to output command-line options and pipe this to `copilot`”
    - Example shown in docs: `echo ./script-outputting-options.sh | copilot`
  - Confirms “Trusted directories” prompt and persistence in `~/.copilot/config.json` via `trusted_folders`.
  - Confirms “Allowed tools” approval flags and syntax:
    - `--allow-all-tools`
    - `--allow-tool 'shell(COMMAND)'` (and that `shell` is optional and can allow all shell commands)
    - `--deny-tool ...`
  - Notes that automatic approvals enable headless operation but increase risk.

- #fetch:https://docs.github.com/en/copilot/how-tos/use-copilot-agents/use-copilot-cli
  - Confirms Copilot will ask for tool approvals in interactive sessions when it wants to run tools (examples include `touch`, `chmod`, `node`, `sed`).
  - Documents additional permission layers beyond tool approval:
    - Path permissions (`--allow-all-paths`)
    - URL permissions (`--allow-all-urls`, `--allow-url <domain>`)
  - Confirms Copilot CLI config is stored by default under `~/.copilot/` and can be relocated with `XDG_CONFIG_HOME`.

- #fetch:https://raw.githubusercontent.com/github/copilot-cli/main/changelog.md
  - Confirms `-p/--prompt` mode is actively maintained for headless-style operation, including:
    - “`copilot -p` will no longer interactively prompt for permission requests” (0.0.359).
    - “Fix file operations timing out while waiting for user permission” (0.0.360).
    - “Added a prompt to approve new paths in `-p` mode. Also added `--allow-all-paths` argument…” (0.0.340).
    - “MCP servers work in `--prompt` mode” (0.0.363).

- #fetch:https://raw.githubusercontent.com/github/copilot-cli/main/README.md
  - Confirms platform support is presented as Linux/macOS/Windows.
  - Does not document the “pipe options into copilot” mechanism (that detail appears in GitHub Docs, not the repo README).

- #fetch:https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-createprocessw
  - Confirms the Windows `CreateProcessW` maximum command line length is 32,767 characters (including the terminating null).

- #fetch:https://learn.microsoft.com/en-us/troubleshoot/windows-client/shell-experience/command-line-string-limitation
  - Confirms `cmd.exe` command lines are limited to 8,191 characters.
  - Documents the canonical workaround: provide parameters via a file instead of a long command line.

### Project Conventions

- Standards referenced: repo policies in `.github/instructions/*` (no code changes performed in this research-only phase).
- Instructions followed: Task Researcher mode (write-only in `artifacts/research/`).

## Key Discoveries

### Project Structure

- The atomic executor delegates “agentic edits” to GitHub Copilot CLI (`copilot` binary installed in the devcontainer) and then runs scoped QC itself via subprocess (Poetry + Black/Ruff/Pyright/Pytest).
- The failure occurs *inside the Copilot CLI session* when Copilot tries to execute shell commands; this is distinct from local subprocess failures.

### Implementation Patterns

- `run_copilot()` is intentionally cross-platform and avoids passing large prompts via argv (Windows command-line limit) by:
  - writing the prompt to a file
  - setting `stdin` to that file

This also means:

- Copilot CLI cannot solicit interactive approvals because stdin is not a TTY (and will be EOF once the prompt file is fully read).
- The observed error string (“Permission denied and could not request permission from user”) is consistent with a tool-approval or permission prompt that cannot be displayed/answered in a non-interactive environment.

Additional internal evidence that atomic executor is *not* using programmatic mode:

- `tests/scripts/dev_tools/atomic_executor/test_cli.py` explicitly asserts `"-p" not in captured_argv`, i.e. the Copilot CLI invocation is expected to omit `-p/--prompt`.
- `scripts/dev_tools/atomic_executor/cli.py` contains comments indicating `-p` was a known prior approach, but was avoided due to Windows argv limits:
  - `WinError 206: ... when prompt passed via -p`.

### Complete Examples

```bash
# From GitHub Docs: programmatic mode requires --prompt (-p) and may require
# approval flags for headless operation.
copilot -p "Show me this week's commits and summarize them" --allow-tool 'shell(git)'
```

Source: GitHub Docs “About GitHub Copilot CLI” (#fetch:https://docs.github.com/en/copilot/concepts/agents/about-copilot-cli)

### API and Schema Documentation

- Copilot CLI approval option syntax (doc-verified):
  - `--allow-tool 'shell(COMMAND)'`
  - `--allow-tool 'shell'` (allow all shell commands)
  - `--allow-tool 'write'` (allow file modifications)
  - `--deny-tool ...` (takes precedence)

Source: GitHub Docs “About GitHub Copilot CLI” (#fetch:https://docs.github.com/en/copilot/concepts/agents/about-copilot-cli)

### Configuration Examples

```json
{
  "trusted_folders": [
    "/workspaces/lexile-corpus-tuner"
  ]
}
```

Source: GitHub Docs “About GitHub Copilot CLI” (trusted directories; config stored by default in `~/.copilot/config.json`).

### Technical Requirements

- To run autonomously, Copilot CLI must not require interactive approval prompts.
- Achieving “no prompts” may require configuration/flags in *multiple layers*:
  - trusted directory (trusted_folders)
  - tool approvals (`--allow-tool ...` or `--allow-all-tools`)
  - path approvals (`--allow-all-paths`), if Copilot touches files outside CWD
  - url approvals (`--allow-all-urls` / `--allow-url`), if Copilot uses `web_fetch` or network commands

- The atomic executor must avoid Windows command-line length failure modes when supplying long prompts:
  - `cmd.exe` imposes an 8,191 character limit.
  - Underlying Windows process creation (`CreateProcessW`) allows up to 32,767 characters, but very large prompts can still exceed it.

## Recommended Approach

User selected path: **switch the atomic executor to Copilot CLI programmatic mode on ALL operating systems**, while retaining a Windows-safe mechanism for very large prompts and minimizing OS-branching.

### Why the current behavior is *not* programmatic mode

GitHub Docs define programmatic mode as explicitly passing the prompt via `-p/--prompt`.
The atomic executor currently invokes `copilot` without `-p` and provides input via `stdin` from a prompt file.

This is supported by internal comments and tests:

- `cli.py` comment: prompt passed via `stdin` to avoid `WinError 206` that occurred when prompt was passed via `-p`.
- `test_cli.py` enforces: `"-p" not in captured_argv`.

So, while the atomic executor run is non-interactive, it is not “programmatic mode” as documented.

### Recommended implementation direction (cross-platform, minimal OS branching)

Prefer a single invocation strategy that works the same way on Linux/macOS/Windows:

1) **Always use Copilot CLI programmatic mode (`-p/--prompt`)**.

2) **Avoid sending the full prompt via argv** (Windows command-line limits). Use a prompt *file* and reference it from the `-p` prompt using an `@path` mention.

This keeps argv small while still switching to programmatic mode everywhere:

- Atomic executor continues to generate the full prompt text in a file (as it does today).
- Copilot is invoked in programmatic mode with a short prompt that “includes” that file:
  - Example prompt: `"Follow these instructions exactly: @<prompt_file_path>"`

Why this is the recommended direction:

- It directly addresses the root cause (not using `-p` today).
- It avoids the Windows argv length problem without requiring OS-conditional logic.
- It was validated locally (Copilot CLI 0.0.377) that `@...` file mentions expand in `-p` mode and that allowed tools can run under `-p`.

Status of the GitHub Docs “pipe options into copilot” mechanism:

- The docs mention a stdin-piped alternative, but local experiments (0.0.377) showed that piping `--help`, `--version`, or a script path into `copilot` did not behave as “options ingestion” (the process remained running until stopped).
- Given this mismatch and the availability of the `@file`-mention strategy, the “pipe options” mechanism should not be relied upon for the atomic executor.

### Approvals that are likely required for headless operation

Per GitHub Docs, in programmatic mode “to allow Copilot to modify and execute files you should also use one of the approval options.”

Minimum approvals typically needed for this repo’s usage:

- Trusted folder: add `/workspaces/lexile-corpus-tuner` to `trusted_folders` in `~/.copilot/config.json`.
- Tool approvals:
  - keep `--allow-tool write`
  - allow shell commands at least for the tools the agent needs to run for QC (Poetry, Python, Git).
  - consider whether `--allow-tool shell` (all shell commands) is required to prevent surprise prompts for secondary tools, then use `--deny-tool` to block dangerous commands.

Important: Copilot CLI also has *path* and *URL* permission layers (documented in “Using Copilot CLI”), which can still trigger approval prompts even when tools are allowed.
If Copilot is touching files outside the CWD (or using network commands), you may need to include `--allow-all-paths` and/or `--allow-all-urls` (or narrower allowlist flags).

Also note the Copilot CLI changelog indicates recent improvements to `-p` behavior around permissions and timeouts (for example, `-p` no longer prompting interactively for permission requests), which strengthens the case that `-p` mode is the intended basis for headless runs.

## Implementation Guidance

- **Objectives**:
  - Ensure Copilot CLI can execute required shell commands during atomic executor runs without interactive approvals.
  - Prevent idle-timeout hangs caused by Copilot waiting for prompts it cannot answer.

- **Key Tasks**:
  - Move to Copilot CLI programmatic mode (`-p/--prompt`) for all operating systems.
  - Implement a Windows-safe (and generally safe) large-prompt delivery mechanism that avoids argv length constraints:
    - Prefer the GitHub-doc-supported “pipe options into copilot” approach.
    - Keep the existing “prompt file” as the source of truth for the full prompt body.
  - Ensure trusted directory state is set for the workspace path.
  - Ensure approval flags cover the tools/paths/urls Copilot will use during atomic tasks.

- **Dependencies**:
  - GitHub Copilot CLI (installed via `https://gh.io/copilot-install`, pinned `0.0.377`).
  - Copilot CLI config file location (default `~/.copilot/config.json`, relocatable via `XDG_CONFIG_HOME`).

- **Success Criteria**:
  - Running atomic executor tasks that require tool execution results in Copilot successfully running (at minimum) `poetry --version` and `poetry run pytest ...` without “Permission denied and could not request permission from user”.
  - No idle-timeout termination due to waiting on approvals.

