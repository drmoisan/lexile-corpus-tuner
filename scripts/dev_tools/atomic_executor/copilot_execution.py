"""
Copilot CLI execution and configuration helpers.

Purpose:
    Encapsulates GitHub Copilot CLI invocation, configuration, session management,
    and output streaming for the atomic executor.

Responsibilities:
    - Execute Copilot CLI with proper permissions and configuration
    - Manage trusted workspace configuration
    - Stream and capture Copilot output with timeout handling
    - Clean session files for reuse

Side Effects:
    - Executes subprocess (copilot CLI)
    - Writes to filesystem (logs, session files, config)
    - Modifies Copilot CLI configuration (~/.config/github-copilot/)
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import cast

from scripts.dev_tools.atomic_executor.arg_parser import (
    DEFAULT_COPILOT_ALLOW_ALL_PATHS,
    DEFAULT_COPILOT_ALLOW_ALL_URLS,
    DEFAULT_COPILOT_ALLOW_SHELL,
    DEFAULT_COPILOT_TRUST_WORKSPACE,
)
from scripts.dev_tools.atomic_executor.copilot_runner import CopilotRunResult

DEFAULT_COPILOT_AGENT = "atomic_execution"

# When Copilot CLI cannot request approval (common in headless/non-interactive
# runs), it emits this exact substring and may then stall until an idle-timeout.
# We detect it during output streaming and fail fast with actionable guidance.
COPILOT_PERMISSION_DENIED_SUBSTRING = (
    "Permission denied and could not request permission from user"
)


class CopilotPermissionDeniedError(RuntimeError):
    """
    Raised when Copilot output indicates an approval/permission dead-end.

    Purpose:
        Provides a typed signal that the Copilot CLI emitted the known
        permission-denied substring and is unlikely to recover without
        additional permissions or an interactive approval path.
    """


def run_copilot(
    *,
    workspace: Path,
    prompt_text: str,
    log_file: Path,
    task_id: str,
    preferred_model: str | None,
    run_id: str,
    resume_session: bool = False,
    is_first_task: bool = True,
    allow_all_paths: bool = DEFAULT_COPILOT_ALLOW_ALL_PATHS,
    allow_all_urls: bool = DEFAULT_COPILOT_ALLOW_ALL_URLS,
    allow_shell: bool = DEFAULT_COPILOT_ALLOW_SHELL,
    trust_workspace: bool = DEFAULT_COPILOT_TRUST_WORKSPACE,
    _idle_timeout_seconds: float | None = None,
    _output_tail_bytes: int | None = None,
) -> CopilotRunResult:
    """
    Invoke GitHub Copilot CLI with prompt and tool permissions.

    Purpose:
        Executes the Copilot CLI command with specified prompt and permissions,
        capturing output to log file and returning bounded tail for diagnostics.

    Args:
        workspace (Path): Repository root directory.
        prompt_text (str): Complete prompt text to execute.
        log_file (Path): Path for writing Copilot output log.
        task_id (str): Current task identifier for log labeling.
        preferred_model (str | None): Preferred model name or None for default.
        run_id (str): Run identifier for grouping artifacts.
        resume_session (bool): Reuse prior Copilot session if True.
        is_first_task (bool): True for first task in plan run.
        allow_all_paths (bool): Allow all path access without approvals.
        allow_all_urls (bool): Allow all URL access without approvals.
        allow_shell (bool): Allow all shell commands without approvals.
        trust_workspace (bool): Persist workspace in trusted folders.
        _idle_timeout_seconds (float | None): Idle timeout override for testing.
        _output_tail_bytes (int | None): Output tail size override for testing.

    Returns:
        CopilotRunResult: Exit code and bounded output tail.

    Raises:
        FileNotFoundError: Copilot CLI executable not found.
        TimeoutError: Copilot produces no output for idle timeout period.

    Side Effects:
        - Executes copilot CLI subprocess
        - Writes to log file
        - Creates per-task session share markdown file
        - May modify Copilot CLI configuration for trusted workspaces
    """
    output_tail_bytes = 4096 if _output_tail_bytes is None else _output_tail_bytes
    if output_tail_bytes < 0:
        output_tail_bytes = 0

    idle_timeout_seconds = _resolve_idle_timeout_seconds(_idle_timeout_seconds)

    # Validate copilot executable exists and is not VS Code shim.
    # Validate copilot executable exists.
    copilot_exe = os.getenv("COPILOT_CLI_PATH", "copilot")
    resolved_exe = shutil.which(copilot_exe)

    if not resolved_exe:
        msg = f"Required executable not found on PATH: {copilot_exe}"
        raise FileNotFoundError(msg)

    # Reject VS Code integrated terminal shim (causes hang).
    # VS Code stores shim at paths like:
    # - .../Code/User/globalStorage/github.copilot-chat/copilotCli/
    # - ~/.vscode-server/data/User/globalStorage/github.copilot-chat/
    resolved_lower = resolved_exe.lower()
    if (
        "vscode-server" in resolved_lower
        or "vscode_server" in resolved_lower
        or "github.copilot-chat" in resolved_lower
    ):
        # VS Code shim detected - cannot use it
        msg = f"Required executable not found on PATH: {copilot_exe}"
        raise FileNotFoundError(msg)

    # Ensure workspace is trusted if requested.
    if trust_workspace:
        _ensure_trusted_workspace(workspace=workspace)

    # Build command with agent identifier and permissions.
    cmd = [resolved_exe, "--agent", DEFAULT_COPILOT_AGENT]

    # Add model preference if specified.
    if preferred_model:
        cmd.extend(["--model", preferred_model])

    # Add tool permissions (must come before --add-dir and file access flags)
    cmd.extend(
        [
            "--add-dir",
            str(workspace),
            "--allow-tool",
            "write",
            "--allow-tool",
            "shell(poetry)",
            "--allow-tool",
            "shell(python)",
            "--allow-tool",
            "shell(python3)",
            "--allow-tool",
            "shell(git)",
        ]
    )

    # Expand permissions for headless sessions when explicitly enabled.
    if allow_shell:
        cmd.extend(["--allow-tool", "shell"])

    # Add permission flags.
    if allow_all_paths:
        cmd.append("--allow-all-paths")
    if allow_all_urls:
        cmd.append("--allow-all-urls")

    # Write prompt to file (avoids Windows command-line length limits).
    prompt_dir = log_file.parent / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = prompt_dir / f"prompt_{run_id}_{task_id}.md"
    prompt_file.write_text(prompt_text, encoding="utf-8")

    # Pass prompt via -p flag with @file reference
    cmd.extend(["-p", f"Follow these instructions exactly: @{prompt_file}"])

    # Session reuse logic.
    # --continue resumes the most recent session. Used for both multi-task
    # continuation and explicit resume after executor restart.
    # --share file is always used for persistent context across tasks.
    session_path = workspace / ".agent_logs" / "sessions" / f"{run_id}_{task_id}.md"

    if resume_session or not is_first_task:
        # Add --continue for session continuation
        cmd.append("--continue")

    # Always use --share for persistent context
    session_path.parent.mkdir(parents=True, exist_ok=True)
    if not session_path.exists():
        session_path.write_text(f"# Session for {task_id}\n\n{prompt_text}\n")
    cmd.extend(["--share", str(session_path)])

    # Ensure log directory exists.
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Log command for diagnostics.
    log_msg(log_file, f"[{task_id}] Running: {' '.join(cmd)}\n")

    # Execute with streaming output capture.
    try:
        proc = subprocess.Popen(  # noqa: S603 - static analysis can't verify runtime validation
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,  # Handle bytes for encoding detection
            cwd=workspace,
        )
    except PermissionError as exc:
        msg = (
            f"Permission denied executing Copilot CLI: {resolved_exe}\n"
            "Ensure the executable has execute permissions.\n"
            f"Try: chmod +x {resolved_exe}"
        )
        raise PermissionError(msg) from exc

    # Stream output to log file with idle timeout.
    tail_buffer = bytearray()
    try:
        _stream_copilot_output(
            proc=proc,
            log_file=log_file,
            task_id=task_id,
            tail_buffer=tail_buffer,
            output_tail_bytes=output_tail_bytes,
            idle_timeout_seconds=idle_timeout_seconds,
        )
    except CopilotPermissionDeniedError:
        # Fail fast with actionable context instead of waiting for idle-timeout.
        argv_summary = " ".join(cmd)
        raise RuntimeError(
            "Copilot CLI reported a permissions dead-end and cannot request "
            "approval from the user in this environment. "
            f"Detected: {COPILOT_PERMISSION_DENIED_SUBSTRING!r}. "
            f"argv: {argv_summary}. "
            "Guidance: ensure the executor uses programmatic mode (-p/--prompt) "
            "and includes explicit tool and directory permissions "
            "(e.g. --allow-tool write, --allow-tool shell(poetry), "
            "--allow-tool shell(python3), --allow-tool shell(git), "
            "--allow-tool shell, --allow-all-paths, "
            "and --add-dir <workspace>). "
            "If policy blocks headless execution, run the command interactively to "
            "grant approvals."
        ) from None
    except TimeoutError:
        proc.kill()
        proc.wait()
        raise

    exit_code = proc.wait()

    # Decode tail buffer for result.
    tail_text = ""
    if tail_buffer:
        try:
            tail_text = tail_buffer.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - CLI top-level error handling
            tail_text = "(failed to decode output tail)"

    # Post-processing: deduplicate prompt from session file.
    _clean_session_file(session_path, prompt_text)

    return CopilotRunResult(exit_code=exit_code, output_tail=tail_text)


def _resolve_idle_timeout_seconds(configured: float | None) -> float | None:
    """
    Resolve idle timeout from config or environment variable.

    Purpose:
        Provides override mechanism for Copilot idle timeout via environment
        variable, supporting test scenarios and production configuration.

    Args:
        configured (float | None): Configured timeout value or None.

    Returns:
        float | None: Resolved timeout in seconds, or None for no timeout.

    Side Effects:
        Reads ATOMIC_EXECUTOR_COPILOT_IDLE_TIMEOUT_SECONDS environment variable.
    """
    env_timeout = os.getenv("ATOMIC_EXECUTOR_COPILOT_IDLE_TIMEOUT_SECONDS")
    if env_timeout is not None:
        try:
            return float(env_timeout)
        except ValueError:
            pass
    return configured


def _copilot_config_dir() -> Path:
    """
    Resolve GitHub Copilot CLI configuration directory.

    Purpose:
        Provides the configuration directory path for reading/writing Copilot
        CLI settings, respecting XDG conventions when available.

    Returns:
        Path: Configuration directory path (XDG_CONFIG_HOME/copilot or ~/.copilot).

    Side Effects:
        None (read-only path resolution).
    """
    xdg_home = os.environ.get("XDG_CONFIG_HOME")
    # Prefer XDG config location when explicitly configured.
    if xdg_home and xdg_home.strip():
        return Path(xdg_home).expanduser().resolve() / "copilot"

    return Path.home() / ".copilot"


def _ensure_trusted_workspace(*, workspace: Path) -> None:
    """
    Add workspace to Copilot CLI trusted folders if not present.

    Purpose:
        Ensures the workspace is in Copilot's trusted folders list, avoiding
        repeated trust prompts during execution.

    Args:
        workspace (Path): Repository root to trust.

    Returns:
        None

    Raises:
        RuntimeError: When the Copilot CLI config.json is malformed.

    Side Effects:
        - Reads config.json configuration file
        - Writes updated configuration if workspace not trusted
        - Creates config directory if missing
    """
    config_dir = _copilot_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.json"

    # Read existing config.
    config_data: dict[str, object] = {}
    if config_file.exists():
        try:
            config_data = json.loads(config_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Copilot CLI config.json is invalid JSON. "
                f"Fix or remove: {config_file}"
            ) from exc

    # Check if workspace already trusted.
    trusted_folders = config_data.get("trusted_folders")
    # Normalize trusted_folders into a list for safe updates.
    if trusted_folders is None:
        trusted_folders_list: list[str] = []
    elif isinstance(trusted_folders, list):
        # Normalize trusted folder entries to strings for stable comparisons.
        trusted_folders_list = [
            str(item) for item in cast(list[object], trusted_folders)
        ]
    else:
        raise RuntimeError(
            "Copilot CLI config.json has non-list trusted_folders. "
            f"Fix: {config_file}"
        )

    workspace_path = str(workspace.resolve())
    # Only append when the workspace is not already trusted.
    if workspace_path not in trusted_folders_list:
        trusted_folders_list.append(workspace_path)
        config_data["trusted_folders"] = trusted_folders_list
        config_file.write_text(json.dumps(config_data, indent=2))


def _stream_copilot_output(
    *,
    proc: subprocess.Popen[bytes],
    log_file: Path,
    task_id: str,
    tail_buffer: bytearray,
    output_tail_bytes: int,
    idle_timeout_seconds: float | None,
) -> None:
    """
    Stream Copilot subprocess output to log file with idle timeout detection.

    Purpose:
        Captures Copilot CLI output incrementally, writing to log file while
        maintaining a bounded tail buffer and detecting idle hangs.

    Args:
        proc (subprocess.Popen[bytes]): Running Copilot subprocess.
        log_file (Path): Log file for writing output.
        task_id (str): Task identifier for log messages.
        tail_buffer (bytearray): Mutable buffer for output tail (updated in-place).
        output_tail_bytes (int): Maximum tail buffer size.
        idle_timeout_seconds (float | None): Timeout for no output, None for unlimited.

    Returns:
        None

    Raises:
        TimeoutError: No output received within idle timeout period.

    Side Effects:
        - Writes to log file
        - Updates tail_buffer in-place
        - Logs timeout warnings
    """
    last_output_time = time.time()
    output_queue: queue.Queue[bytes | None] = queue.Queue()

    def _read_output() -> None:
        """Read subprocess stdout and enqueue chunks."""
        if proc.stdout is None:
            return
        try:
            while True:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                output_queue.put(chunk)
        finally:
            output_queue.put(None)  # EOF sentinel

    reader_thread = threading.Thread(target=_read_output, daemon=True)
    reader_thread.start()

    with log_file.open("ab") as log_fh:
        while True:
            if idle_timeout_seconds is not None:
                elapsed = time.time() - last_output_time
                if elapsed > idle_timeout_seconds:
                    timeout_msg = (
                        f"[{task_id}] ERROR: Copilot idle {elapsed:.1f}s "
                        f"(timeout: {idle_timeout_seconds}s)\n"
                    )
                    log_msg(log_file, timeout_msg)
                    msg = f"Copilot CLI idle timeout after {elapsed:.1f} seconds"
                    raise TimeoutError(msg)

            # Use short queue timeout to allow frequent idle checking
            queue_timeout = (
                min(0.1, idle_timeout_seconds / 2) if idle_timeout_seconds else 0.5
            )
            try:
                chunk = output_queue.get(timeout=queue_timeout)
            except queue.Empty:
                # No output available - check if process ended
                if proc.poll() is not None:
                    # Process ended - drain remaining queue
                    while not output_queue.empty():
                        try:
                            chunk = output_queue.get_nowait()
                            if chunk is None:
                                break
                            log_fh.write(chunk)
                            log_fh.flush()
                            _update_tail_buffer(tail_buffer, chunk, output_tail_bytes)
                        except queue.Empty:
                            break
                    break
                # Process still running - check timeout on next iteration
                continue

            if chunk is None:
                # EOF reached - check if we got ANY output before exiting
                if not tail_buffer and idle_timeout_seconds is not None:
                    # No output received - wait full timeout before declaring failure
                    elapsed = time.time() - last_output_time
                    if elapsed < idle_timeout_seconds:
                        # EOF came too quickly - sleep remainder and recheck
                        time.sleep(idle_timeout_seconds - elapsed + 0.1)
                        elapsed = time.time() - last_output_time
                    if elapsed > idle_timeout_seconds:
                        timeout_msg = (
                            f"[{task_id}] ERROR: Copilot produced no output "
                            f"(timeout: {idle_timeout_seconds}s)\n"
                        )
                        log_msg(log_file, timeout_msg)
                        msg = f"Copilot CLI idle timeout after {elapsed:.1f} seconds"
                        raise TimeoutError(msg)
                break

            log_fh.write(chunk)
            log_fh.flush()
            _update_tail_buffer(tail_buffer, chunk, output_tail_bytes)

            # Check for permission denied pattern in accumulated output.
            # Decode the tail buffer to check for the substring pattern.
            try:
                tail_text = tail_buffer.decode("utf-8", errors="replace")
                if COPILOT_PERMISSION_DENIED_SUBSTRING in tail_text:
                    # Fail fast - kill process and raise typed exception.
                    proc.kill()
                    proc.wait()
                    raise CopilotPermissionDeniedError(
                        COPILOT_PERMISSION_DENIED_SUBSTRING
                    )
            except UnicodeDecodeError:
                pass  # Incomplete bytes - check again on next chunk

            last_output_time = time.time()


def _update_tail_buffer(tail_buffer: bytearray, chunk: bytes, max_bytes: int) -> None:
    """Update bounded tail buffer with new chunk."""
    if max_bytes == 0:
        return
    tail_buffer.extend(chunk)
    if len(tail_buffer) > max_bytes:
        tail_buffer[:] = tail_buffer[-max_bytes:]


def _clean_session_file(session_path: Path, prompt_text: str) -> None:
    """Clean Copilot session file for reuse by removing old messages."""
    if not session_path.exists():
        return

    content = session_path.read_text(encoding="utf-8", errors="replace")
    lines = content.split("\n")
    header_lines: list[str] = []

    for line in lines:
        if line.strip().startswith(("##", "**User:**", "**Copilot:**")):
            break
        header_lines.append(line)

    cleaned = "\n".join(header_lines).strip()
    session_path.write_text(f"{cleaned}\n\n{prompt_text}\n")


def log_msg(log_file: Path, msg: str) -> None:
    """Append message to log file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(msg)
