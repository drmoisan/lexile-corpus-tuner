"""Status board helpers for fix-all output."""

from __future__ import annotations

import ctypes
import sys
from typing import Protocol, TextIO, cast


class Kernel32Api(Protocol):
    """Typed protocol for the subset of Kernel32 APIs used for VT enablement."""

    def GetStdHandle(self, n_std_handle: int) -> int: ...

    def GetConsoleMode(self, handle: int, mode: object) -> int: ...

    def SetConsoleMode(self, handle: int, mode: int) -> int: ...


def format_status_transition_line(branch: str, status: str) -> str:
    """
    Format a non-interactive status transition line.

    Purpose:
        Provide a deterministic, line-oriented status update for CI or redirected
        output streams.

    Args:
        branch (str): Branch name to include in the output line.
        status (str): Status string to include in the output line.

    Returns:
        str: A formatted status line using the required STATUS|... template.

    Raises:
        ValueError: Raised if branch or status is empty.

    Side Effects:
        None. Pure formatting function.
    """
    if not branch:
        raise ValueError("branch cannot be empty.")
    if not status:
        raise ValueError("status cannot be empty.")
    return f"STATUS|branch={branch}|status={status}"


def render_status_board(lines: list[str], *, width: int) -> str:
    """
    Render a fixed-height status board for interactive terminals.

    Purpose:
        Produce deterministic board text with one line per branch for in-place
        redraws in interactive terminals.

    Args:
        lines (list[str]): Preformatted status lines to render.
        width (int): Target board width for padding or truncation decisions.

    Returns:
        str: Rendered board text with one newline per line and a trailing newline.

    Raises:
        ValueError: Raised when width is not positive.

    Side Effects:
        None. Pure rendering function.
    """
    if width <= 0:
        raise ValueError("width must be positive.")

    if not lines:
        # Return empty output to avoid trailing newline for empty boards.
        return ""

    rendered_lines: list[str] = []
    # Pad or trim each line to keep the board width stable between redraws.
    for line in lines:
        if len(line) > width:
            rendered_lines.append(line[:width])
        else:
            rendered_lines.append(line.ljust(width))
    return "\n".join(rendered_lines) + "\n"


def format_ansi_redraw(board: str, *, line_count: int) -> str:
    """
    Format an ANSI redraw payload using erase-line and cursor-up sequences.

    Purpose:
        Build a deterministic ANSI redraw string that rewrites a fixed-height
        status board without emitting unsupported control sequences.

    Args:
        board (str): Rendered board content to write.
        line_count (int): Number of lines in the board to move the cursor up.

    Returns:
        str: ANSI redraw payload using only erase-line and cursor-up sequences.

    Raises:
        ValueError: Raised when line_count is negative.

    Side Effects:
        None. Pure formatting function.
    """
    if line_count < 0:
        raise ValueError("line_count cannot be negative.")

    output_parts: list[str] = []
    if line_count:
        output_parts.append("\x1b[1A" * line_count)
    # Clear each line before writing to avoid leftover characters from prior redraws.
    for line in board.splitlines():
        output_parts.append(f"\x1b[2K\r{line}\n")
    return "".join(output_parts)


def should_use_interactive_board(*, isatty: bool, vt_enabled: bool) -> bool:
    """
    Decide whether interactive status rendering should be used.

    Purpose:
        Gate terminal redraw behavior on TTY availability and VT support.

    Args:
        isatty (bool): Whether the output stream is a TTY.
        vt_enabled (bool): Whether VT/ANSI sequences are supported.

    Returns:
        bool: True when interactive rendering should be enabled.

    Raises:
        None.

    Side Effects:
        None. Pure decision function.
    """
    return isatty and vt_enabled


def is_vt_enabled_for_stream(stream: TextIO) -> bool:
    """
    Determine whether VT/ANSI support is enabled for the provided stream.

    Purpose:
        Enable Windows VT processing when possible and report whether ANSI
        sequences should be used for interactive rendering.

    Args:
        stream (TextIO): Stream to evaluate for VT support.

    Returns:
        bool: True when VT/ANSI sequences are supported for the stream.

    Raises:
        None.

    Side Effects:
        On Windows, attempts to enable VT processing for the console handle.
    """
    if not sys.platform.startswith("win"):
        return True

    from ctypes import wintypes

    enable_virtual_terminal_processing = 0x0004
    enable_processed_output = 0x0001
    std_output_handle = -11

    windll = getattr(ctypes, "windll", None)
    if windll is None:
        return False

    kernel32 = cast(Kernel32Api, windll.kernel32)
    handle = kernel32.GetStdHandle(std_output_handle)
    if handle in (0, -1):
        return False

    mode = wintypes.DWORD()
    # On Windows, enable VT processing when a console mode is available.
    if kernel32.GetConsoleMode(handle, ctypes.byref(mode)) == 0:
        return False

    new_mode = mode.value | enable_virtual_terminal_processing | enable_processed_output
    if kernel32.SetConsoleMode(handle, new_mode) == 0:
        return False
    return True
