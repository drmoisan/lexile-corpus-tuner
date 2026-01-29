"""
Clipboard helpers for atomic executor.

Purpose:
    Platform-aware clipboard command detection and copy functionality
    with WSL support and pyperclip fallback.

Usage:
    from scripts.dev_tools.atomic_executor.clipboard_helpers import (
        get_clipboard_command,
        copy_to_clipboard,
    )
"""

from __future__ import annotations

import shutil
import subprocess
import sys


def get_clipboard_command() -> list[str] | None:
    """
    Detect the correct clipboard command for the current platform.

    Purpose:
        Platform-aware clipboard command detection with WSL support.
        Tries multiple clipboard backends in priority order based on
        the detected platform.

    Returns:
        list[str] | None: Command and arguments list if available
            (e.g., ["xclip", "-selection", "clipboard"]),
            or None if no clipboard support detected.

    Side Effects:
        None - pure detection function that only checks PATH and
        /proc/version for WSL detection.
    """
    # Detect platform
    if sys.platform == "win32":
        candidates: list[list[str]] = [["clip"]]
    elif sys.platform == "darwin":
        candidates = [["pbcopy"]]
    else:  # Linux/Unix
        # Check for WSL (reports linux but needs Windows clipboard)
        is_wsl = False
        try:
            with open("/proc/version") as f:
                if "microsoft" in f.read().lower():
                    is_wsl = True
        except FileNotFoundError:
            pass

        if is_wsl:
            candidates = [
                ["clip.exe"],  # WSL prefers Windows clipboard
                ["pbcopy"],  # Fallback if macOS tools installed
                ["wl-copy"],  # Wayland
                ["xclip", "-selection", "clipboard"],  # X11
                ["xsel", "--clipboard", "--input"],  # X11 alternative
            ]
        else:
            candidates = [
                ["wl-copy"],  # Wayland
                ["xclip", "-selection", "clipboard"],  # X11
                ["xsel", "--clipboard", "--input"],  # X11 alternative
            ]

    # Validate candidates exist on PATH
    for cmd in candidates:
        if shutil.which(cmd[0]):
            return cmd

    return None


def copy_to_clipboard(text: str) -> bool:
    """
    Copy text to system clipboard using platform-appropriate command.

    Purpose:
        Provides clipboard access via explicit platform detection + validation.
        Tries pyperclip first (if available), then falls back to
        platform-specific command-line tools.

    Args:
        text (str): Text to copy to clipboard.

    Returns:
        bool: True if copy succeeded, False if no clipboard command
            available or copy operation failed.

    Side Effects:
        Executes system clipboard command (clip/pbcopy/xclip/etc.)
        or calls pyperclip.copy() if pyperclip is installed.
    """

    def _try_pyperclip_copy() -> bool:
        """
        Attempt copy via optional pyperclip dependency.

        Purpose:
            Tries to use pyperclip for clipboard access before falling
            back to platform-specific commands.

        Returns:
            bool: True when pyperclip is available and succeeds,
                False to allow fallback to platform-specific commands.
        """
        try:
            import pyperclip  # type: ignore[import-untyped]
        except ImportError:
            return False

        try:
            pyperclip.copy(text)
            return True
        except Exception:
            return False

    if _try_pyperclip_copy():
        return True

    # Get platform-appropriate clipboard command
    cmd = get_clipboard_command()
    if not cmd:
        return False

    # Execute clipboard command with validation
    exe = shutil.which(cmd[0])
    if not exe:
        return False

    try:
        subprocess.run(  # noqa: S603 - static analysis can't verify runtime validation
            [exe, *cmd[1:]],
            input=text,
            text=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False
