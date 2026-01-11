"""Typed seams for running GitHub Copilot CLI.

Purpose:
    The atomic executor needs deterministic unit tests that can simulate Copilot
    outcomes (success, throttling, and non-throttling failures) without invoking
    the real `copilot` binary.

    This module defines small, typed value objects and protocols that enable
    dependency injection around Copilot execution.

Notes:
    The concrete subprocess implementation currently lives in
    `scripts.dev_tools.atomic_executor.cli.run_copilot()`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CopilotRunResult:
    """Result of a single Copilot CLI invocation.

    Purpose:
        Capture enough information from one Copilot call to support:
        - throttling detection/classification
        - retry/backoff decisions
        - actionable error reporting

    Attributes:
        exit_code (int): Process exit code. Zero indicates success.
        output_tail (str): A bounded tail snippet of the Copilot output stream
            (used for classification and error messages).
    """

    exit_code: int
    output_tail: str
