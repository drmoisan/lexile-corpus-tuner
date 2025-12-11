"""Unit tests for the PowerShell PR context collection script."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "collect-pull-request-context.ps1"

pytestmark = pytest.mark.skipif(
    shutil.which("pwsh") is None,
    reason="PowerShell is required to validate the PR context helper",
)


def run_pwsh(expression: str) -> str:
    """Execute a PowerShell expression after dot-sourcing the script and return stdout."""
    command = [
        "pwsh",
        "-NoLogo",
        "-NoProfile",
        "-Command",
        f". '{SCRIPT_PATH}'; {expression}",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def test_format_diffpath_handles_git_rename_syntax() -> None:
    """Ensure Format-DiffPath normalizes both brace-style and simple rename paths."""
    brace_output = run_pwsh("Format-DiffPath 'dir/{old => new}/file.txt'")
    simple_output = run_pwsh("Format-DiffPath 'legacy => renamed.cs'")
    quoted_output = run_pwsh("Format-DiffPath '  \"docs/note.md\"  '" )

    assert brace_output == "dir/new/file.txt"
    assert simple_output == "renamed.cs"
    assert quoted_output == "docs/note.md"


def test_convert_from_numstat_sums_changes_and_files() -> None:
    """Verify ConvertFrom-Numstat aggregates additions, deletions, and file list."""
    numstat_text = """
10\t2\tsrc/module.py
5\t1\tdir/{old => new}/file.txt
-\t-\tREADME
"""
    expression = (
        "$text = @'" + numstat_text + "'@; "
        "ConvertFrom-Numstat -NumstatText $text | ConvertTo-Json -Compress"
    )
    payload = json.loads(run_pwsh(expression))

    assert payload["Additions"] == 15
    assert payload["Deletions"] == 3
    assert payload["Files"] == ["src/module.py", "dir/{old => new}/file.txt", "README"]


def test_get_extension_summary_counts_extensions() -> None:
    """Get-ExtensionSummary should bucket files by extension after path normalization."""
    expression = """
$files = @(
    'src/app.py',
    'dir/{old => new}/file.ts',
    'README',
    'docs/guide.md',
    'legacy => renamed.txt'
)
$summary = Get-ExtensionSummary -Files $files
$map = @{}
foreach ($line in ($summary -split "`n")) {
    if (-not [string]::IsNullOrWhiteSpace($line)) {
        $parts = $line -split '\\s+'
        $count = [int]$parts[0]
        $ext = $parts[-1]
        $map[$ext] = $count
    }
}
$map | ConvertTo-Json -Compress
"""
    extension_counts = json.loads(run_pwsh(expression))

    assert extension_counts == {
        "(noext)": 1,
        ".md": 1,
        ".py": 1,
        ".ts": 1,
        ".txt": 1,
    }


def test_get_issue_references_extracts_unique_ids() -> None:
    """Get-IssueReferences returns a sorted, de-duplicated list of issue tokens."""
    expression = (
        "$text = 'Fixes #12 and ABC-99; relate to #7 and #12 again'; "
        "Get-IssueReferences -Text $text | ConvertTo-Json -Compress"
    )
    issues = json.loads(run_pwsh(expression))

    assert issues == ["#12", "#7", "ABC-99"]

