"""Unit tests for copy_research_to_issue dev tool.

These tests avoid filesystem writes by using an in-memory filesystem adapter,
matching repo policy (no temp files / no runtime file creation).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts.dev_tools import copy_research_to_issue as mod
from scripts.dev_tools import tk_dialog_helpers as tk


@dataclass
class FakeFileSystem(mod.FileSystem):
    """Minimal in-memory filesystem for deterministic unit tests."""

    existing: set[Path]
    copies: list[tuple[Path, Path]]

    def exists(self, path: Path) -> bool:
        return path in self.existing

    def copy_file(self, src: Path, dest: Path) -> None:
        self.copies.append((src, dest))
        self.existing.add(dest)

    def resolve_path(self, path_str: str) -> Path:
        return Path(path_str)


def test_resolve_issue_parent_dir_file_returns_parent() -> None:
    issue_file = Path("/repo/docs/features/active/issue-1/plan.md")
    assert mod.resolve_issue_parent_dir(issue_file) == Path(
        "/repo/docs/features/active/issue-1"
    )


def test_build_destination_path_always_research_md() -> None:
    issue_dir = Path("/repo/docs/features/active/issue-1")
    assert mod.build_destination_path(issue_dir) == issue_dir / "research.md"


def test_copy_research_document_copies_to_issue_parent_dir() -> None:
    fs = FakeFileSystem(existing=set(), copies=[])
    research = Path("/repo/docs/research/foo.md")
    issue_file = Path("/repo/docs/features/active/issue-1/plan.md")
    fs.existing.add(research)

    dest = mod.copy_research_document(
        fs=fs, research_path=research, issue_path=issue_file, overwrite=False
    )

    assert dest == Path("/repo/docs/features/active/issue-1/research.md")
    assert fs.copies == [(research, dest)]


def test_copy_research_document_missing_source_raises() -> None:
    fs = FakeFileSystem(existing=set(), copies=[])
    research = Path("/repo/docs/research/missing.md")
    issue_file = Path("/repo/docs/features/active/issue-1/plan.md")

    with pytest.raises(FileNotFoundError):
        mod.copy_research_document(
            fs=fs, research_path=research, issue_path=issue_file, overwrite=False
        )


def test_copy_research_document_existing_dest_without_overwrite_raises() -> None:
    fs = FakeFileSystem(existing=set(), copies=[])
    research = Path("/repo/docs/research/foo.md")
    issue_file = Path("/repo/docs/features/active/issue-1/plan.md")
    dest = Path("/repo/docs/features/active/issue-1/research.md")
    fs.existing.update({research, dest})

    with pytest.raises(FileExistsError):
        mod.copy_research_document(
            fs=fs, research_path=research, issue_path=issue_file, overwrite=False
        )

    assert fs.copies == []


def test_copy_research_document_existing_dest_with_overwrite_copies() -> None:
    fs = FakeFileSystem(existing=set(), copies=[])
    research = Path("/repo/docs/research/foo.md")
    issue_file = Path("/repo/docs/features/active/issue-1/plan.md")
    dest = Path("/repo/docs/features/active/issue-1/research.md")
    fs.existing.update({research, dest})

    result = mod.copy_research_document(
        fs=fs, research_path=research, issue_path=issue_file, overwrite=True
    )

    assert result == dest
    assert fs.copies == [(research, dest)]


def test_compute_tk_scaling_defaults_to_96dpi_when_unknown() -> None:
    scaling = tk.compute_tk_scaling(
        env={},
        logical_dpi=None,
        screen_px=None,
        screen_mm=None,
    )
    expected = 96.0 / 72.0
    assert abs(scaling - expected) < 1e-9


def test_compute_tk_scaling_uses_physical_dpi_when_much_higher() -> None:
    # 3840 px wide, 344 mm wide (~283 DPI). This should override a bogus 96 DPI.
    scaling = tk.compute_tk_scaling(
        env={},
        logical_dpi=96.0,
        screen_px=3840,
        screen_mm=344,
    )
    assert scaling > (96.0 / 72.0) * 2


def test_compute_tk_scaling_applies_gdk_scale_when_tk_unscaled() -> None:
    scaling = tk.compute_tk_scaling(
        env={"GDK_SCALE": "2"},
        logical_dpi=96.0,
        screen_px=None,
        screen_mm=None,
    )
    expected = (96.0 / 72.0) * 2
    assert abs(scaling - expected) < 1e-9


def test_compute_tk_scaling_respects_explicit_override_multiplier() -> None:
    scaling = tk.compute_tk_scaling(
        env={"LEXILE_TK_SCALE": "1.5"},
        logical_dpi=96.0,
        screen_px=None,
        screen_mm=None,
    )
    expected = (96.0 / 72.0) * 1.5
    assert abs(scaling - expected) < 1e-9
