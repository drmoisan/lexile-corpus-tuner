"""Tests for the Python rewrite of potential-to-issue tooling."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from scripts.dev_tools import potential_to_issue as mod

if TYPE_CHECKING:
    from collections.abc import Iterable


class FakeFileSystem(mod.FileSystem):
    def __init__(self) -> None:
        self.files: dict[Path, str] = {}
        self.dirs: set[Path] = set()
        self.moves: list[tuple[Path, Path]] = []

    def resolve_path(self, path_str: str) -> Path:
        return Path(path_str)

    def exists(self, path: Path) -> bool:
        return path in self.files

    def read_text(self, path: Path) -> str:
        return self.files[path]

    def write_text(self, path: Path, content: str) -> None:
        self.files[path] = content

    def write_lines(self, path: Path, lines: Iterable[str]) -> None:
        self.files[path] = "\n".join(lines)

    def ensure_dir(self, path: Path) -> None:
        self.dirs.add(path)

    def move(self, src: Path, dest: Path) -> None:
        if src not in self.files:
            raise FileNotFoundError(src)
        self.files[dest] = self.files[src]
        del self.files[src]
        self.moves.append((src, dest))
        self.dirs.add(dest.parent)


class FakeGhClient(mod.GhClient):
    def __init__(
        self,
        create_result: mod.GhResult,
        view_result: mod.GhResult | None = None,
        authenticated: bool = True,
    ) -> None:
        self.create_result = create_result
        self.view_result = view_result
        self.authenticated = authenticated
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def is_authenticated(self) -> bool:
        return self.authenticated

    def issue_create(self, title: str, body: str, promotion_type: str) -> mod.GhResult:
        self.calls.append(("create", (title, body, promotion_type)))
        return self.create_result

    def issue_view(self, issue_number: str) -> mod.GhResult:
        self.calls.append(("view", (issue_number,)))
        return self.view_result or mod.GhResult([], 0)


def test_get_feature_name_variants() -> None:
    assert (
        mod.get_feature_name("# My Feature Name\n## Section", Path("test.md"))
        == "My Feature Name"
    )
    assert mod.get_feature_name("# Feature (Potential)\n", Path("test.md")) == "Feature"
    assert mod.get_feature_name("No heading", Path("feature-name.md")) == "feature-name"
    assert mod.get_feature_name("No heading", Path("my-feature")) == "my-feature"
    assert (
        mod.get_feature_name("#   Feature Name (Potential)  \n", Path("test.md"))
        == "Feature Name"
    )
    assert (
        mod.get_feature_name("# First Feature\n## Second\n# Third", Path("test.md"))
        == "First Feature"
    )
    assert (
        mod.get_feature_name("# Bug Title (Potential Bug)\n", Path("test.md"))
        == "Bug Title"
    )


def test_get_feature_path_variants() -> None:
    assert mod.get_feature_path("My Feature Name") == "My_Feature_Name"
    assert mod.get_feature_path("Feature: (v2.0) @ Test!") == "Feature_v20__Test"
    assert mod.get_feature_path("Feature   Name") == "Feature_Name"
    assert mod.get_feature_path("my-feature-name") == "my-feature-name"
    assert mod.get_feature_path("Feature v2 Update") == "Feature_v2_Update"
    assert mod.get_feature_path("A") == "A"


def test_get_section_variants() -> None:
    content = "## Problem / Why\nabc\n## Proposed Behavior\ndef"
    assert mod.get_section(content, "Problem / Why") == "abc"

    multi_line = "## Problem / Why\nline1\nline2\nline3\n## Next Section\nother"
    assert mod.get_section(multi_line, "Problem / Why") == "line1\nline2\nline3"

    assert mod.get_section(content, "NonExistent") == ""

    end_section = "## Problem / Why\nabc\n## Last Section\nfinal content"
    assert mod.get_section(end_section, "Last Section") == "final content"

    trimmed = "## Problem / Why\n  abc  \n  def  \n## Next"
    assert mod.get_section(trimmed, "Problem / Why") == "abc  \n  def"

    special_heading = "## Acceptance Criteria (early draft)\ncontent here\n## Next"
    assert (
        mod.get_section(special_heading, "Acceptance Criteria (early draft)")
        == "content here"
    )

    empty_section = "## Problem / Why\n\n## Proposed Behavior\ndef"
    assert mod.get_section(empty_section, "Problem / Why") == ""

    windows_endings = "## Problem / Why\r\nabc\r\n## Proposed Behavior\r\ndef"
    assert mod.get_section(windows_endings, "Problem / Why") == "abc"


def test_promote_potential_success_updates_metadata_and_moves_file() -> None:
    workspace = Path("/workspace")
    potential = workspace / "docs/features/potential/sample.md"
    fs = FakeFileSystem()
    fs.files[potential] = "\n".join(
        [
            "# Feature Title",
            "## Problem / Why",
            "why",
            "## Proposed Behavior",
            "behave",
            "## Acceptance Criteria (early draft)",
            "criteria",
            "## Constraints & Risks",
            "risk",
            "## Test Conditions to Consider",
            "tests",
        ]
    )

    create_result = mod.GhResult(["Created: https://example.com/issues/123"], 0)
    view_result = mod.GhResult(
        [
            '{"number":123,"title":"t","url":"https://example.com/issues/123","author":{"login":"me"},"updatedAt":"2024-01-02T00:00:00Z"}',
        ],
        0,
    )
    gh = FakeGhClient(create_result, view_result)
    messages: list[str] = []

    outcome = mod.promote_potential(
        potential_path=str(potential),
        promotion_type="feature",
        fs=fs,
        gh=gh,
        workspace=workspace,
        emit=messages.append,
    )

    assert outcome.exit_code == 0
    assert outcome.destination is not None
    assert (
        outcome.destination == workspace / "docs/features/potential/promoted/sample.md"
    )
    assert len(gh.calls) == 2
    assert (potential, outcome.destination) in fs.moves

    promoted_content = fs.files[outcome.destination]
    lines = promoted_content.splitlines()
    assert lines[0] == "# Feature Title (Issue #123)"
    assert "- Issue: #123" in lines
    assert "- Issue URL: https://example.com/issues/123" in lines
    assert "- Last Updated: 2024-01-02" in lines
    assert (
        "- Status: Promoted -> docs/features/active/Feature_Title/ (Issue #123)"
        in lines
    )
    assert any(line.startswith("Moved potential file") for line in messages)


def test_promote_potential_failure_does_not_move_file() -> None:
    workspace = Path("/workspace")
    potential = workspace / "docs/features/potential/sample.md"
    fs = FakeFileSystem()
    original_content = "# Feature Title\n## Problem / Why\nwhy"
    fs.files[potential] = original_content

    create_result = mod.GhResult(["line1", "line2"], 1)
    gh = FakeGhClient(create_result)
    messages: list[str] = []

    outcome = mod.promote_potential(
        potential_path=str(potential),
        promotion_type="feature",
        fs=fs,
        gh=gh,
        workspace=workspace,
        emit=messages.append,
    )

    assert outcome.exit_code == 1
    assert fs.moves == []
    assert fs.files[potential] == original_content
    assert gh.calls
    verb, (title, body, label) = gh.calls[0]
    assert verb == "create"
    assert title == "Feature: Feature Title"
    assert label == "feature"
    assert "## Problem / Why\nwhy" in body
    assert "## Proposed Behavior\n(not provided in potential file)" in body
    assert "## Acceptance Criteria\n(not provided in potential file)" in body
    assert "## Constraints & Risks\n(not provided in potential file)" in body
    assert "## Test Conditions\n(not provided in potential file)" in body
    assert potential.relative_to(workspace).as_posix() in body
    assert "line1" in messages and "line2" in messages


def test_promote_potential_bug_builds_issue_body_from_bug_sections() -> None:
    workspace = Path("/workspace")
    potential = workspace / "docs/features/potential/sample-bug.md"
    fs = FakeFileSystem()
    fs.files[potential] = "\n".join(
        [
            "# Sample Bug (Potential Bug)",
            "## Summary",
            "summary details",
            "## Environment",
            "- OS: Linux",
            "## Steps to Reproduce",
            "1. step one",
            "## Expected Behavior",
            "expected results",
            "## Actual Behavior",
            "actual results",
            "## Impact / Severity",
            "medium",
            "## Logs / Screenshots",
            "screenshot attached",
        ]
    )

    create_result = mod.GhResult(["Created: https://example.com/issues/200"], 0)
    view_result = mod.GhResult(
        [
            '{"number":200,"title":"Bug title","url":"https://example.com/issues/200","author":{"login":"me"},"updatedAt":"2024-02-01T00:00:00Z"}',
        ],
        0,
    )
    gh = FakeGhClient(create_result, view_result)

    outcome = mod.promote_potential(
        potential_path=str(potential),
        promotion_type="bug",
        fs=fs,
        gh=gh,
        workspace=workspace,
    )

    assert outcome.exit_code == 0
    verb, (title, body, label) = gh.calls[0]
    assert verb == "create"
    assert title == "Bug: Sample Bug"
    assert label == "bug"
    assert "## Summary\nsummary details" in body
    assert "## Environment\n- OS: Linux" in body
    assert "## Steps to Reproduce\n1. step one" in body
    assert "## Expected Behavior\nexpected results" in body
    assert "## Actual Behavior\nactual results" in body
    assert "## Impact / Severity\nmedium" in body
    assert "## Logs / Screenshots\nscreenshot attached" in body
    assert "## Source\nFrom: docs/features/potential/sample-bug.md" in body


def test_promote_potential_bug_missing_sections_use_placeholders() -> None:
    workspace = Path("/workspace")
    potential = workspace / "docs/features/potential/placeholder-bug.md"
    fs = FakeFileSystem()
    fs.files[potential] = "\n".join(["# Placeholder Bug", "## Summary", "only summary"])

    create_result = mod.GhResult(["Created: https://example.com/issues/300"], 0)
    gh = FakeGhClient(create_result)

    mod.promote_potential(
        potential_path=str(potential),
        promotion_type="bug",
        fs=fs,
        gh=gh,
        workspace=workspace,
    )

    _, (_, body, _) = gh.calls[0]
    assert "## Summary\nonly summary" in body
    assert "## Environment\n(not provided in potential file)" in body
    assert "## Steps to Reproduce\n(not provided in potential file)" in body
    assert "## Expected Behavior\n(not provided in potential file)" in body
    assert "## Actual Behavior\n(not provided in potential file)" in body
    assert "## Impact / Severity\n(not provided in potential file)" in body
    assert "## Logs / Screenshots\n(not provided in potential file)" in body


def test_promote_potential_normalizes_smart_punctuation_in_issue_body_and_title() -> (
    None
):
    workspace = Path("/workspace")
    potential = workspace / "docs/features/potential/smart.md"
    fs = FakeFileSystem()
    fs.files[potential] = "\n".join(
        [
            "# “Curly” Feature (Potential Bug)",
            "## Summary",
            (
                "Title uses “smart” quotes and an en dash – "
                "plus non-breaking space\u00a0here."
            ),
            "## Environment",
            "- OS: “Windows”\u00a0",
            "## Steps to Reproduce",
            "1. step one with “quote”",
            "## Expected Behavior",
            "expected with em dash — and quote “",
            "## Actual Behavior",
            "actual with smart apostrophe ’",
            "## Impact / Severity",
            "medium",
            "## Logs / Screenshots",
            "none",
        ]
    )

    create_result = mod.GhResult(["Created: https://example.com/issues/400"], 0)
    view_result = mod.GhResult(
        [
            '{"number":400,"title":"t","url":"https://example.com/issues/400","author":{"login":"me"},"updatedAt":"2024-03-01T00:00:00Z"}',
        ],
        0,
    )
    gh = FakeGhClient(create_result, view_result)

    mod.promote_potential(
        potential_path=str(potential),
        promotion_type="bug",
        fs=fs,
        gh=gh,
        workspace=workspace,
    )

    verb, (title, body, label) = gh.calls[0]
    assert verb == "create"
    assert label == "bug"

    assert "Curly" in title and "“" not in title and "”" not in title
    assert "smart" in body and "“" not in body and "”" not in body
    assert "–" not in body and "—" not in body
    assert "\u00a0" not in body


def test_promote_potential_raises_on_missing_file() -> None:
    fs = FakeFileSystem()
    with pytest.raises(mod.PromotionError):
        mod.promote_potential(
            "/missing.md", fs=fs, gh=FakeGhClient(mod.GhResult([], 0))
        )


def test_promote_potential_rejects_invalid_promotion_type() -> None:
    fs = FakeFileSystem()
    invalid_path = Path("/workspace/tmp/file.md")
    fs.files[invalid_path] = "# Title"
    with pytest.raises(mod.PromotionError):
        mod.promote_potential(
            str(invalid_path),
            promotion_type="invalid",
            fs=fs,
            gh=FakeGhClient(mod.GhResult([], 0)),
        )


def test_promote_potential_checks_authentication_before_proceeding() -> None:
    """Verify that promotion succeeds when gh is authenticated."""
    content = (
        "# Test Feature (Potential)\n"
        "- Author: test\n"
        "- Date: 2024-01-01\n"
        "- Status: potential\n"
        "\n"
        "## Problem / Why\n"
        "Test problem\n"
        "\n"
        "## Proposed Behavior\n"
        "Test behavior\n"
        "\n"
        "## Acceptance Criteria (early draft)\n"
        "Test criteria\n"
        "\n"
        "## Constraints & Risks\n"
        "Test constraints\n"
        "\n"
        "## Test Conditions to Consider\n"
        "Test conditions\n"
    )

    fs = FakeFileSystem()
    potential_path = Path("docs/features/potential/test.md")
    fs.files[potential_path] = content

    create_result = mod.GhResult(["Created: https://example.com/issues/123"], 0)
    view_result = mod.GhResult(
        ['{\n  "number": 123,\n  "updatedAt": "2024-01-01T00:00:00Z"\n}'],
        0,
    )
    gh = FakeGhClient(create_result, view_result, authenticated=True)

    outcome = mod.promote_potential(
        str(potential_path),
        fs=fs,
        gh=gh,
        workspace=Path("/fake/workspace"),
    )

    assert outcome.exit_code == 0
    assert gh.calls[0][0] == "create"
    assert "Test problem" in gh.calls[0][1][1]


def test_promote_potential_fails_fast_when_not_authenticated() -> None:
    """Verify that promotion fails with clear message when gh is not authenticated."""
    content = "# Test Feature\n## Problem / Why\nTest problem\n"

    fs = FakeFileSystem()
    potential_path = Path("docs/features/potential/test.md")
    fs.files[potential_path] = content

    create_result = mod.GhResult(["should not be called"], 1)
    gh = FakeGhClient(create_result, authenticated=False)

    with pytest.raises(
        mod.PromotionError,
        match="GitHub CLI is not authenticated. Run 'gh auth login' first.",
    ):
        mod.promote_potential(
            str(potential_path),
            fs=fs,
            gh=gh,
            workspace=Path("/fake/workspace"),
        )

    assert (
        len(gh.calls) == 0
    ), "No gh commands should be executed when not authenticated"


def test_real_gh_client_invokes_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_which(name: str) -> str:
        return "/usr/bin/gh"

    class DummyCompleted:
        def __init__(self, code: int, stdout: str = "", stderr: str = "") -> None:
            self.returncode = code
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(args: list[str], **kwargs: object) -> DummyCompleted:
        calls.append(list(args))
        if "auth" in args:
            return DummyCompleted(0, "ok", "")
        return DummyCompleted(0, "output", "")

    monkeypatch.setattr(mod.shutil, "which", fake_which)
    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    client = mod.RealGhClient()
    assert client.is_authenticated() is True
    create = client.issue_create("Title", "Body", "feature")
    view = client.issue_view("5")
    assert create.exit_code == 0 and view.exit_code == 0
    assert any("issue" in call for call in calls)


def test_real_gh_client_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_: str) -> None:
        return None

    monkeypatch.setattr(mod.shutil, "which", missing)
    with pytest.raises(FileNotFoundError):
        mod.RealGhClient()


def test_real_filesystem_round_trip(tmp_path: Path) -> None:
    fs = mod.RealFileSystem()
    target = fs.resolve_path(str(tmp_path / "nested" / "file.txt"))
    fs.ensure_dir(target.parent)
    fs.write_text(target, "content")
    assert fs.read_text(target) == "content"
    dest = target.parent / "moved.txt"
    fs.move(target, dest)
    assert dest.exists()


def test_parse_args_and_main_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    potential = tmp_path / "p.md"
    args = [
        "prog",
        "--potential-path",
        str(potential),
        "--promotion-type",
        "epic",
    ]
    monkeypatch.setattr(sys, "argv", args)
    parsed = mod.parse_args()
    assert parsed.potential_path == str(potential)
    assert parsed.promotion_type == "epic"

    fake_args = SimpleNamespace(potential_path=str(potential), promotion_type="feature")

    def fake_parse_args() -> SimpleNamespace:
        return fake_args

    def fake_promote(potential_path: str, promotion_type: str) -> mod.PromotionOutcome:
        return mod.PromotionOutcome(0, [], None)

    monkeypatch.setattr(mod, "parse_args", fake_parse_args)
    monkeypatch.setattr(mod, "promote_potential", fake_promote)
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 0


def test_main_exits_on_promotion_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_args = SimpleNamespace(
        potential_path=str(tmp_path / "p.md"), promotion_type="feature"
    )

    def fake_parse_args() -> SimpleNamespace:
        return fake_args

    monkeypatch.setattr(mod, "parse_args", fake_parse_args)

    def raise_error(**kwargs: object) -> object:
        raise mod.PromotionError("boom")

    monkeypatch.setattr(mod, "promote_potential", raise_error)
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 1
