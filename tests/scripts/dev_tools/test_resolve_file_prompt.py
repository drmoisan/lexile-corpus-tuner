"""Tests for scripts.dev_tools.resolve_file_prompt."""

import argparse
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.dev_tools.resolve_file_prompt import (
    build_context_injection,
    main,
    resolve_prompt,
    strip_front_matter,
)


def test_resolve_prompt_relative_path():
    """Test standard case where target is inside workspace."""
    template = "Plan for ${file} please."

    # Use real logic but controlled paths
    # We construct paths that assume the current CWD structure
    cwd = Path.cwd()
    target = cwd / "src" / "module.py"

    result = resolve_prompt(template, target, cwd)

    # We expect src/module.py (forward slashes)
    expected_fragment = "src/module.py"
    assert expected_fragment in result
    assert "${file}" not in result
    assert "\\" not in result


def test_resolve_prompt_outside_cwd():
    """Test fallback when target is not relative to CWD."""
    template = "Analyze ${file}"
    cwd = Path("/workspace/A")
    target = Path("/workspace/B/file.py")

    # We force relative_to to fail by using disjoint paths (on Unix)
    # or just mocking the call to ensure isolation from OS specifics.

    with patch("pathlib.Path.relative_to", side_effect=ValueError):
        result = resolve_prompt(template, target, cwd)

        # Should contain the full target path
        assert str(target).replace("\\", "/") in result


def test_resolve_prompt_forward_slashes():
    """Test that backslashes are converted to forward slashes."""
    template = "File: ${file}"
    cwd = Path.cwd()
    target = cwd / "subdir" / "file.py"

    result = resolve_prompt(template, target, cwd)

    # Verification
    assert "subdir/file.py" in result
    assert "\\" not in result


def test_strip_front_matter():
    """Test that YAML front matter is correctly stripped."""
    content = """---
agent: 'test'
description: 'test description'
---
# Main Content

This is the actual content."""

    result = strip_front_matter(content)

    assert "---" not in result
    assert "agent:" not in result
    assert "# Main Content" in result
    assert result.startswith("# Main Content")


def test_strip_front_matter_no_front_matter():
    """Test content without front matter passes through unchanged."""
    content = "# Regular content\n\nNo front matter here."

    result = strip_front_matter(content)

    assert result == content


def test_build_context_injection_for_plan_with_docs(tmp_path: Path):
    """Test context injection when spec.md and user-story.md exist."""
    # Create test directory structure
    feature_dir = tmp_path / "docs" / "features" / "active" / "test-feature"
    feature_dir.mkdir(parents=True)

    plan_path = feature_dir / "plan.md"
    spec_path = feature_dir / "spec.md"
    user_story_path = feature_dir / "user-story.md"

    plan_path.write_text("# Plan")
    spec_path.write_text("# Spec")
    user_story_path.write_text("# User Story")

    result = build_context_injection(plan_path)

    assert "## Authoritative Requirements" in result
    assert "spec.md" in result
    assert "user-story.md" in result
    assert "Technical specification" in result
    assert "User stories and acceptance criteria" in result


def test_build_context_injection_missing_docs(tmp_path: Path):
    """Test no context injection when spec.md or user-story.md missing."""
    feature_dir = tmp_path / "docs" / "features" / "active" / "test-feature"
    feature_dir.mkdir(parents=True)

    plan_path = feature_dir / "plan.md"
    plan_path.write_text("# Plan")

    result = build_context_injection(plan_path)

    assert result == ""


def test_build_context_injection_non_plan_file(tmp_path: Path):
    """Test no context injection for non-plan.md files."""
    feature_dir = tmp_path / "docs" / "features" / "active" / "test-feature"
    feature_dir.mkdir(parents=True)

    other_path = feature_dir / "spec.md"
    user_story_path = feature_dir / "user-story.md"

    other_path.write_text("# Spec")
    user_story_path.write_text("# User Story")

    result = build_context_injection(other_path)

    assert result == ""


def test_resolve_prompt_strips_front_matter():
    """Test that resolve_prompt strips front matter from template."""
    template = """---
agent: 'test'
---
# Content with ${file}"""

    cwd = Path.cwd()
    target = cwd / "test.md"

    result = resolve_prompt(template, target, cwd)

    assert "---" not in result
    assert "agent:" not in result
    assert "test.md" in result


@patch("scripts.dev_tools.resolve_file_prompt.pyperclip.copy")
@patch("pathlib.Path.read_text")
@patch("pathlib.Path.exists")
@patch("argparse.ArgumentParser.parse_args")
def test_main_success(
    mock_args: MagicMock,
    mock_exists: MagicMock,
    mock_read: MagicMock,
    mock_copy: MagicMock,
) -> None:
    """Test happy path for main."""
    mock_args.return_value = argparse.Namespace(
        template="prompt.md", target="src/main.py"
    )
    mock_exists.return_value = True
    mock_read.return_value = "Content with ${file}"

    # Mock sys.argv to avoid side effects if argparse falls back
    with patch.object(sys, "argv", ["prog"]):
        # Capture stdout
        captured = StringIO()
        with patch.object(sys, "stdout", captured):
            main()

    assert "Successfully resolved" in captured.getvalue()

    # Verify processing
    mock_copy.assert_called_once()
    copied_text = mock_copy.call_args[0][0]
    if isinstance(copied_text, str):
        # args.target is "src/main.py", which Path("src/main.py") usually resolves
        # relative to CWD.
        assert "src/main.py" in copied_text


@patch("argparse.ArgumentParser.parse_args")
def test_main_template_not_found(mock_args: MagicMock) -> None:
    """Test exit when template missing."""
    mock_args.return_value = argparse.Namespace(
        template="missing.md", target="src/main.py"
    )

    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(SystemExit) as exc:
            with patch.object(sys, "stderr", StringIO()):
                main()
        assert exc.value.code == 1


@patch("argparse.ArgumentParser.parse_args")
def test_main_exception_handling(mock_args: MagicMock) -> None:
    """Test generic exception catching."""
    mock_args.return_value = argparse.Namespace(
        template="prompt.md", target="src/main.py"
    )

    err = StringIO()
    with patch("pathlib.Path.exists", return_value=True):
        with patch("pathlib.Path.read_text", side_effect=RuntimeError("Disk error")):
            with pytest.raises(SystemExit) as exc:
                with patch.object(sys, "stderr", err):
                    main()
            assert exc.value.code == 1
            assert "Disk error" in err.getvalue()
