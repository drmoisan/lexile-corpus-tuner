"""Tests for markdown_label_formatter."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.dev_tools import markdown_label_formatter
from scripts.dev_tools.markdown_label_formatter import process_markdown


def test_formats_first_label_with_content() -> None:
    """Label on first line becomes heading and content is spaced."""
    input_text = "User: Hello world"
    expected = "\n".join(["# User:", "", "", "> Hello world"])

    assert process_markdown(input_text) == expected


def test_inserts_separator_before_noninitial_label() -> None:
    """Labels not on the first line receive a separator block above them."""
    input_text = "\n".join(["Intro line", "User: Hi there"])
    expected = "\n".join(
        ["> Intro line", "", "---", "", "# User:", "", "", "> Hi there"]
    )

    assert process_markdown(input_text) == expected


def test_multiple_labels_and_quotes_other_lines() -> None:
    """Non-label lines are quoted and separators precede subsequent labels."""
    input_text = "\n".join(
        [
            "User: First message",
            "Second line from user",
            "GitHub Copilot: Reply text",
            "Follow up line",
        ]
    )
    expected = "\n".join(
        [
            "# User:",
            "",
            "",
            "> First message",
            "> Second line from user",
            "",
            "---",
            "",
            "# GitHub Copilot:",
            "",
            "",
            "> Reply text",
            "> Follow up line",
        ]
    )

    assert process_markdown(input_text) == expected


def test_label_without_inline_text_skips_extra_spacing() -> None:
    """Labels without inline text omit the double-blank insertion."""
    input_text = "\n".join(["Intro", "User:", "Next line"])
    expected = "\n".join(["> Intro", "", "---", "", "# User:", "> Next line"])

    assert process_markdown(input_text) == expected


def test_preserves_trailing_newline() -> None:
    """Trailing newline is preserved in the formatted output."""
    input_text = "User: Hi\n"
    expected = "\n".join(["# User:", "", "", "> Hi", ""])

    assert process_markdown(input_text) == expected


def test_handles_whitespace_separator() -> None:
    """Test that whitespace-only separator is preserved as blank line."""
    input_text = "User: text\n   \nmore content"
    result = process_markdown(input_text)
    # Whitespace-only lines are treated as separators and become blank
    assert "\n\n" in result


# Tests for helper functions


def test_is_label_line_recognizes_user() -> None:
    """Test that 'User:' prefix is recognized as a label."""
    assert markdown_label_formatter.is_label_line("User: some text")


def test_is_label_line_recognizes_copilot() -> None:
    """Test that 'GitHub Copilot:' prefix is recognized as a label."""
    assert markdown_label_formatter.is_label_line("GitHub Copilot: response")


def test_is_label_line_rejects_non_label() -> None:
    """Test that regular text is not recognized as a label."""
    assert not markdown_label_formatter.is_label_line("Just some text")


def test_is_separator_line_blank() -> None:
    """Test that blank line is recognized as separator."""
    assert markdown_label_formatter.is_separator_line("")


def test_is_separator_line_separator_token() -> None:
    """Test that '---' is recognized as separator."""
    assert markdown_label_formatter.is_separator_line("---")


def test_is_separator_line_rejects_content() -> None:
    """Test that content line is not recognized as separator."""
    assert not markdown_label_formatter.is_separator_line("Some content")


def test_format_label_heading_user() -> None:
    """Test formatting of User: label into heading."""
    heading, trailing = markdown_label_formatter.format_label_heading(
        "User: hello world"
    )
    assert heading == "# User:"
    assert trailing == "hello world"


def test_format_label_heading_no_trailing() -> None:
    """Test formatting when there is no text after the label."""
    heading, trailing = markdown_label_formatter.format_label_heading("User:")
    assert heading == "# User:"
    assert trailing == ""


def test_prefix_content_line_with_text() -> None:
    """Test that non-empty line gets quote prefix."""
    result = markdown_label_formatter.prefix_content_line("content text")
    assert result == "> content text"


def test_prefix_content_line_empty() -> None:
    """Test that empty line gets single quote marker."""
    result = markdown_label_formatter.prefix_content_line("")
    assert result == ">"


def test_ensure_separator_block_adds_separator() -> None:
    """Test that separator block is added to output."""
    lines = ["content"]
    markdown_label_formatter.ensure_separator_block(lines)
    assert lines == ["content", "", "---", ""]


def test_ensure_separator_block_removes_trailing_blanks() -> None:
    """Test that trailing blank lines are removed before adding separator."""
    lines = ["content", "", ""]
    markdown_label_formatter.ensure_separator_block(lines)
    assert lines == ["content", "", "---", ""]


# Tests for I/O functions


def test_read_content_from_file() -> None:
    """Test reading content from a file path."""
    mock_path = Mock(spec=Path)
    mock_path.read_text.return_value = "file content"

    result = markdown_label_formatter.read_content(mock_path)

    assert result == "file content"
    mock_path.read_text.assert_called_once_with(encoding="utf-8")


def test_read_content_from_stdin() -> None:
    """Test reading content from stdin when path is None."""
    with patch("sys.stdin", io.StringIO("stdin content")):
        result = markdown_label_formatter.read_content(None)
        assert result == "stdin content"


def test_write_output_to_file() -> None:
    """Test writing content to a file path."""
    mock_path = Mock(spec=Path)
    content = "output content"

    markdown_label_formatter.write_output(content, mock_path)

    mock_path.write_text.assert_called_once_with(content, encoding="utf-8")


def test_write_output_to_stdout() -> None:
    """Test writing content to stdout when path is None."""
    with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        markdown_label_formatter.write_output("stdout content", None)
        assert mock_stdout.getvalue() == "stdout content"


# Tests for CLI argument parsing


def test_parse_args_with_path() -> None:
    """Test parsing of positional path argument."""
    args = markdown_label_formatter.parse_args(["input.md"])
    assert args.path == Path("input.md")
    assert args.output is None


def test_parse_args_with_output_long() -> None:
    """Test parsing of --output option."""
    args = markdown_label_formatter.parse_args(["input.md", "--output", "out.md"])
    assert args.path == Path("input.md")
    assert args.output == Path("out.md")


def test_parse_args_with_output_short() -> None:
    """Test parsing of -o short option."""
    args = markdown_label_formatter.parse_args(["input.md", "-o", "out.md"])
    assert args.path == Path("input.md")
    assert args.output == Path("out.md")


def test_parse_args_no_arguments() -> None:
    """Test parsing with no arguments defaults to stdin/stdout."""
    args = markdown_label_formatter.parse_args([])
    assert args.path is None
    assert args.output is None


# Tests for main entry point


def test_main_successful_file_to_file() -> None:
    """Test successful execution with file input and output."""
    with (
        patch("scripts.dev_tools.markdown_label_formatter.read_content") as mock_read,
        patch("scripts.dev_tools.markdown_label_formatter.write_output") as mock_write,
    ):
        mock_read.return_value = "User: test"

        exit_code = markdown_label_formatter.main(["input.md", "--output", "output.md"])

        assert exit_code == 0
        mock_read.assert_called_once()
        mock_write.assert_called_once()


def test_main_successful_stdin_to_stdout() -> None:
    """Test successful execution with stdin/stdout."""
    with (
        patch("scripts.dev_tools.markdown_label_formatter.read_content") as mock_read,
        patch("scripts.dev_tools.markdown_label_formatter.write_output") as mock_write,
    ):
        mock_read.return_value = "User: test"

        exit_code = markdown_label_formatter.main([])

        assert exit_code == 0
        mock_read.assert_called_once_with(None)
        mock_write.assert_called_once()


def test_main_handles_read_error() -> None:
    """Test error handling when read fails."""
    with (
        patch("scripts.dev_tools.markdown_label_formatter.read_content") as mock_read,
        patch("sys.stderr", new_callable=io.StringIO),
    ):
        mock_read.side_effect = FileNotFoundError("File not found")

        exit_code = markdown_label_formatter.main(["missing.md"])

        assert exit_code == 1


def test_main_handles_write_error() -> None:
    """Test error handling when write fails."""
    with (
        patch("scripts.dev_tools.markdown_label_formatter.read_content") as mock_read,
        patch("scripts.dev_tools.markdown_label_formatter.write_output") as mock_write,
        patch("sys.stderr", new_callable=io.StringIO),
    ):
        mock_read.return_value = "User: test"
        mock_write.side_effect = PermissionError("Cannot write")

        exit_code = markdown_label_formatter.main(
            ["input.md", "--output", "readonly.md"]
        )

        assert exit_code == 1
