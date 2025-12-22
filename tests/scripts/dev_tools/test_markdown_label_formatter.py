"""Tests for markdown_label_formatter."""

from __future__ import annotations

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
