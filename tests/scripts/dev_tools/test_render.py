"""Unit tests for scripts/dev_tools/pr_context/render.py module."""

from unittest.mock import Mock

from scripts.dev_tools.pr_context.models import (
    CommandResult,
    IssueDetails,
    PullRequestDetails,
)
from scripts.dev_tools.pr_context.render import (
    build_close_candidates_section,
    completed_plan_tasks,
    convert_numstat,
    extension_summary,
    extract_changed_paths,
    extract_issue_references,
    extract_merge_pr_numbers,
    format_diff_path,
    format_issue_details,
    format_pr_details,
    parse_section,
    select_default_base,
    summarize_conventional_commits,
)


class TestSelectDefaultBase:
    def test_select_default_base_finds_origin_main(self) -> None:
        """select_default_base returns origin/main when it exists."""
        git = Mock()
        git.run.return_value = CommandResult(code=0, stdout="abc123\n", stderr="")
        result = select_default_base(git)
        assert result == "origin/main"
        git.run.assert_called_once_with(
            ["rev-parse", "--verify", "--quiet", "origin/main"], allow_error=True
        )

    def test_select_default_base_tries_master_fallback(self) -> None:
        """select_default_base falls back to origin/master."""
        git = Mock()
        git.run.side_effect = [
            CommandResult(code=1, stdout="", stderr=""),
            CommandResult(code=0, stdout="def456\n", stderr=""),
        ]
        result = select_default_base(git)
        assert result == "origin/master"

    def test_select_default_base_returns_none_when_all_fail(self) -> None:
        """select_default_base returns None when no candidates exist."""
        git = Mock()
        git.run.return_value = CommandResult(code=1, stdout="", stderr="")
        result = select_default_base(git)
        assert result is None


class TestFormatDiffPath:
    def test_format_diff_path_none_input(self) -> None:
        """format_diff_path returns empty string for None."""
        assert format_diff_path(None) == ""

    def test_format_diff_path_empty_string(self) -> None:
        """format_diff_path preserves empty string."""
        assert format_diff_path("") == ""
        assert format_diff_path("   ") == "   "

    def test_format_diff_path_removes_quotes(self) -> None:
        """format_diff_path strips surrounding quotes."""
        assert format_diff_path('"file.py"') == "file.py"

    def test_format_diff_path_handles_brace_rename(self) -> None:
        """format_diff_path extracts target from brace syntax."""
        assert format_diff_path("file{old => new}.py") == "filenew.py"

    def test_format_diff_path_handles_arrow_rename(self) -> None:
        """format_diff_path extracts target from arrow rename."""
        assert format_diff_path("old/path.py => new/path.py") == "new/path.py"

    def test_format_diff_path_plain_path(self) -> None:
        """format_diff_path returns plain path unchanged."""
        assert format_diff_path("src/module.py") == "src/module.py"


class TestConvertNumstat:
    def test_convert_numstat_single_file(self) -> None:
        """convert_numstat parses single file stats."""
        numstat = "10\t5\tsrc/file.py"
        adds, dels, files = convert_numstat(numstat)
        assert adds == 10
        assert dels == 5
        assert files == ["src/file.py"]

    def test_convert_numstat_multiple_files(self) -> None:
        """convert_numstat sums stats across multiple files."""
        numstat = "10\t5\tfile1.py\n20\t15\tfile2.py"
        adds, dels, files = convert_numstat(numstat)
        assert adds == 30
        assert dels == 20
        assert files == ["file1.py", "file2.py"]

    def test_convert_numstat_binary_files(self) -> None:
        """convert_numstat handles binary file markers."""
        numstat = "10\t5\tfile.py\n-\t-\tbinary.bin"
        adds, dels, files = convert_numstat(numstat)
        assert adds == 10
        assert dels == 5
        assert files == ["file.py", "binary.bin"]

    def test_convert_numstat_empty_input(self) -> None:
        """convert_numstat returns zeros for empty input."""
        adds, dels, files = convert_numstat("")
        assert adds == 0
        assert dels == 0
        assert files == []

    def test_convert_numstat_malformed_lines(self) -> None:
        """convert_numstat skips malformed lines."""
        numstat = "10\t5\tfile.py\nmalformed\n20\t10\tfile2.py"
        adds, dels, files = convert_numstat(numstat)
        assert adds == 30
        assert dels == 15
        assert files == ["file.py", "file2.py"]


class TestExtensionSummary:
    def test_extension_summary_groups_by_extension(self) -> None:
        """extension_summary groups files by extension."""
        files = ["file1.py", "file2.py", "file3.js"]
        result = extension_summary(files)
        assert "2  .py" in result
        assert "1  .js" in result

    def test_extension_summary_handles_no_extension(self) -> None:
        """extension_summary labels files without extension."""
        files = ["Makefile", "README"]
        result = extension_summary(files)
        assert "(noext)" in result

    def test_extension_summary_handles_unknown_paths(self) -> None:
        """extension_summary handles paths that raise ValueError."""
        files = ["normal.py", ""]
        result = extension_summary(files)
        assert ".py" in result

    def test_extension_summary_empty_list(self) -> None:
        """extension_summary handles empty file list."""
        result = extension_summary([])
        assert result == ""


class TestExtractIssueReferences:
    def test_extract_issue_references_finds_github(self) -> None:
        """extract_issue_references finds #123 patterns."""
        text = "Fix #42 and close #100"
        result = extract_issue_references(text)
        assert "#42" in result
        assert "#100" in result

    def test_extract_issue_references_finds_jira(self) -> None:
        """extract_issue_references finds PROJ-123 patterns."""
        text = "Related to ABC-456 and XYZ-789"
        result = extract_issue_references(text)
        assert "ABC-456" in result
        assert "XYZ-789" in result

    def test_extract_issue_references_deduplicates(self) -> None:
        """extract_issue_references removes duplicates."""
        text = "#10 again #10 and #10"
        result = extract_issue_references(text)
        assert result.count("#10") == 1

    def test_extract_issue_references_empty(self) -> None:
        """extract_issue_references returns empty list for no matches."""
        result = extract_issue_references("no refs here")
        assert result == []


class TestExtractMergePrNumbers:
    def test_extract_merge_pr_numbers_finds_merge_commits(self) -> None:
        """extract_merge_pr_numbers extracts PR numbers from merge commits."""
        log = ["Merge pull request #42 from branch", "Merge pull request #100"]
        result = extract_merge_pr_numbers(log)
        assert set(result) == {"#42", "#100"}
        assert len(result) == 2

    def test_extract_merge_pr_numbers_no_matches(self) -> None:
        """extract_merge_pr_numbers returns empty for non-merge commits."""
        log = ["Normal commit message", "Another commit"]
        result = extract_merge_pr_numbers(log)
        assert result == []

    def test_extract_merge_pr_numbers_empty_input(self) -> None:
        """extract_merge_pr_numbers handles empty list."""
        result = extract_merge_pr_numbers([])
        assert result == []


class TestSummarizeConventionalCommits:
    def test_summarize_conventional_commits_groups_by_type(self) -> None:
        """summarize_conventional_commits groups by conventional type."""
        commits = "feat: add feature\nfeat: another\nfix: bug"
        result = summarize_conventional_commits(commits)
        assert "feat" in result.lower()
        assert "fix" in result.lower()

    def test_summarize_conventional_commits_ignores_non_conventional(self) -> None:
        """summarize_conventional_commits skips non-conventional commits."""
        commits = "feat: feature\nrandom commit\nfix: bug"
        result = summarize_conventional_commits(commits)
        assert "feat" in result.lower()
        assert "fix" in result.lower()
        assert "random" not in result.lower()

    def test_summarize_conventional_commits_empty_list(self) -> None:
        """summarize_conventional_commits handles empty input."""
        result = summarize_conventional_commits("")
        assert "(no recognizable conventional commit types)" in result

    def test_summarize_conventional_commits_scope_handling(self) -> None:
        """summarize_conventional_commits handles scoped commits."""
        commits = "feat(api): add endpoint\nfix(ui): button"
        result = summarize_conventional_commits(commits)
        assert "feat" in result.lower()


class TestExtractChangedPaths:
    def test_extract_changed_paths_from_context_text(self) -> None:
        """extract_changed_paths parses paths from context text."""
        context = "===== Changed files =====\n10\t5\tfile1.py\n20\t10\tfile2.py\n====="
        result = extract_changed_paths(context)
        assert "file1.py" in result
        assert "file2.py" in result

    def test_extract_changed_paths_empty_section(self) -> None:
        """extract_changed_paths returns empty list for no changed files section."""
        context = "Some other text"
        result = extract_changed_paths(context)
        assert result == []

    def test_extract_changed_paths_no_files(self) -> None:
        """extract_changed_paths returns empty list when section is empty."""
        context = "===== Changed files =====\n====="
        result = extract_changed_paths(context)
        assert result == []


class TestParseSection:
    def test_parse_section_finds_matching_heading(self) -> None:
        """parse_section extracts content under matching H2."""
        markdown = "## Heading\nContent here\n## Other\nOther content"
        result = parse_section(markdown, "Heading")
        assert result == "Content here"

    def test_parse_section_no_match(self) -> None:
        """parse_section returns empty string when heading not found."""
        markdown = "## Heading\nContent"
        result = parse_section(markdown, "Missing")
        assert result == ""

    def test_parse_section_stops_at_next_heading(self) -> None:
        """parse_section stops at next ## heading."""
        markdown = "## Target\nLine 1\nLine 2\n## Next\nOther"
        result = parse_section(markdown, "Target")
        assert result == "Line 1\nLine 2"


class TestCompletedPlanTasks:
    def test_completed_plan_tasks_finds_checked_items(self) -> None:
        """completed_plan_tasks extracts [x] and [X] items."""
        markdown = "- [x] Task 1\n- [ ] Task 2\n- [X] Task 3"
        result = completed_plan_tasks(markdown)
        assert result == ["Task 1", "Task 3"]

    def test_completed_plan_tasks_respects_limit(self) -> None:
        """completed_plan_tasks respects limit parameter."""
        markdown = "\n".join(f"- [x] Task {i}" for i in range(20))
        result = completed_plan_tasks(markdown, limit=5)
        assert len(result) == 5

    def test_completed_plan_tasks_empty_markdown(self) -> None:
        """completed_plan_tasks returns empty list for no checked items."""
        markdown = "- [ ] Task 1\n- [ ] Task 2"
        result = completed_plan_tasks(markdown)
        assert result == []


class TestFormatIssueDetails:
    def test_format_issue_details_basic(self) -> None:
        """format_issue_details formats issue with required fields."""
        issue = IssueDetails(
            number="42",
            title="Test Issue",
            state="open",
            author="user1",
            body="Issue body",
            labels=["bug", "high-priority"],
            assignees=["dev1", "dev2"],
            created_at="2024-01-01",
            updated_at="2024-01-02",
            comments=[],
            user_story_path=None,
            user_story_content=None,
        )
        result = format_issue_details(issue)
        assert "Issue 42: Test Issue" in result
        assert "State: open" in result
        assert "Author: user1" in result
        assert "bug, high-priority" in result
        assert "dev1, dev2" in result

    def test_format_issue_details_with_user_story(self) -> None:
        """format_issue_details includes user story when present."""
        issue = IssueDetails(
            number="42",
            title="Test",
            state="open",
            author="user1",
            body="Body",
            labels=[],
            assignees=[],
            created_at="2024-01-01",
            updated_at="2024-01-02",
            comments=[],
            user_story_path="user-story.md",
            user_story_content="User story content",
        )
        result = format_issue_details(issue)
        assert "User story (user-story.md):" in result
        assert "User story content" in result

    def test_format_issue_details_no_labels_assignees(self) -> None:
        """format_issue_details shows (none) for empty lists."""
        issue = IssueDetails(
            number="42",
            title="Test",
            state="open",
            author="user1",
            body="Body",
            labels=[],
            assignees=[],
            created_at="2024-01-01",
            updated_at="2024-01-02",
            comments=[],
            user_story_path=None,
            user_story_content=None,
        )
        result = format_issue_details(issue)
        assert "Labels: (none)" in result
        assert "Assignees: (none)" in result


class TestFormatPRDetails:
    def test_format_pr_details_basic(self) -> None:
        """format_pr_details formats PR with required fields."""
        pr = PullRequestDetails(
            number="100",
            title="Test PR",
            state="merged",
            author="dev1",
            body="PR body",
            base_ref="main",
            head_ref="feature",
            labels=["enhancement"],
            assignees=["reviewer1"],
            created_at="2024-01-01",
            updated_at="2024-01-02",
            merged_at="2024-01-03",
            closing_issues=["#42", "#43"],
            files_changed=["file1.py", "file2.py"],
        )
        result = format_pr_details(pr)
        assert "Pull Request 100: Test PR" in result
        assert "State: merged" in result
        assert "Base: main" in result
        assert "Head: feature" in result
        assert "Merged: 2024-01-03" in result

    def test_format_pr_details_not_merged(self) -> None:
        """format_pr_details shows (not merged) when merged_at is None."""
        pr = PullRequestDetails(
            number="100",
            title="Test",
            state="open",
            author="dev1",
            body="Body",
            base_ref="main",
            head_ref="feature",
            labels=[],
            assignees=[],
            created_at="2024-01-01",
            updated_at="2024-01-02",
            merged_at=None,
            closing_issues=[],
            files_changed=[],
        )
        result = format_pr_details(pr)
        assert "Merged: (not merged)" in result


class TestBuildCloseCandidatesSection:
    def test_build_close_candidates_section_with_all_types(self) -> None:
        """build_close_candidates_section formats all candidate types."""
        result = build_close_candidates_section(
            verified=["#1", "#2"],
            author_asserted=["#3"],
            referenced=["#4", "#5"],
            verified_reason="Found in PR metadata",
            author_reason="Found in commit messages",
        )
        assert "Close candidates" in result
        assert "#1" in result
        assert "#2" in result
        assert "#3" in result

    def test_build_close_candidates_section_combines_author_and_ref(self) -> None:
        """build_close_candidates_section merges author_asserted and referenced."""
        result = build_close_candidates_section(
            verified=[],
            author_asserted=["#1"],
            referenced=["#2", "#1"],
            verified_reason="(none)",
            author_reason="(found)",
        )
        # author_auto_close should be sorted(set(author_asserted + referenced))
        # which is ["#1", "#2"]
        assert "#1" in result
        assert "#2" in result
