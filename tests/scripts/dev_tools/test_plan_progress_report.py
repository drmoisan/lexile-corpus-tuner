"""Unit tests for scripts/dev_tools/plan_progress_report.py.

These tests intentionally avoid filesystem temp directories per repo policy.
Instead, they focus on pure functions and use monkeypatching when filesystem
methods would otherwise be required.
"""

from pathlib import Path

import pytest  # noqa: TCH002 - pytest required at runtime for fixtures

from scripts.dev_tools.plan_progress_report import (
    PlanProgressRow,
    build_report_rows,
    count_checkboxes,
    discover_plan_files,
    parse_primary_issue,
    render_markdown_table,
    resolve_feature_and_type,
)


class TestCountCheckboxes:
    def test_counts_unchecked_and_total(self) -> None:
        markdown = """
### Phase 1
- [ ] Task A
- [x] Task B
- [X] Task C
""".strip()

        unchecked, total = count_checkboxes(markdown)
        assert (unchecked, total) == (1, 3)

    def test_ignores_non_list_checkbox_like_text(self) -> None:
        markdown = """
This is not a task: [ ]
And neither is this: [x]
""".strip()

        unchecked, total = count_checkboxes(markdown)
        assert (unchecked, total) == (0, 0)


class TestParsePrimaryIssue:
    def test_parses_plain_issue_line(self) -> None:
        markdown = """
# Feature - Plan

- Issue: #77
""".strip()

        assert parse_primary_issue(markdown) == "#77"

    def test_parses_bold_issue_line(self) -> None:
        markdown = """
# Feature - Plan

- **Issue:** #92
""".strip()

        assert parse_primary_issue(markdown) == "#92"

    def test_returns_empty_when_missing(self) -> None:
        assert parse_primary_issue("# Plan\n(no issue line)") == ""


class TestResolveFeatureAndType:
    def test_non_version_folder_uses_parent_as_feature(self) -> None:
        active_root = Path("/repo/docs/features/active")
        plan_path = Path("/repo/docs/features/active/feat-77/plan.md")

        feature, plan_type = resolve_feature_and_type(
            plan_path, active_root=active_root
        )
        assert feature == "feat-77"
        assert plan_type == "base"

    def test_version_folder_uses_parent_feature_and_type_is_version_dir(self) -> None:
        active_root = Path("/repo/docs/features/active")
        plan_path = Path(
            "/repo/docs/features/active/feat-73/v2/plan.2026-01-07T16-59.md"
        )

        feature, plan_type = resolve_feature_and_type(
            plan_path, active_root=active_root
        )
        assert feature == "feat-73"
        assert plan_type == "v2"


class TestBuildReportRows:
    def test_excludes_complete_plans(self) -> None:
        active_root = Path("/repo/docs/features/active")
        plan_path = Path("/repo/docs/features/active/feat-77/plan.md")
        markdown = "- Issue: #77\n- [x] Done\n"

        rows = build_report_rows([(plan_path, markdown)], active_root=active_root)
        assert rows == []

    def test_excludes_plans_without_any_checkboxes(self) -> None:
        active_root = Path("/repo/docs/features/active")
        plan_path = Path("/repo/docs/features/active/feat-77/plan.md")
        markdown = "- Issue: #77\n(no tasks)\n"

        rows = build_report_rows([(plan_path, markdown)], active_root=active_root)
        assert rows == []

    def test_includes_incomplete_plan_and_renders_issue(self) -> None:
        active_root = Path("/repo/docs/features/active")
        plan_path = Path(
            "/repo/docs/features/active/2026-01-07-atomic-executor-77/plan.md"
        )
        markdown = """
# Plan

- Issue: #77
- [ ] Todo
- [x] Done
""".strip()

        rows = build_report_rows([(plan_path, markdown)], active_root=active_root)
        assert len(rows) == 1
        assert rows[0].feature == "2026-01-07-atomic-executor-77"
        assert rows[0].issue == "#77"
        assert rows[0].plan_type == "base"
        assert rows[0].plan_path == plan_path
        assert (rows[0].unchecked, rows[0].total) == (1, 2)


class TestRenderMarkdownTable:
    def test_renders_header_only_when_empty(self) -> None:
        table = render_markdown_table([])
        assert "| feature | issue | type | remaining | plan |" in table
        assert table.count("\n") >= 1

    def test_renders_rows(self) -> None:
        rows = [
            PlanProgressRow(
                feature="feat",
                issue="#1",
                plan_type="base",
                plan_path=Path("/repo/docs/features/active/feat/plan.md"),
                unchecked=2,
                total=5,
            )
        ]
        table = render_markdown_table(rows)
        assert "| feat | #1 | base | 2/5 | [plan]" in table

    def test_renders_relative_plan_links_when_report_path_provided(self) -> None:
        rows = [
            PlanProgressRow(
                feature="feat",
                issue="#1",
                plan_type="base",
                plan_path=Path("/repo/docs/features/active/feat/plan.md"),
                unchecked=2,
                total=5,
            )
        ]
        report_path = Path("/repo/artifacts/active_plan_progress.md")

        table = render_markdown_table(rows, report_path=report_path)
        assert "[plan](../docs/features/active/feat/plan.md)" in table


class TestDiscoverPlanFiles:
    def test_includes_plan_dash_variants(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ensure plan discovery includes plan-*.md without touching disk."""

        active_root = Path("/repo/docs/features/active")
        plan = Path("/repo/docs/features/active/feat/plan.md")
        plan_dot = Path("/repo/docs/features/active/feat/plan.2026-01-01T00-00.md")
        plan_dash = Path("/repo/docs/features/active/feat/v1/plan-20260106-1731.md")

        # Route by glob pattern to simulate Path.rglob behavior.
        def fake_rglob(self: Path, pattern: str) -> list[Path]:
            if pattern == "plan.md":
                return [plan]
            if pattern == "plan.*.md":
                return [plan_dot]
            if pattern == "plan-*.md":
                return [plan_dash]
            return []

        monkeypatch.setattr(Path, "rglob", fake_rglob)

        discovered = discover_plan_files(active_root)
        assert discovered == sorted([plan, plan_dot, plan_dash])
