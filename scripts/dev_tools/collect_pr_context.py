"""Collect Git repository context for pull request authorship.

This script supersedes the PowerShell-based collector by providing richer context
in a Python implementation. It separates PR numbers from issues, verifies
auto-close targets via the GitHub CLI when available, and embeds excerpts from
active feature documentation so PR authors can explain the "why" of the change.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

SECTION_LINE = "===== {title} ====="
CONVENTIONAL_TYPES = (
    "feat",
    "fix",
    "refactor",
    "perf",
    "docs",
    "test",
    "chore",
    "build",
    "ci",
    "style",
)


@dataclass
class CommandResult:
    """Represents the outcome of a shell command."""

    stdout: str
    stderr: str
    code: int


class CommandRunner(Protocol):
    """Runs shell commands and returns structured results."""

    def run(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        allow_error: bool = False,
    ) -> CommandResult: ...


class SubprocessRunner(CommandRunner):
    """Command runner that shells out using subprocess.run."""

    def run(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        allow_error: bool = False,
    ) -> CommandResult:
        completed = subprocess.run(  # noqa: S603
            args,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
        )

        stdout = (completed.stdout or "").rstrip("\n")
        stderr = (completed.stderr or "").rstrip("\n")
        result = CommandResult(
            stdout=stdout, stderr=stderr, code=int(completed.returncode)
        )

        if not allow_error and result.code != 0:
            joined = (stdout + "\n" + stderr).strip()
            raise RuntimeError(f"{' '.join(args)} failed ({result.code}): {joined}")

        return result


class GitClient:
    """Thin wrapper around git for typed access."""

    def __init__(self, runner: CommandRunner, cwd: Path) -> None:
        self._runner = runner
        self._cwd = cwd

    def run(self, args: Sequence[str], *, allow_error: bool = False) -> CommandResult:
        return self._runner.run(["git", *args], cwd=self._cwd, allow_error=allow_error)

    def resolve_root(self) -> Path:
        candidate = self._cwd / ".git"
        if candidate.exists():
            return self._cwd

        top = self.run(["rev-parse", "--show-toplevel"]).stdout
        return Path(top).resolve()

    def rev_parse(self, ref: str) -> str:
        return self.run(["rev-parse", "--verify", ref]).stdout

    def remote_verbose(self) -> str:
        return self.run(["remote", "-v"]).stdout

    def branch_name(self) -> str:
        return self.run(["rev-parse", "--abbrev-ref", "HEAD"]).stdout

    def upstream(self) -> str:
        res = self.run(
            ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            allow_error=True,
        )
        return res.stdout

    def status_short(self) -> str:
        return self.run(["status", "-sb"]).stdout

    def untracked(self) -> str:
        return self.run(["ls-files", "--others", "--exclude-standard"]).stdout

    def diff_name_status(self, *, staged: bool) -> str:
        args = ["diff", "--name-status"]
        if staged:
            args.insert(1, "--cached")
        return self.run(args, allow_error=True).stdout

    def diff_patch(self, *, staged: bool) -> str:
        args = ["diff"]
        if staged:
            args.append("--cached")
        return self.run(args, allow_error=True).stdout

    def merge_base(self, base: str, head: str) -> str:
        return self.run(["merge-base", base, head]).stdout

    def log(self, fmt: str, rev_range: str) -> str:
        return self.run(
            ["log", "--date=short", fmt, rev_range], allow_error=True
        ).stdout

    def diff_range(self, args: Sequence[str]) -> str:
        return self.run(["diff", *args], allow_error=True).stdout


class GhClient:
    """GitHub CLI wrapper used for entity classification and auto-close detection."""

    def __init__(self, runner: CommandRunner, cwd: Path) -> None:
        self._runner = runner
        self._cwd = cwd
        self._gh_path = shutil.which("gh")
        self._repo_cache: str | None = None
        self._available = self._gh_path is not None and self._is_authenticated()

    @property
    def available(self) -> bool:
        return self._available

    def _is_authenticated(self) -> bool:
        if not self._gh_path:
            return False
        result = self._runner.run(
            [self._gh_path, "auth", "status"], cwd=self._cwd, allow_error=True
        )
        return result.code == 0

    def _repo_name(self) -> str | None:
        if not self._available:
            return None
        if self._repo_cache:
            return self._repo_cache

        result = self._runner.run(
            [self._gh_path or "gh", "repo", "view", "--json", "nameWithOwner"],
            cwd=self._cwd,
            allow_error=True,
        )
        if result.code != 0 or not result.stdout:
            return None

        try:
            payload = json.loads(result.stdout)
            name = payload.get("nameWithOwner")
        except json.JSONDecodeError:
            name = None

        if isinstance(name, str):
            self._repo_cache = name
            return name
        return None

    def classify_entity(self, number: str) -> str | None:
        """Return "issue" or "pull" if determinable, otherwise None."""

        repo = self._repo_name()
        if not repo:
            return None

        api_path = f"repos/{repo}/issues/{number}"
        result = self._runner.run(
            [self._gh_path or "gh", "api", api_path],
            cwd=self._cwd,
            allow_error=True,
        )
        if result.code != 0 or not result.stdout:
            return None

        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None

        if "pull_request" in payload:
            return "pull"
        return "issue"

    def closing_issues(self, pr_number: str | None = None) -> list[str]:
        repo = self._repo_name()
        if not repo:
            return []

        args = [
            self._gh_path or "gh",
            "pr",
            "view",
            "--json",
            "closingIssuesReferences,number",
        ]
        if pr_number:
            args.insert(3, pr_number)

        result = self._runner.run(args, cwd=self._cwd, allow_error=True)
        if result.code != 0 or not result.stdout:
            return []

        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return []

        if not isinstance(payload, dict):
            return []

        payload_dict = cast(dict[str, object], payload)
        issues_raw: object | None = payload_dict.get("closingIssuesReferences")
        numbers: list[str] = []
        if isinstance(issues_raw, list):
            issues_list = cast(list[object], issues_raw)
            for entry in issues_list:
                if not isinstance(entry, dict):
                    continue

                entry_dict = cast(dict[str, object], entry)
                number = entry_dict.get("number")
                if isinstance(number, int):
                    numbers.append(f"#{number}")

        return sorted(set(numbers))


def section(title: str) -> str:
    return "\n" + SECTION_LINE.format(title=title) + "\n"


def format_diff_path(path_text: str | None) -> str:
    if path_text is None:
        return ""
    if path_text.strip() == "":
        return path_text

    trimmed = path_text.strip().strip('"')
    trimmed = re.sub(r"\{[^{}]*\s=>\s([^{}]*)\}", r"\1", trimmed)

    arrow_match = re.match(r"^\s*(.+?)\s=>\s(.+?)\s*$", trimmed)
    if arrow_match:
        return arrow_match.group(2)
    return trimmed


def convert_numstat(numstat_text: str) -> tuple[int, int, list[str]]:
    adds = 0
    dels = 0
    files: list[str] = []

    for raw_line in numstat_text.splitlines():
        if not raw_line.strip():
            continue

        parts = raw_line.split("\t")
        if len(parts) < 3:
            continue

        add_part, del_part, file_part = parts[0], parts[1], parts[2]
        if add_part.isdigit():
            adds += int(add_part)
        if del_part.isdigit():
            dels += int(del_part)
        files.append(file_part)

    return adds, dels, files


def extension_summary(files: Iterable[str]) -> str:
    counts: dict[str, int] = {}
    for raw in files:
        name = format_diff_path(raw)
        ext = "(unknown)"
        try:
            suffix = Path(name).suffix
            ext = suffix if suffix else "(noext)"
        except ValueError:
            fallback = re.search(r"\.([A-Za-z0-9_]+)$", name)
            ext = f".{fallback.group(1)}" if fallback else "(unknown)"

        counts[ext] = counts.get(ext, 0) + 1

    lines = [f"{counts[k]:8d}  {k}" for k in sorted(counts)]
    return "\n".join(lines)


def extract_issue_references(text: str) -> list[str]:
    if not text:
        return []
    matches = re.findall(r"(?<!\w)#\d+|\b[A-Z][A-Z0-9]+-\d+\b", text)
    seen: set[str] = set()
    ordered: list[str] = []
    for item in matches:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def extract_merge_pr_numbers(subjects: Iterable[str]) -> list[str]:
    numbers: set[str] = set()
    pattern = re.compile(r"Merge pull request #(\d+)", re.IGNORECASE)
    for subj in subjects:
        match = pattern.search(subj)
        if match:
            numbers.add(f"#{match.group(1)}")
    return sorted(numbers)


def summarize_conventional_commits(subjects: str) -> str:
    counts = {key: 0 for key in CONVENTIONAL_TYPES}
    counts["other"] = 0

    for line in subjects.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(
            r"(feat|fix|refactor|perf|docs|test|chore|build|ci|style)(\(|!|:)", line
        )
        label = match.group(1) if match else "other"
        counts[label] += 1

    non_zero = [(k, v) for k, v in counts.items() if v > 0]
    if not non_zero:
        return "(no recognizable conventional commit types)"
    return "\n".join(f"{name:<9} : {value}" for name, value in non_zero)


def parse_section(markdown: str, heading: str) -> str:
    escaped = re.escape(heading)
    pattern = rf"^##\s+{escaped}\s*\r?\n(.*?)(?=^##\s+|\Z)"
    match = re.search(pattern, markdown, flags=re.MULTILINE | re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


def truncate(text: str, limit: int = 800) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def completed_plan_tasks(markdown: str, *, limit: int = 10) -> list[str]:
    tasks: list[str] = []
    for line in markdown.splitlines():
        if re.search(r"\[x\]", line, flags=re.IGNORECASE):
            cleaned = re.sub(r"^[-*]\s*\[[xX]\]\s*", "", line).strip()
            tasks.append(cleaned)
        if len(tasks) >= limit:
            break
    return tasks


def gather_feature_excerpts(root: Path, changed_files: Iterable[str]) -> list[str]:
    features: set[str] = set()
    for raw in changed_files:
        parts = Path(raw).parts
        if (
            len(parts) >= 4
            and parts[0] == "docs"
            and parts[1] == "features"
            and parts[2] == "active"
        ):
            features.add(parts[3])

    excerpts: list[str] = []
    for feature in sorted(features):
        spec_path = root / "docs" / "features" / "active" / feature / "spec.md"
        plan_path = root / "docs" / "features" / "active" / feature / "plan.md"

        spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
        plan_text = plan_path.read_text(encoding="utf-8") if plan_path.exists() else ""

        spec_parts: list[str] = []
        for heading in (
            "Context",
            "Root Cause Analysis",
            "Proposed Fix",
            "Acceptance Criteria",
        ):
            section_text = parse_section(spec_text, heading)
            if section_text:
                spec_parts.append(f"{heading}: {truncate(section_text)}")

        plan_tasks = completed_plan_tasks(plan_text)
        plan_section = (
            "\n".join(f"- {task}" for task in plan_tasks) if plan_tasks else ""
        )

        lines: list[str] = [section(f"Feature doc: {feature}")]
        if spec_parts:
            lines.append("Spec excerpts:\n" + "\n\n".join(spec_parts))
        if plan_section:
            lines.append("Plan completed tasks:\n" + plan_section)
        if len(lines) == 1:
            lines.append("(no spec/plan excerpts found)")

        excerpts.append("\n".join(lines))

    return excerpts


def select_default_base(git: GitClient) -> str | None:
    candidates = [
        "origin/main",
        "origin/master",
        "main",
        "master",
        "origin/develop",
        "develop",
    ]
    for ref in candidates:
        result = git.run(["rev-parse", "--verify", "--quiet", ref], allow_error=True)
        if result.code == 0 and result.stdout.strip():
            return ref
    return None


def build_pr_context(
    *,
    git: GitClient,
    gh: GhClient | None,
    base_ref: str | None,
    head_ref: str | None,
    include_untracked: bool,
) -> str:
    branch_name = git.branch_name()
    upstream = git.upstream() or "(none)"

    remotes = git.remote_verbose()
    status_short = git.status_short()
    untracked = git.untracked() if include_untracked else ""
    untracked_display = untracked if untracked.strip() else "(none)"

    pr_block = ""
    try:
        resolved_base = base_ref or select_default_base(git)
        if not resolved_base:
            raise RuntimeError("Failed to resolve base ref (tried common defaults)")

        base = git.rev_parse(resolved_base)
        head = git.rev_parse(head_ref or "HEAD")
        merge_base = git.merge_base(base, head)
        rev_range = f"{merge_base}..{head}"

        oneline = git.log("--pretty=format:%h %ad %an %s", rev_range)
        subjects = git.log("--pretty=%s", rev_range)
        authors = git.log("--format=%an <%ae>", rev_range)
        authors_list = sorted(
            {line.strip() for line in authors.splitlines() if line.strip()}
        )

        name_status = git.diff_range(["--name-status", merge_base, head])
        numstat = git.diff_range(["--numstat", merge_base, head])
        shortstat = git.diff_range(["--shortstat", merge_base, head])
        stat = git.diff_range(["--stat", merge_base, head])

        additions, deletions, files = convert_numstat(numstat)
        ext_summary = extension_summary(files)
        merge_prs = extract_merge_pr_numbers(oneline.splitlines())

        issue_candidates = [
            ref
            for ref in extract_issue_references(oneline + "\n" + subjects)
            if ref not in merge_prs
        ]
        issues: list[str] = []
        prs: list[str] = []
        if gh and gh.available:
            for ref in issue_candidates:
                number = ref.lstrip("#")
                entity = gh.classify_entity(number)
                if entity == "issue":
                    issues.append(ref if ref.startswith("#") else f"#{ref}")
                elif entity == "pull":
                    prs.append(ref if ref.startswith("#") else f"#{ref}")
                else:
                    issues.append(ref)
        else:
            issues = issue_candidates

        issues_display = ", ".join(sorted(set(issues))) if issues else "(none)"
        prs_combined = sorted(set(prs + merge_prs))
        prs_display = ", ".join(prs_combined) if prs_combined else "(none)"

        closing = gh.closing_issues(None) if gh and gh.available else []
        closing_display = (
            ", ".join(closing)
            if closing
            else "(gh unavailable or no closing issues detected)"
        )

        oneline_display = oneline if oneline.strip() else "(none)"
        authors_display = "\n".join(authors_list) if authors_list else "(none)"
        name_status_display = name_status if name_status.strip() else "(none)"
        short_display = shortstat if shortstat.strip() else "(none)"
        ext_display = ext_summary if ext_summary else "(none)"
        stat_display = stat if stat.strip() else "(none)"

        pr_block = "\n".join(
            [
                section("PR Comparison"),
                f"Base: {base_ref or resolved_base}",
                f"Head: {head_ref or branch_name}",
                f"Merge-base: {merge_base}",
                f"Range: {rev_range}\n",
                section("Commits in range"),
                oneline_display,
                "",
                section("Conventional commit type summary"),
                summarize_conventional_commits(subjects),
                "",
                section("Authors"),
                authors_display,
                "",
                section("Changed files (name-status)"),
                name_status_display,
                "",
                section("Diff shortstat"),
                short_display,
                "",
                section("Additions/Deletions totals (from numstat)"),
                f"Additions: {additions}\nDeletions: {deletions}\n",
                section("Files by extension"),
                ext_display,
                "",
                section("Referenced issues (classified)"),
                issues_display,
                "",
                section("PRs in range"),
                prs_display,
                "",
                section(
                    "Issues to autoclose (verified)"
                    if closing
                    else "Issues to autoclose (verified or pending)"
                ),
                closing_display,
                "",
                section("Diff stat"),
                stat_display,
            ]
        )
    except Exception as exc:  # noqa: BLE001
        pr_block = section("PR Comparison") + f"(FAILED to compute PR context: {exc})\n"

    intent = "\n".join(
        [
            section("PR Intent (edit before generating PR body)"),
            "Primary outcome:",
            "Impact (user/developer):",
            "Risks:",
            "Author-asserted autoclose issues:",
        ]
    )

    return "\n".join(
        [
            intent,
            section("Repository remotes"),
            remotes,
            "",
            section("Current branch"),
            branch_name,
            "",
            section("Upstream"),
            upstream,
            "",
            section("Status (short)"),
            status_short,
            "",
            section("Untracked files"),
            untracked_display,
            "",
            section("Working tree diff (staged)"),
            git.diff_name_status(staged=True),
            git.diff_patch(staged=True),
            "",
            section("Working tree diff (unstaged)"),
            git.diff_name_status(staged=False),
            git.diff_patch(staged=False),
            pr_block,
        ]
    )


def write_output(text: str, out_path: Path, append: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with out_path.open(mode, encoding="utf-8") as handle:
        handle.write(text)


def collect_and_write(
    *,
    base: str | None,
    head: str | None,
    out: Path,
    repo_root: Path,
    append: bool,
    include_untracked: bool,
) -> None:
    runner = SubprocessRunner()
    git = GitClient(runner, repo_root)
    resolved_root = git.resolve_root()
    git = GitClient(runner, resolved_root)
    gh = GhClient(runner, resolved_root)

    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S %Z")
    header = section("Context generated") + "\n" + timestamp + "\n\n"

    context = build_pr_context(
        git=git,
        gh=gh if gh.available else None,
        base_ref=base,
        head_ref=head,
        include_untracked=include_untracked,
    )

    files = gather_feature_excerpts(resolved_root, extract_changed_paths(context))
    feature_block = "\n".join(files)
    final_text = header + context
    if feature_block:
        final_text = final_text + "\n" + feature_block

    write_output(final_text, out, append)
    print(f"Wrote context to: {out}")


def extract_changed_paths(context_text: str) -> list[str]:
    paths: list[str] = []
    capture = False
    for line in context_text.splitlines():
        if line.startswith("===== Changed files"):
            capture = True
            continue
        if capture:
            if line.startswith("====="):
                break
            if line.strip() and "\t" in line:
                path_part = line.split("\t")[-1]
                paths.append(format_diff_path(path_part.strip()))
            elif line.strip():
                paths.append(format_diff_path(line.strip()))
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect PR context for GitHub.")
    parser.add_argument(
        "--base", dest="base", help="Base ref (default: auto-detect origin/main)"
    )
    parser.add_argument("--head", dest="head", help="Head ref (default: HEAD)")
    parser.add_argument(
        "--out",
        dest="out",
        default="artifacts/pr_context.txt",
        help="Output file path",
    )
    parser.add_argument(
        "--repo-root",
        dest="repo_root",
        default=".",
        help="Repository root (defaults to current directory)",
    )
    parser.add_argument(
        "--append",
        dest="append",
        action="store_true",
        help="Append instead of overwrite",
    )
    parser.add_argument(
        "--no-untracked",
        dest="no_untracked",
        action="store_true",
        help="Exclude untracked files from status",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    out_path = Path(args.out).expanduser()
    repo_root = Path(args.repo_root).expanduser().resolve()
    collect_and_write(
        base=args.base,
        head=args.head,
        out=out_path,
        repo_root=repo_root,
        append=bool(args.append),
        include_untracked=not bool(args.no_untracked),
    )


if __name__ == "__main__":
    main()
