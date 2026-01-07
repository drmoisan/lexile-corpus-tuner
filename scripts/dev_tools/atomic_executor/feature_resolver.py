"""
Feature folder resolution from CLI args and git branch heuristics.

Supports automatic detection of feature folder based on current branch name,
manual override via --feature, or direct path specification.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


class FeatureResolver:
    """
    Resolve feature folders from args/branch heuristics.

    Purpose:
        Automatically determines which feature folder under docs/features/active
        should be used for plan execution, based on CLI arguments or git branch.

    Usage:
        resolver = FeatureResolver(workspace, active_dir)
        feature_name, feature_dir = resolver.resolve(path_arg, feature_arg)

    Flow:
        1. If path_arg is a directory or plan.md, use it directly
        2. If --feature provided, use that
        3. Otherwise, infer from git branch name using fuzzy matching
        4. Raise error if resolution is ambiguous or fails

    Invariants:
        - workspace and active_dir must exist
        - Resolved feature folder must contain plan.md

    Side Effects:
        - Calls git commands to determine current branch (if needed)
    """

    def __init__(self, workspace: Path, active_dir: Path) -> None:
        """
        Initialize the resolver with workspace paths.

        Args:
            workspace (Path): Repository root directory.
            active_dir (Path): docs/features/active directory.

        Raises:
            FileNotFoundError: If active_dir does not exist.
        """
        if not active_dir.is_dir():
            raise FileNotFoundError(
                f"Active features directory not found: {active_dir}"
            )
        self.workspace = workspace
        self.active_dir = active_dir

    def resolve(self, path_arg: str, feature_arg: str | None) -> tuple[str, Path]:
        """
        Resolve feature folder from path argument or feature name.

        Purpose:
            Determine the feature folder to use for plan execution.

        Args:
            path_arg (str): CLI path argument (feature folder, plan.md, or name).
            feature_arg (str | None): Explicit --feature name override.

        Returns:
            tuple[str, Path]: (feature_name, feature_dir_path)

        Raises:
            RuntimeError: If resolution fails or is ambiguous.
        """
        p = Path(path_arg)

        # Direct path resolution (feature folder or plan.md)
        if p.is_dir():
            return p.name, p.resolve()
        if p.is_file() and p.name == "plan.md":
            feature_dir = p.parent.resolve()
            return feature_dir.name, feature_dir

        # Resolve from active directory using feature name or branch
        branch = self._current_branch()
        feature_name = self._select_feature_folder(feature_arg, branch)
        return feature_name, (self.active_dir / feature_name).resolve()

    def list_folders(self) -> list[str]:
        """
        List all feature folders under active directory.

        Returns:
            list[str]: Sorted list of feature folder names.
        """
        return sorted(
            entry.name for entry in self.active_dir.iterdir() if entry.is_dir()
        )

    def _select_feature_folder(self, requested: str | None, branch: str | None) -> str:
        """
        Select feature folder using explicit request or branch heuristic.

        Purpose:
            Implements the feature folder selection logic with disambiguation.

        Args:
            requested (str | None): Explicit --feature name.
            branch (str | None): Current git branch name.

        Returns:
            str: Selected feature folder name.

        Raises:
            RuntimeError: If no folders found, multiple matches, or no match.
        """
        candidates = self.list_folders()
        if not candidates:
            raise RuntimeError(f"No feature folders found under {self.active_dir}")

        # Explicit feature name takes precedence
        if requested:
            if requested in candidates:
                return requested
            raise RuntimeError(
                f"Feature folder '{requested}' not found under {self.active_dir}"
            )

        # Branch-based fuzzy matching
        if branch:
            suffix = self._normalize_branch_suffix(branch)
            matches: list[str] = []

            for name in candidates:
                # Match normalized suffix or full branch in folder name
                if suffix and suffix in name:
                    matches.append(name)
                    continue
                if branch in name:
                    matches.append(name)

            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise RuntimeError(
                    "Multiple feature folders match the current branch; "
                    "rerun with --feature to disambiguate: " + ", ".join(matches)
                )

        # No matches found
        raise RuntimeError(
            "Could not resolve feature folder automatically; "
            "provide one with --feature. Available: " + ", ".join(candidates)
        )

    def _current_branch(self) -> str | None:
        """
        Get the current git branch name.

        Returns:
            str | None: Branch name, or None if not in a git repo or error.

        Side Effects:
            Calls git command.
        """
        git_exe = shutil.which("git")
        if not git_exe:
            return None

        try:
            result = subprocess.run(  # noqa: S603 - git_exe resolved via shutil.which
                [git_exe, "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                check=True,
            )
            branch = result.stdout.strip()
            return branch or None
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def _normalize_branch_suffix(self, branch: str) -> str:
        """
        Extract normalized suffix from branch name for matching.

        Purpose:
            Remove common prefixes and issue numbers for fuzzy matching.
            Example: "feature/my-feature-#42" -> "my-feature"

        Args:
            branch (str): Full branch name.

        Returns:
            str: Normalized suffix for matching.
        """
        # Take last component after /
        suffix = branch.split("/")[-1]
        # Remove # symbol
        suffix = suffix.replace("#", "")
        # Remove trailing digits (issue numbers)
        suffix = re.sub(r"-?\d+$", "", suffix)
        return suffix
