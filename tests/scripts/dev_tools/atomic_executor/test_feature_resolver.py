"""
Tests for atomic_executor.feature_resolver module.

Tests cover FeatureResolver class methods for resolving feature folder paths
from CLI args, explicit flags, or git branch inference with fuzzy matching.
"""

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from scripts.dev_tools.atomic_executor.feature_resolver import FeatureResolver

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


def _mock_git_path(_: str) -> str:
    """Return a fake git path for monkeypatching shutil.which."""

    return "/usr/bin/git"


def _mock_missing_git(_: str) -> None:
    """Return None to simulate git not being installed."""

    return None


class TestFeatureResolverInit:
    """Tests for FeatureResolver initialization."""

    def test_init_with_valid_paths(self, tmp_path: Path) -> None:
        """FeatureResolver initializes with valid workspace and active_dir."""
        active_dir = tmp_path / "active"
        active_dir.mkdir()

        resolver = FeatureResolver(tmp_path, active_dir)
        assert resolver.workspace == tmp_path
        assert resolver.active_dir == active_dir

    def test_init_raises_for_nonexistent_active_dir(self, tmp_path: Path) -> None:
        """FeatureResolver raises FileNotFoundError for nonexistent active_dir."""
        active_dir = tmp_path / "missing"

        with pytest.raises(FileNotFoundError, match="Active features directory"):
            FeatureResolver(tmp_path, active_dir)


class TestFeatureResolverListFolders:
    """Tests for FeatureResolver.list_folders() method."""

    def test_list_folders_returns_subdirectories(self, tmp_path: Path) -> None:
        """list_folders returns all subdirectories in active_dir."""
        active_dir = tmp_path / "active"
        active_dir.mkdir()
        (active_dir / "feature-one").mkdir()
        (active_dir / "feature-two").mkdir()
        (active_dir / "ignore.txt").write_text("not a folder")

        resolver = FeatureResolver(tmp_path, active_dir)
        folders = resolver.list_folders()
        assert len(folders) == 2
        assert "feature-one" in folders
        assert "feature-two" in folders

    def test_list_folders_returns_empty_for_no_subdirs(self, tmp_path: Path) -> None:
        """list_folders returns empty list when no subdirectories exist."""
        active_dir = tmp_path / "active"
        active_dir.mkdir()

        resolver = FeatureResolver(tmp_path, active_dir)
        folders = resolver.list_folders()
        assert folders == []


class TestFeatureResolverResolve:
    """Tests for FeatureResolver.resolve() method."""

    def test_resolve_with_direct_path_absolute(self, tmp_path: Path) -> None:
        """resolve() returns path when path_arg is absolute dir that exists."""
        active_dir = tmp_path / "active"
        active_dir.mkdir()
        feature_dir = tmp_path / "my-feature"
        feature_dir.mkdir()

        resolver = FeatureResolver(tmp_path, active_dir)
        name, path = resolver.resolve(str(feature_dir), None)
        assert name == "my-feature"
        assert path == feature_dir.resolve()

    def test_resolve_with_plan_md_path(self, tmp_path: Path) -> None:
        """resolve() handles path_arg pointing to plan.md file."""
        active_dir = tmp_path / "active"
        active_dir.mkdir()
        feature_dir = tmp_path / "my-feature"
        feature_dir.mkdir()
        plan_file = feature_dir / "plan.md"
        plan_file.write_text("# Plan", encoding="utf-8")

        resolver = FeatureResolver(tmp_path, active_dir)
        name, path = resolver.resolve(str(plan_file), None)
        assert name == "my-feature"
        assert path == feature_dir.resolve()

    def test_resolve_with_explicit_feature_arg(self, tmp_path: Path) -> None:
        """resolve() uses feature_arg when provided."""
        active_dir = tmp_path / "active"
        active_dir.mkdir()
        feature_dir = active_dir / "my-feature"
        feature_dir.mkdir()

        resolver = FeatureResolver(tmp_path, active_dir)
        # Use non-existent path_arg to trigger feature selection logic
        name, path = resolver.resolve("nonexistent", "my-feature")
        assert name == "my-feature"
        assert path == feature_dir.resolve()

    def test_resolve_raises_for_nonexistent_feature_arg(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """resolve() raises RuntimeError for nonexistent feature_arg."""
        active_dir = tmp_path / "active"
        active_dir.mkdir()
        # Create at least one folder so we don't hit "no folders" error first
        (active_dir / "other-feature").mkdir()

        # Mock git to avoid real branch influencing the test
        def mock_run(*args: object, **kwargs: object) -> Mock:
            result = Mock()
            result.stdout = "feature/some-branch\n"
            result.returncode = 0
            return result

        monkeypatch.setattr("subprocess.run", mock_run)
        monkeypatch.setattr("shutil.which", _mock_git_path)

        resolver = FeatureResolver(tmp_path, active_dir)
        with pytest.raises(RuntimeError, match="Feature folder .* not found"):
            resolver.resolve("nonexistent", "nonexistent-feature")

    def test_resolve_with_git_branch_inference(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """resolve() infers from git branch when no explicit args."""
        active_dir = tmp_path / "active"
        active_dir.mkdir()
        feature_dir = active_dir / "add-tests"
        feature_dir.mkdir()

        # Mock git command
        def mock_run(*args: object, **kwargs: object) -> Mock:
            result = Mock()
            result.stdout = "feature/add-tests-#123\n"
            result.returncode = 0
            return result

        monkeypatch.setattr("subprocess.run", mock_run)
        monkeypatch.setattr("shutil.which", _mock_git_path)

        resolver = FeatureResolver(tmp_path, active_dir)
        # Use non-existent path to trigger feature selection
        name, path = resolver.resolve("nonexistent", None)
        assert name == "add-tests"
        assert path == feature_dir.resolve()

    def test_resolve_raises_when_no_matches_found(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """resolve() raises RuntimeError when git branch has no matches."""
        active_dir = tmp_path / "active"
        active_dir.mkdir()
        (active_dir / "other-feature").mkdir()

        def mock_run(*args: object, **kwargs: object) -> Mock:
            result = Mock()
            result.stdout = "feature/nonexistent-#99\n"
            result.returncode = 0
            return result

        monkeypatch.setattr("subprocess.run", mock_run)
        monkeypatch.setattr("shutil.which", _mock_git_path)

        resolver = FeatureResolver(tmp_path, active_dir)
        with pytest.raises(RuntimeError, match="Could not resolve feature folder"):
            resolver.resolve("nonexistent", None)

    def test_resolve_raises_for_multiple_matches(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """resolve() raises RuntimeError when multiple folders match."""
        active_dir = tmp_path / "active"
        active_dir.mkdir()
        (active_dir / "test-one").mkdir()
        (active_dir / "test-two").mkdir()

        def mock_run(*args: object, **kwargs: object) -> Mock:
            result = Mock()
            result.stdout = "feature/test-#5\n"
            result.returncode = 0
            return result

        monkeypatch.setattr("subprocess.run", mock_run)
        monkeypatch.setattr("shutil.which", _mock_git_path)

        resolver = FeatureResolver(tmp_path, active_dir)
        with pytest.raises(RuntimeError, match="Multiple feature folders match"):
            resolver.resolve("nonexistent", None)


class TestFeatureResolverEdgeCases:
    """Edge case tests for FeatureResolver."""

    def test_resolve_with_issue_number_suffix(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """resolve() strips #<issue> suffix from branch names."""
        active_dir = tmp_path / "active"
        active_dir.mkdir()
        feature_dir = active_dir / "my-feature"
        feature_dir.mkdir()

        def mock_run(*args: object, **kwargs: object) -> Mock:
            result = Mock()
            result.stdout = "feature/my-feature-#42\n"
            result.returncode = 0
            return result

        monkeypatch.setattr("subprocess.run", mock_run)
        monkeypatch.setattr("shutil.which", _mock_git_path)

        resolver = FeatureResolver(tmp_path, active_dir)
        name, path = resolver.resolve("nonexistent", None)
        assert name == "my-feature"
        assert path == feature_dir.resolve()

    def test_resolve_handles_git_not_found(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """resolve() handles missing git executable gracefully."""
        active_dir = tmp_path / "active"
        active_dir.mkdir()
        (active_dir / "some-feature").mkdir()

        monkeypatch.setattr("shutil.which", _mock_missing_git)

        resolver = FeatureResolver(tmp_path, active_dir)
        # Should raise because no explicit feature and git not available
        with pytest.raises(RuntimeError, match="Could not resolve feature folder"):
            resolver.resolve("nonexistent", None)

    def test_resolve_handles_git_subprocess_error(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """resolve() handles git command failure gracefully."""
        active_dir = tmp_path / "active"
        active_dir.mkdir()
        # Need at least one folder so we get past that check
        (active_dir / "some-feature").mkdir()

        def mock_run(*args: object, **kwargs: object) -> Mock:
            raise subprocess.CalledProcessError(1, "git")

        monkeypatch.setattr("subprocess.run", mock_run)
        monkeypatch.setattr("shutil.which", _mock_git_path)

        resolver = FeatureResolver(tmp_path, active_dir)
        with pytest.raises(RuntimeError, match="Could not resolve feature folder"):
            resolver.resolve("nonexistent", None)

    def test_list_folders_raises_for_no_folders(
        self, tmp_path: Path, monkeypatch: "MonkeyPatch"
    ) -> None:
        """resolve() raises RuntimeError when active_dir has no folders."""
        active_dir = tmp_path / "active"
        active_dir.mkdir()

        def mock_run(*args: object, **kwargs: object) -> Mock:
            result = Mock()
            result.stdout = "feature/something\n"
            result.returncode = 0
            return result

        monkeypatch.setattr("subprocess.run", mock_run)
        monkeypatch.setattr("shutil.which", _mock_git_path)

        resolver = FeatureResolver(tmp_path, active_dir)
        with pytest.raises(RuntimeError, match="No feature folders found"):
            resolver.resolve("nonexistent", None)
