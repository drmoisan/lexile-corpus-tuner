from __future__ import annotations

from pathlib import Path  # noqa: TCH003 - required for pytest fixtures

from scripts.dev_tools.json_config import (
    EXCLUDE_GLOBS,
    GOVERNED_GLOBS,
    iter_governed_files,
)


def test_governed_globs_constant() -> None:
    """Verify GOVERNED_GLOBS constant is defined."""
    assert isinstance(GOVERNED_GLOBS, tuple)
    assert len(GOVERNED_GLOBS) > 0
    assert ".vscode/*.json" in GOVERNED_GLOBS


def test_exclude_globs_constant() -> None:
    """Verify EXCLUDE_GLOBS constant is defined."""
    assert isinstance(EXCLUDE_GLOBS, tuple)
    assert len(EXCLUDE_GLOBS) > 0
    assert "data/**" in EXCLUDE_GLOBS


def test_iter_governed_files_empty(tmp_path: Path) -> None:
    """No JSON files should yield nothing."""
    result = list(iter_governed_files(tmp_path))
    assert result == []


def test_iter_governed_files_finds_vscode_json(tmp_path: Path) -> None:
    """Files in .vscode/*.json should be found."""
    vscode_dir = tmp_path / ".vscode"
    vscode_dir.mkdir()
    tasks_json = vscode_dir / "tasks.json"
    tasks_json.write_text("{}")

    result = list(iter_governed_files(tmp_path))
    assert tasks_json in result


def test_iter_governed_files_finds_nested_vscode_json(tmp_path: Path) -> None:
    """Files matching .vscode/**/*.json should be found."""
    nested_dir = tmp_path / ".vscode" / "subdir"
    nested_dir.mkdir(parents=True)
    nested_json = nested_dir / "config.json"
    nested_json.write_text("{}")

    result = list(iter_governed_files(tmp_path))
    assert nested_json in result


def test_iter_governed_files_excludes_data_dir(tmp_path: Path) -> None:
    """Files under data/** should be excluded."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    data_json = data_dir / "metadata.json"
    data_json.write_text("{}")

    result = list(iter_governed_files(tmp_path))
    assert data_json not in result


def test_iter_governed_files_excludes_artifacts_dir(tmp_path: Path) -> None:
    """Files under artifacts/** should be excluded."""
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    artifact_json = artifacts_dir / "output.json"
    artifact_json.write_text("{}")

    result = list(iter_governed_files(tmp_path))
    assert artifact_json not in result


def test_iter_governed_files_excludes_parent_in_excluded(tmp_path: Path) -> None:
    """Files with any parent in exclusion set should be excluded."""
    htmlcov_dir = tmp_path / "htmlcov" / "subdir"
    htmlcov_dir.mkdir(parents=True)
    report_json = htmlcov_dir / "report.json"
    report_json.write_text("{}")

    result = list(iter_governed_files(tmp_path))
    assert report_json not in result


def test_iter_governed_files_finds_devcontainer_json(tmp_path: Path) -> None:
    """Files in .devcontainer/*.json should be found."""
    devcontainer_dir = tmp_path / ".devcontainer"
    devcontainer_dir.mkdir()
    devcontainer_json = devcontainer_dir / "devcontainer.json"
    devcontainer_json.write_text("{}")

    result = list(iter_governed_files(tmp_path))
    assert devcontainer_json in result


def test_iter_governed_files_finds_scripts_json(tmp_path: Path) -> None:
    """Files matching scripts/**/*.json should be found."""
    scripts_dir = tmp_path / "scripts" / "subdir"
    scripts_dir.mkdir(parents=True)
    script_json = scripts_dir / "config.json"
    script_json.write_text("{}")

    result = list(iter_governed_files(tmp_path))
    assert script_json in result


def test_iter_governed_files_finds_docs_json(tmp_path: Path) -> None:
    """Files matching docs/**/*.json should be found."""
    docs_dir = tmp_path / "docs" / "features"
    docs_dir.mkdir(parents=True)
    doc_json = docs_dir / "manifest.json"
    doc_json.write_text("{}")

    result = list(iter_governed_files(tmp_path))
    assert doc_json in result


def test_iter_governed_files_finds_examples_json(tmp_path: Path) -> None:
    """Files matching examples/**/*.json should be found."""
    examples_dir = tmp_path / "examples" / "meta"
    examples_dir.mkdir(parents=True)
    example_json = examples_dir / "sample.json"
    example_json.write_text("{}")

    result = list(iter_governed_files(tmp_path))
    assert example_json in result


def test_iter_governed_files_accepts_str_path(tmp_path: Path) -> None:
    """iter_governed_files should accept str paths."""
    vscode_dir = tmp_path / ".vscode"
    vscode_dir.mkdir()
    tasks_json = vscode_dir / "tasks.json"
    tasks_json.write_text("{}")

    result = list(iter_governed_files(str(tmp_path)))
    assert tasks_json in result


def test_iter_governed_files_mixed_included_excluded(tmp_path: Path) -> None:
    """Mix of included and excluded files should only yield included ones."""
    vscode_dir = tmp_path / ".vscode"
    vscode_dir.mkdir()
    included_json = vscode_dir / "settings.json"
    included_json.write_text("{}")

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    excluded_json = data_dir / "corpus.json"
    excluded_json.write_text("{}")

    result = list(iter_governed_files(tmp_path))
    assert included_json in result
    assert excluded_json not in result


def test_iter_governed_files_handles_non_file_matches(tmp_path: Path) -> None:
    """glob matches that are directories should be skipped."""
    vscode_dir = tmp_path / ".vscode"
    vscode_dir.mkdir()
    # Create a directory with .json suffix (unusual but possible)
    json_dir = vscode_dir / "weird.json"
    json_dir.mkdir()

    result = list(iter_governed_files(tmp_path))
    assert json_dir not in result
