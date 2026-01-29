from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

import pytest  # noqa: TCH002  # Needed at runtime for pytest fixtures

import scripts.dev_tools.shell_qc as shell_qc


def _fixture_root() -> Path:
    return Path(__file__).resolve().parents[2] / "fixtures" / "shell_qc"


def test_discover_shell_scripts_includes_expected() -> None:
    root = _fixture_root()

    discovered = shell_qc.discover_shell_scripts(root)

    expected = {
        root / "tools" / "format_me.sh",
        root / "scripts" / "with_shebang",
    }
    assert set(discovered) == expected
    assert root / "scripts" / "ignored.txt" not in discovered
    assert root / "scripts" / "pwsh_script.ps1" not in discovered
    assert root / "scripts" / "node_modules" / "skip.sh" not in discovered


def test_find_bats_test_dirs_detects_shell() -> None:
    root = _fixture_root()

    assert shell_qc.find_bats_test_dirs(root) == [root / "tests" / "shell"]


def test_run_check_skips_when_no_scripts(
    monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _discover(_: Path | None = None) -> list[Path]:
        return []

    monkeypatch.setattr(shell_qc, "discover_shell_scripts", _discover)

    exit_code = shell_qc.run_check()

    assert exit_code == 0
    assert "No shell scripts found; skipping." in capsys.readouterr().out


def test_run_check_missing_tool(
    monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _discover(_: Path | None = None) -> list[Path]:
        return [Path("tools/test.sh")]

    monkeypatch.setattr(shell_qc, "discover_shell_scripts", _discover)

    def _which(tool: str) -> str | None:
        return None if tool == "shfmt" else "/usr/bin/shellcheck"

    monkeypatch.setattr(shell_qc.shutil, "which", _which)

    exit_code = shell_qc.run_check()

    output = capsys.readouterr().out
    assert exit_code == 127
    assert "Missing required tool: shfmt" in output
    assert "apt-get install -y shfmt" in output
    assert "brew install shfmt" in output
    assert "WSL" in output


def test_run_check_runs_shellcheck_per_file(monkeypatch: MonkeyPatch) -> None:
    files = [Path("tools/a.sh"), Path("scripts/b.sh")]

    def _discover(_: Path | None = None) -> list[Path]:
        return files

    def _which(tool: str) -> str:
        return f"/usr/bin/{tool}"

    monkeypatch.setattr(shell_qc, "discover_shell_scripts", _discover)
    monkeypatch.setattr(shell_qc.shutil, "which", _which)

    calls: list[list[str]] = []

    def _run(command: list[str], *args: Any, **kwargs: Any) -> SimpleNamespace:
        calls.append(list(command))
        # The implementation resolves tools via shutil.which(), so the executable
        # is typically a full path (e.g., /usr/bin/shellcheck) rather than the bare
        # command name.
        if Path(command[0]).name == "shellcheck" and "tools" in command[1]:
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(shell_qc.subprocess, "run", _run)

    exit_code = shell_qc.run_check()

    assert exit_code == 1
    assert calls[0] == [
        "/usr/bin/shfmt",
        "-d",
        str(Path("tools/a.sh")),
        str(Path("scripts/b.sh")),
    ]
    shellcheck_calls = [call for call in calls if call[0] == "/usr/bin/shellcheck"]
    assert shellcheck_calls == [
        ["/usr/bin/shellcheck", str(Path("tools/a.sh"))],
        ["/usr/bin/shellcheck", str(Path("scripts/b.sh"))],
    ]


def test_run_format_missing_tool(
    monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _discover(_: Path | None = None) -> list[Path]:
        return [Path("tools/a.sh")]

    def _which(_: str) -> str | None:
        return None

    monkeypatch.setattr(shell_qc, "discover_shell_scripts", _discover)
    monkeypatch.setattr(shell_qc.shutil, "which", _which)

    exit_code = shell_qc.run_format()

    assert exit_code == 127
    assert "Missing required tool: shfmt" in capsys.readouterr().out


def test_run_format_invokes_shfmt(monkeypatch: MonkeyPatch) -> None:
    files = [Path("tools/a.sh")]

    def _discover(_: Path | None = None) -> list[Path]:
        return files

    def _which(_: str) -> str:
        return "/usr/bin/shfmt"

    monkeypatch.setattr(shell_qc, "discover_shell_scripts", _discover)
    monkeypatch.setattr(shell_qc.shutil, "which", _which)

    calls: list[list[str]] = []

    def _run(command: list[str], *args: Any, **kwargs: Any) -> SimpleNamespace:
        calls.append(list(command))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(shell_qc.subprocess, "run", _run)

    exit_code = shell_qc.run_format()

    assert exit_code == 0
    assert calls == [["/usr/bin/shfmt", "-w", str(Path("tools/a.sh"))]]


def test_run_test_skips_when_no_dirs(
    monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _find(_: Path | None = None) -> list[Path]:
        return []

    def _which(_: str) -> str:
        return "/usr/bin/bats"

    monkeypatch.setattr(shell_qc, "find_bats_test_dirs", _find)
    monkeypatch.setattr(shell_qc.shutil, "which", _which)

    exit_code = shell_qc.run_test()

    assert exit_code == 0
    assert "No shell test directories found; skipping." in capsys.readouterr().out


def test_run_test_skips_when_bats_missing(
    monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _find(_: Path | None = None) -> list[Path]:
        return [Path("tests/shell")]

    def _which(_: str) -> str | None:
        return None

    monkeypatch.setattr(shell_qc, "find_bats_test_dirs", _find)
    monkeypatch.setattr(shell_qc.shutil, "which", _which)

    exit_code = shell_qc.run_test()

    assert exit_code == 0
    assert "bats not installed" in capsys.readouterr().out


def test_run_test_runs_bats(monkeypatch: MonkeyPatch) -> None:
    def _find(_: Path | None = None) -> list[Path]:
        return [Path("tests/shell")]

    def _which(_: str) -> str:
        return "/usr/bin/bats"

    monkeypatch.setattr(shell_qc, "find_bats_test_dirs", _find)
    monkeypatch.setattr(shell_qc.shutil, "which", _which)

    calls: list[list[str]] = []

    def _run(command: list[str], *args: Any, **kwargs: Any) -> SimpleNamespace:
        calls.append(list(command))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(shell_qc.subprocess, "run", _run)

    exit_code = shell_qc.run_test()

    assert exit_code == 0
    assert calls == [["/usr/bin/bats", str(Path("tests/shell"))]]


def test_run_test_with_coverage_requires_bats(monkeypatch: MonkeyPatch) -> None:
    """When coverage is requested, missing bats should be treated as an error."""

    def _find(_: Path | None = None) -> list[Path]:
        return [Path("tests/shell")]

    def _which(tool: str) -> str | None:
        if tool == "bats":
            return None
        return "/usr/bin/tool"

    monkeypatch.setattr(shell_qc, "find_bats_test_dirs", _find)
    monkeypatch.setattr(shell_qc.shutil, "which", _which)

    exit_code = shell_qc.run_test_with_options(coverage=True)

    assert exit_code == 127


def test_run_test_with_coverage_invokes_kcov_and_merge(
    monkeypatch: MonkeyPatch,
) -> None:
    """Coverage mode with multiple test dirs should merge results."""

    def _find(_: Path | None = None) -> list[Path]:
        # Return multiple test directories to trigger merge behavior.
        return [Path("tests/shell"), Path("tests/other")]

    def _which(tool: str) -> str | None:
        if tool in {"bats", "kcov"}:
            return f"/usr/bin/{tool}"
        return "/usr/bin/tool"

    monkeypatch.setattr(shell_qc, "find_bats_test_dirs", _find)
    monkeypatch.setattr(shell_qc.shutil, "which", _which)

    prepared: dict[str, Path] = {}

    def _prepare(repo_root: Path, out_dir: Path) -> tuple[Path, Path]:
        prepared["repo_root"] = repo_root
        prepared["out_dir"] = out_dir
        return Path("artifacts/pester/kcov"), Path("artifacts/pester/kcov/.kcov_runs")

    def _cleanup(_: Path) -> None:
        return None

    monkeypatch.setattr(shell_qc, "_prepare_kcov_output_dirs", _prepare)
    monkeypatch.setattr(shell_qc, "_cleanup_kcov_runs_dir", _cleanup)

    calls: list[list[str]] = []

    def _run(command: list[str], *args: Any, **kwargs: Any) -> SimpleNamespace:
        calls.append(list(command))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(shell_qc.subprocess, "run", _run)

    exit_code = shell_qc.run_test_with_options(coverage=True)

    assert exit_code == 0
    # First two calls should be kcov running bats for each test directory.
    assert calls[0][0] == "/usr/bin/kcov"
    assert "--cobertura-only" in calls[0]
    assert calls[1][0] == "/usr/bin/kcov"
    assert "--cobertura-only" in calls[1]

    # Merge should be the last call when multiple test directories exist.
    assert calls[-1][:2] == ["/usr/bin/kcov", "--merge"]


def test_run_test_with_coverage_single_dir_skips_merge(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Coverage mode with single test dir should copy cov.xml directly, not merge.

    kcov --merge with --cobertura-only produces empty results for a single
    input directory. When there's only one test directory, we skip the merge
    and copy the coverage file directly.
    """

    def _find(_: Path | None = None) -> list[Path]:
        return [Path("tests/shell")]

    def _which(tool: str) -> str | None:
        if tool in {"bats", "kcov"}:
            return f"/usr/bin/{tool}"
        return "/usr/bin/tool"

    monkeypatch.setattr(shell_qc, "find_bats_test_dirs", _find)
    monkeypatch.setattr(shell_qc.shutil, "which", _which)

    # Set up realistic paths for the test.
    merged_output_dir = tmp_path / "kcov"
    runs_dir = merged_output_dir / ".kcov_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / "shell"
    run_dir.mkdir(parents=True, exist_ok=True)

    def _prepare(repo_root: Path, out_dir: Path) -> tuple[Path, Path]:
        return merged_output_dir, runs_dir

    def _cleanup(_: Path) -> None:
        return None

    monkeypatch.setattr(shell_qc, "_prepare_kcov_output_dirs", _prepare)
    monkeypatch.setattr(shell_qc, "_cleanup_kcov_runs_dir", _cleanup)

    calls: list[list[str]] = []

    def _run(command: list[str], *args: Any, **kwargs: Any) -> SimpleNamespace:
        calls.append(list(command))
        # Simulate kcov creating a cov.xml in the run directory.
        cov_xml = run_dir / "cov.xml"
        cov_xml.write_text('<coverage line-rate="0.5"/>')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(shell_qc.subprocess, "run", _run)

    exit_code = shell_qc.run_test_with_options(coverage=True)

    assert exit_code == 0

    # Should only have ONE kcov call (no merge).
    assert len(calls) == 1
    assert calls[0][0] == "/usr/bin/kcov"
    assert "--cobertura-only" in calls[0]
    assert "--merge" not in calls[0]

    # The cov.xml should have been copied to the kcov-merged directory.
    copied_cov = merged_output_dir / "kcov-merged" / "cov.xml"
    assert copied_cov.exists()
    assert copied_cov.read_text() == '<coverage line-rate="0.5"/>'


def test_parse_args_accepts_commands() -> None:
    assert shell_qc.parse_args(["check"]).command == "check"
    assert shell_qc.parse_args(["format"]).command == "format"
    assert shell_qc.parse_args(["test"]).command == "test"
    assert shell_qc.parse_args(["test", "--coverage"]).coverage is True


def test_main_dispatches_to_check(monkeypatch: MonkeyPatch) -> None:
    def _run_check(_: Path | None = None) -> int:
        return 0

    monkeypatch.setattr(shell_qc, "run_check", _run_check)

    assert shell_qc.main(["check"]) == 0


def test_main_dispatches_to_format(monkeypatch: MonkeyPatch) -> None:
    def _run_format(_: Path | None = None) -> int:
        return 0

    monkeypatch.setattr(shell_qc, "run_format", _run_format)

    assert shell_qc.main(["format"]) == 0


def test_main_dispatches_to_test(monkeypatch: MonkeyPatch) -> None:
    def _run_test_with_options(
        root: Path | None = None, *, coverage: bool = False, **_: Any
    ) -> int:
        assert root is None
        assert coverage is False
        return 0

    monkeypatch.setattr(shell_qc, "run_test_with_options", _run_test_with_options)

    assert shell_qc.main(["test"]) == 0


def test_main_check_wrapper(monkeypatch: MonkeyPatch) -> None:
    called = {"hit": False}

    def _run_check(root: Path | None = None) -> int:
        called["hit"] = True
        return 0

    monkeypatch.setattr(shell_qc, "run_check", _run_check)

    assert shell_qc.main_check() == 0
    assert called["hit"] is True


def test_main_format_wrapper(monkeypatch: MonkeyPatch) -> None:
    called = {"hit": False}

    def _run_format(root: Path | None = None) -> int:
        called["hit"] = True
        return 0

    monkeypatch.setattr(shell_qc, "run_format", _run_format)

    assert shell_qc.main_format() == 0
    assert called["hit"] is True


def test_main_test_wrapper(monkeypatch: MonkeyPatch) -> None:
    called = {"hit": False}

    def _run_test_with_options(root: Path | None = None, **_: Any) -> int:
        called["hit"] = True
        return 0

    monkeypatch.setattr(shell_qc, "run_test_with_options", _run_test_with_options)

    assert shell_qc.main_test() == 0
    assert called["hit"] is True


# ---------------------------------------------------------------------------
# Tests for _extract_cobertura_line_rate and _print_coverage_summary
# ---------------------------------------------------------------------------


class MockPath:
    """Mock Path that returns controlled content without filesystem access."""

    def __init__(self, content: str | None, exists: bool = True) -> None:
        self._content = content
        self._exists = exists

    def exists(self) -> bool:
        return self._exists

    def read_text(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        if self._content is None:
            raise OSError("Mock read_text failure")
        return self._content


def test_extract_cobertura_line_rate_parses_double_quotes() -> None:
    """Extracts line-rate from double-quoted attribute."""
    content = '<?xml version="1.0"?><coverage line-rate="0.75">'
    mock_path = MockPath(content)

    result = shell_qc._extract_cobertura_line_rate(mock_path)  # type: ignore[arg-type]

    assert result == 0.75


def test_extract_cobertura_line_rate_parses_single_quotes() -> None:
    """Extracts line-rate from single-quoted attribute."""
    content = "<?xml version='1.0'?><coverage line-rate='0.5'>"
    mock_path = MockPath(content)

    result = shell_qc._extract_cobertura_line_rate(mock_path)  # type: ignore[arg-type]

    assert result == 0.5


def test_extract_cobertura_line_rate_returns_none_when_missing_file() -> None:
    """Returns None when the file does not exist."""
    mock_path = MockPath(content=None, exists=False)

    result = shell_qc._extract_cobertura_line_rate(mock_path)  # type: ignore[arg-type]

    assert result is None


def test_extract_cobertura_line_rate_returns_none_when_read_fails() -> None:
    """Returns None when read_text raises OSError."""
    mock_path = MockPath(content=None, exists=True)

    result = shell_qc._extract_cobertura_line_rate(mock_path)  # type: ignore[arg-type]

    assert result is None


def test_extract_cobertura_line_rate_returns_none_when_no_match() -> None:
    """Returns None when line-rate attribute is not present."""
    content = '<?xml version="1.0"?><coverage>'
    mock_path = MockPath(content)

    result = shell_qc._extract_cobertura_line_rate(mock_path)  # type: ignore[arg-type]

    assert result is None


def test_extract_cobertura_line_rate_returns_none_when_invalid_value() -> None:
    """Returns None when line-rate value is not a valid float."""
    content = '<?xml version="1.0"?><coverage line-rate="invalid">'
    mock_path = MockPath(content)

    result = shell_qc._extract_cobertura_line_rate(mock_path)  # type: ignore[arg-type]

    assert result is None


def test_print_coverage_summary_outputs_percentage(
    monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Prints formatted percentage when line-rate is available."""

    def _extract(path: Path) -> float | None:
        return 0.85

    monkeypatch.setattr(shell_qc, "_extract_cobertura_line_rate", _extract)

    # Private function access is expected for unit testing internal helpers.
    shell_qc._print_coverage_summary(  # pyright: ignore[reportPrivateUsage]
        Path("cov.xml")
    )

    output = capsys.readouterr().out
    assert "Bash coverage (lines): 85.0%" in output


def test_print_coverage_summary_silent_when_none(
    monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No output when line-rate cannot be extracted."""

    def _extract(path: Path) -> float | None:
        return None

    monkeypatch.setattr(shell_qc, "_extract_cobertura_line_rate", _extract)

    # Private function access is expected for unit testing internal helpers.
    shell_qc._print_coverage_summary(  # pyright: ignore[reportPrivateUsage]
        Path("cov.xml")
    )

    output = capsys.readouterr().out
    assert output == ""


def test_run_test_with_coverage_prints_summary(
    monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Coverage mode should print summary after successful run."""

    def _find(_: Path | None = None) -> list[Path]:
        return [Path("tests/shell")]

    def _which(tool: str) -> str | None:
        if tool in {"bats", "kcov"}:
            return f"/usr/bin/{tool}"
        return "/usr/bin/tool"

    monkeypatch.setattr(shell_qc, "find_bats_test_dirs", _find)
    monkeypatch.setattr(shell_qc.shutil, "which", _which)

    def _prepare(repo_root: Path, out_dir: Path) -> tuple[Path, Path]:
        return Path("artifacts/pester/kcov"), Path("artifacts/pester/kcov/.kcov_runs")

    def _cleanup(_: Path) -> None:
        return None

    monkeypatch.setattr(shell_qc, "_prepare_kcov_output_dirs", _prepare)
    monkeypatch.setattr(shell_qc, "_cleanup_kcov_runs_dir", _cleanup)

    def _run(command: list[str], *args: Any, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(shell_qc.subprocess, "run", _run)

    # Mock the summary print to verify it's called with correct path.
    printed: dict[str, Path | None] = {"path": None}

    def _print_summary(cov_xml_path: Path) -> None:
        printed["path"] = cov_xml_path
        print("Bash coverage (lines): 80.0%")

    monkeypatch.setattr(shell_qc, "_print_coverage_summary", _print_summary)

    exit_code = shell_qc.run_test_with_options(coverage=True)

    assert exit_code == 0
    # kcov --merge creates a 'kcov-merged' subdirectory inside the output dir.
    assert printed["path"] == Path("artifacts/pester/kcov/kcov-merged/cov.xml")
    assert "Bash coverage (lines):" in capsys.readouterr().out
