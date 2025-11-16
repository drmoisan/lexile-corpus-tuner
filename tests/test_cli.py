import json
from pathlib import Path

from typer.testing import CliRunner

from lexile_corpus_tuner.cli import app

runner = CliRunner()


def test_cli_analyze_outputs_summary():
    """CLI analyze command returns JSON summary listing documents."""
    result = runner.invoke(
        app, ["analyze", "--input-path", "examples/example_corpus"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "documents" in payload
    assert payload["documents"][0]["doc_id"].endswith(".txt")


def test_cli_rewrite_writes_files(tmp_path: Path):
    """rewrite command emits tuned text and summary files in the output dir."""
    output_dir = tmp_path / "tuned"
    result = runner.invoke(
        app,
        [
            "rewrite",
            "--input-path",
            "examples/example_corpus",
            "--output-path",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0
    tuned_file = output_dir / "chapter1.txt"
    summary_file = output_dir / "summary.json"
    assert tuned_file.exists()
    assert summary_file.exists()
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    assert "documents" in summary


def test_cli_print_config():
    """print-config command dumps the current configuration values."""
    result = runner.invoke(app, ["print-config"])
    assert result.exit_code == 0
    assert "window_size" in result.stdout
