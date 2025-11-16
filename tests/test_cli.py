import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from lexile_corpus_tuner.cli import app

runner = CliRunner()


def test_cli_analyze_outputs_summary():
    """CLI analyze command returns JSON summary listing documents."""
    result = runner.invoke(app, ["analyze", "--input-path", "examples/example_corpus"])
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


def test_cli_rewrite_with_openai_options(monkeypatch, tmp_path: Path):
    """rewrite command wires OpenAI settings into the rewriter when enabled."""
    output_dir = tmp_path / "rewritten"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "max_window_lexile: 200\nrewrite_enabled: true\n", encoding="utf-8"
    )
    calls: dict[str, Any] = {}

    class DummyClient:
        def __init__(self, settings, api_key) -> None:
            calls["settings"] = settings
            calls["api_key"] = api_key

        def rewrite(self, *, system_prompt, user_prompt, metadata):
            calls.setdefault("prompts", []).append(
                {"system": system_prompt, "user": user_prompt, "metadata": metadata}
            )
            return user_prompt

    class DummyRewriter:
        def __init__(self, client, **_ignored) -> None:
            self._client = client

        def rewrite(self, request):
            return request.text.replace("storm", "rain")

    monkeypatch.setattr("lexile_corpus_tuner.cli.OpenAIRewriteClient", DummyClient)
    monkeypatch.setattr("lexile_corpus_tuner.cli.OpenAIRewriter", DummyRewriter)

    result = runner.invoke(
        app,
        [
            "rewrite",
            "--input-path",
            "examples/example_corpus",
            "--output-path",
            str(output_dir),
            "--config",
            str(config_path),
            "--rewrite-enabled",
            "--openai-enabled",
            "--openai-model",
            "gpt-4.1-mini",
        ],
        env={"OPENAI_API_KEY": "dummy-key"},
    )
    assert result.exit_code == 0
    tuned_file = output_dir / "chapter1.txt"
    assert tuned_file.exists()
    tuned_text = tuned_file.read_text(encoding="utf-8")
    assert "rain" in tuned_text
    assert calls["settings"].model == "gpt-4.1-mini"
    assert calls["api_key"] == "dummy-key"
