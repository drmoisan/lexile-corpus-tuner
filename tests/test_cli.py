import json
from pathlib import Path
from typing import Any

from pytest import MonkeyPatch
from typer.testing import CliRunner

from lexile_corpus_tuner.cli import app
from tests.utils import write_minimal_epub

runner = CliRunner()


def test_cli_analyze_outputs_summary(tmp_path: Path):
    """CLI analyze command returns JSON summary listing .txt and .epub docs."""
    corpus_dir = _create_sample_corpus(tmp_path)
    result = runner.invoke(app, ["analyze", "--input-path", str(corpus_dir)])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "documents" in payload
    doc_ids = [doc["doc_id"] for doc in payload["documents"]]
    assert any(doc_id.endswith(".txt") for doc_id in doc_ids)
    assert any(doc_id.endswith(".epub") for doc_id in doc_ids)


def test_cli_rewrite_writes_files(tmp_path: Path):
    """rewrite command emits tuned text and summary files in the output dir."""
    corpus_dir = _create_sample_corpus(tmp_path)
    output_dir = tmp_path / "tuned"
    result = runner.invoke(
        app,
        [
            "rewrite",
            "--input-path",
            str(corpus_dir),
            "--output-path",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0
    tuned_txt = output_dir / "chapter1.txt"
    tuned_epub = output_dir / "novella.txt"
    summary_file = output_dir / "summary.json"
    assert tuned_txt.exists()
    assert tuned_epub.exists()
    assert summary_file.exists()
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    assert "documents" in summary


def test_cli_print_config():
    """print-config command dumps the current configuration values."""
    result = runner.invoke(app, ["print-config"])
    assert result.exit_code == 0
    assert "window_size" in result.stdout


def test_cli_rewrite_with_openai_options(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """rewrite command wires OpenAI settings into the rewriter when enabled."""
    corpus_dir = _create_sample_corpus(tmp_path)
    output_dir = tmp_path / "rewritten"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "max_window_lexile: 200\nrewrite_enabled: true\n", encoding="utf-8"
    )
    calls: dict[str, Any] = {}

    class DummyClient:
        def __init__(self, settings: Any, api_key: str) -> None:
            calls["settings"] = settings
            calls["api_key"] = api_key

        def rewrite(
            self, *, system_prompt: str, user_prompt: str, metadata: Any
        ) -> str:
            calls.setdefault("prompts", []).append(
                {"system": system_prompt, "user": user_prompt, "metadata": metadata}
            )
            return user_prompt

    class DummyRewriter:
        def __init__(self, client: DummyClient, **_ignored: Any) -> None:
            self._client = client

        def rewrite(self, request: Any) -> str:
            return request.text.replace("storm", "rain")

    monkeypatch.setattr("lexile_corpus_tuner.cli.OpenAIRewriteClient", DummyClient)
    monkeypatch.setattr("lexile_corpus_tuner.cli.OpenAIRewriter", DummyRewriter)

    result = runner.invoke(
        app,
        [
            "rewrite",
            "--input-path",
            str(corpus_dir),
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


def test_cli_analyze_accepts_epub_file(tmp_path: Path):
    """CLI analyze command ingests a standalone EPUB file."""
    epub_path = tmp_path / "single.epub"
    write_minimal_epub(
        epub_path,
        chapters=[
            "<html xmlns='http://www.w3.org/1999/xhtml'><body><p>Single doc.</p></body></html>"
        ],
    )
    result = runner.invoke(app, ["analyze", "--input-path", str(epub_path)])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["documents"][0]["doc_id"] == "single.epub"


def _create_sample_corpus(tmp_path: Path) -> Path:
    """Create a small corpus containing both .txt and .epub sources."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "chapter1.txt").write_text(
        "The storm clouds rolled over the bay. Sailors watched the winds.",
        encoding="utf-8",
    )
    write_minimal_epub(
        corpus_dir / "novella.epub",
        chapters=[
            "<html xmlns='http://www.w3.org/1999/xhtml'><body><p>The captain stood on deck.</p></body></html>"
        ],
    )
    return corpus_dir
