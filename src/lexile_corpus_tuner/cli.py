from __future__ import annotations

import json
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Dict, List, Tuple

import typer
import yaml

from .config import LexileTunerConfig, load_config
from .estimators import build_estimator_from_config
from .models import ConstraintViolation, Document, DocumentLexileStats
from .pipeline import process_corpus
from .rewriting import LLMRewriter, NoOpRewriter, Rewriter

app = typer.Typer(help="Lexile Corpus Tuner CLI.", no_args_is_help=True)


@app.command()
def analyze(
    input_path: Path = typer.Option(..., exists=True, readable=True, dir_okay=True, file_okay=True),
    config: Path | None = typer.Option(None, "--config", "-c"),
    estimator_name: str | None = typer.Option(
        None,
        "--estimator-name",
        "-e",
        help="Estimator to use (e.g., 'dummy' or 'lexile_v2').",
    ),
    lexile_v2_model_path: Path | None = typer.Option(
        None, help="Path to lexile-determination-v2 Keras model."
    ),
    lexile_v2_vectorizer_path: Path | None = typer.Option(
        None, help="Path to lexile-determination-v2 vectorizer."
    ),
    lexile_v2_label_encoder_path: Path | None = typer.Option(
        None, help="Path to lexile-determination-v2 label encoder."
    ),
) -> None:
    """Analyze the input corpus and emit a JSON summary."""
    cfg = load_config(config)
    _apply_estimator_overrides(
        cfg,
        estimator_name,
        lexile_v2_model_path,
        lexile_v2_vectorizer_path,
        lexile_v2_label_encoder_path,
    )
    cfg.rewrite_enabled = False
    documents, _ = _load_documents(input_path)
    estimator = build_estimator_from_config(cfg)
    summary = _build_summary(
        process_corpus(documents, cfg, estimator, NoOpRewriter()),
    )
    typer.echo(json.dumps({"documents": summary}, indent=2))


@app.command()
def rewrite(
    input_path: Path = typer.Option(..., exists=True, readable=True, dir_okay=True, file_okay=True),
    output_path: Path = typer.Option(..., file_okay=False),
    config: Path | None = typer.Option(None, "--config", "-c"),
    estimator_name: str | None = typer.Option(
        None,
        "--estimator-name",
        "-e",
        help="Estimator to use (e.g., 'dummy' or 'lexile_v2').",
    ),
    lexile_v2_model_path: Path | None = typer.Option(
        None, help="Path to lexile-determination-v2 Keras model."
    ),
    lexile_v2_vectorizer_path: Path | None = typer.Option(
        None, help="Path to lexile-determination-v2 vectorizer."
    ),
    lexile_v2_label_encoder_path: Path | None = typer.Option(
        None, help="Path to lexile-determination-v2 label encoder."
    ),
) -> None:
    """Rewrite violating windows and save tuned documents + summary."""
    cfg = load_config(config)
    _apply_estimator_overrides(
        cfg,
        estimator_name,
        lexile_v2_model_path,
        lexile_v2_vectorizer_path,
        lexile_v2_label_encoder_path,
    )
    documents, _ = _load_documents(input_path)
    estimator = build_estimator_from_config(cfg)

    baseline_cfg = dc_replace(cfg, rewrite_enabled=False)
    baseline_results = process_corpus(documents, baseline_cfg, estimator, NoOpRewriter())

    rewrite_cfg = dc_replace(cfg, rewrite_enabled=True)
    rewriter = _build_rewriter(rewrite_cfg)
    final_results = process_corpus(documents, rewrite_cfg, estimator, rewriter)

    output_path.mkdir(parents=True, exist_ok=True)
    for doc_id, (final_doc, _, _) in final_results.items():
        dest = output_path / doc_id
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(final_doc.text, encoding="utf-8")

    summary_items = []
    for doc_id in sorted(final_results.keys()):
        baseline_entry = baseline_results.get(doc_id)
        final_entry = final_results.get(doc_id)
        if baseline_entry is None or final_entry is None:
            continue
        summary_items.append(
            {
                "doc_id": doc_id,
                "baseline": _stats_dict(baseline_entry[1], baseline_entry[2]),
                "final": _stats_dict(final_entry[1], final_entry[2]),
            }
        )

    summary_path = output_path / "summary.json"
    summary_path.write_text(json.dumps({"documents": summary_items}, indent=2), encoding="utf-8")
    typer.echo(f"Wrote tuned documents to {output_path} and summary to {summary_path}")


@app.command("print-config")
def print_config() -> None:
    """Print the default configuration as YAML."""
    cfg = LexileTunerConfig()
    typer.echo(yaml.safe_dump(cfg.to_dict(), sort_keys=False))


def main() -> None:
    app()


def _apply_estimator_overrides(
    config: LexileTunerConfig,
    estimator_name: str | None,
    model_path: Path | None,
    vectorizer_path: Path | None,
    label_encoder_path: Path | None,
) -> None:
    if estimator_name:
        config.estimator_name = estimator_name
    if model_path:
        config.lexile_v2_model_path = str(model_path)
    if vectorizer_path:
        config.lexile_v2_vectorizer_path = str(vectorizer_path)
    if label_encoder_path:
        config.lexile_v2_label_encoder_path = str(label_encoder_path)


def _build_rewriter(config: LexileTunerConfig) -> Rewriter:
    if config.rewrite_enabled and config.rewrite_model:
        typer.echo(
            "LLM rewriting configured but no client provided; using placeholder behaviour.",
            err=True,
        )
        return LLMRewriter(
            model_name=config.rewrite_model,
            completion_fn=lambda prompt, text, target: text,
        )
    return NoOpRewriter()


def _load_documents(input_path: Path) -> Tuple[List[Document], Dict[str, Path]]:
    if input_path.is_file():
        text = input_path.read_text(encoding="utf-8")
        doc = Document(doc_id=input_path.name, text=text)
        return [doc], {doc.doc_id: input_path}

    files = sorted(p for p in input_path.rglob("*.txt") if p.is_file())
    documents: List[Document] = []
    mapping: Dict[str, Path] = {}
    for file in files:
        relative_id = str(file.relative_to(input_path))
        text = file.read_text(encoding="utf-8")
        documents.append(Document(doc_id=relative_id, text=text))
        mapping[relative_id] = file
    return documents, mapping


def _build_summary(
    results: Dict[
        str, Tuple[Document, DocumentLexileStats, List[ConstraintViolation]]
    ]
) -> List[dict]:
    summary: List[dict] = []
    for doc_id, (_, stats, violations) in sorted(results.items()):
        summary.append(
            {
                "doc_id": doc_id,
                "avg_lexile": stats.avg_lexile,
                "max_lexile": stats.max_lexile,
                "violations": [_violation_dict(v) for v in violations],
            }
        )
    return summary


def _stats_dict(stats: DocumentLexileStats, violations: List[ConstraintViolation]) -> dict:
    return {
        "avg_lexile": stats.avg_lexile,
        "max_lexile": stats.max_lexile,
        "violations": [_violation_dict(v) for v in violations],
    }


def _violation_dict(violation: ConstraintViolation) -> dict:
    return {
        "window_id": violation.window_id,
        "lexile": violation.lexile,
        "reason": violation.reason,
        "start_token_idx": violation.start_token_idx,
        "end_token_idx": violation.end_token_idx,
    }
