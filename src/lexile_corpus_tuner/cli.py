from __future__ import annotations

import json
import os
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Dict, List, Tuple

import typer
import yaml

from .config import LexileTunerConfig, OpenAISettings, load_config
from .epub import EPUBParseError, extract_text_from_epub
from .estimators import build_estimator_from_config
from .llm import OpenAIRewriteClient
from .models import ConstraintViolation, Document, DocumentLexileStats
from .pipeline import process_corpus
from .rewriting import NoOpRewriter, OpenAIRewriter, Rewriter

app = typer.Typer(help="Lexile Corpus Tuner CLI.", no_args_is_help=True)


@app.command()
def analyze(
    input_path: Path = typer.Option(
        ..., exists=True, readable=True, dir_okay=True, file_okay=True
    ),
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
    rewrite_enabled: bool | None = typer.Option(
        None,
        "--rewrite-enabled/--no-rewrite-enabled",
        help="Override config rewrite_enabled flag.",
    ),
    rewrite_model: str | None = typer.Option(
        None, "--rewrite-model", help="Override rewrite_model config value."
    ),
    openai_enabled: bool | None = typer.Option(
        None,
        "--openai-enabled/--openai-disabled",
        help="Toggle OpenAI-backed rewriting.",
    ),
    openai_model: str | None = typer.Option(
        None, "--openai-model", help="OpenAI model identifier (e.g., gpt-4.1-mini)."
    ),
    openai_api_key: str | None = typer.Option(
        None, "--openai-api-key", help="Explicit OpenAI API key (prefer env vars)."
    ),
    openai_api_key_env: str | None = typer.Option(
        None,
        "--openai-api-key-env",
        help="Environment variable to read the OpenAI API key from.",
    ),
    openai_base_url: str | None = typer.Option(
        None, "--openai-base-url", help="Custom OpenAI base URL (Azure, proxy, etc.)."
    ),
    openai_organization: str | None = typer.Option(
        None, "--openai-organization", help="OpenAI organization identifier."
    ),
    openai_temperature: float | None = typer.Option(
        None, "--openai-temperature", help="Sampling temperature for rewrites."
    ),
    openai_max_output_tokens: int | None = typer.Option(
        None, "--openai-max-output-tokens", help="Max tokens the rewrite can emit."
    ),
    openai_top_p: float | None = typer.Option(
        None, "--openai-top-p", help="top_p value for nucleus sampling."
    ),
    openai_request_timeout: float | None = typer.Option(
        None, "--openai-request-timeout", help="Request timeout (seconds)."
    ),
    openai_parallel_requests: int | None = typer.Option(
        None,
        "--openai-parallel-requests",
        help="Max parallel OpenAI calls (1 = sequential).",
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
    _apply_rewriter_overrides(
        cfg,
        rewrite_enabled,
        rewrite_model,
        openai_enabled,
        openai_model,
        openai_api_key,
        openai_api_key_env,
        openai_base_url,
        openai_organization,
        openai_temperature,
        openai_max_output_tokens,
        openai_top_p,
        openai_request_timeout,
        openai_parallel_requests,
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
    input_path: Path = typer.Option(
        ..., exists=True, readable=True, dir_okay=True, file_okay=True
    ),
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
    rewrite_enabled: bool | None = typer.Option(
        None,
        "--rewrite-enabled/--no-rewrite-enabled",
        help="Override config rewrite_enabled flag.",
    ),
    rewrite_model: str | None = typer.Option(
        None, "--rewrite-model", help="Override rewrite_model config value."
    ),
    openai_enabled: bool | None = typer.Option(
        None,
        "--openai-enabled/--openai-disabled",
        help="Toggle OpenAI-backed rewriting.",
    ),
    openai_model: str | None = typer.Option(
        None, "--openai-model", help="OpenAI model identifier (e.g., gpt-4.1-mini)."
    ),
    openai_api_key: str | None = typer.Option(
        None, "--openai-api-key", help="Explicit OpenAI API key (prefer env vars)."
    ),
    openai_api_key_env: str | None = typer.Option(
        None,
        "--openai-api-key-env",
        help="Environment variable to read the OpenAI API key from.",
    ),
    openai_base_url: str | None = typer.Option(
        None, "--openai-base-url", help="Custom OpenAI base URL (Azure, proxy, etc.)."
    ),
    openai_organization: str | None = typer.Option(
        None, "--openai-organization", help="OpenAI organization identifier."
    ),
    openai_temperature: float | None = typer.Option(
        None, "--openai-temperature", help="Sampling temperature for rewrites."
    ),
    openai_max_output_tokens: int | None = typer.Option(
        None, "--openai-max-output-tokens", help="Max tokens the rewrite can emit."
    ),
    openai_top_p: float | None = typer.Option(
        None, "--openai-top-p", help="top_p value for nucleus sampling."
    ),
    openai_request_timeout: float | None = typer.Option(
        None, "--openai-request-timeout", help="Request timeout (seconds)."
    ),
    openai_parallel_requests: int | None = typer.Option(
        None,
        "--openai-parallel-requests",
        help="Max parallel OpenAI calls (1 = sequential).",
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
    _apply_rewriter_overrides(
        cfg,
        rewrite_enabled,
        rewrite_model,
        openai_enabled,
        openai_model,
        openai_api_key,
        openai_api_key_env,
        openai_base_url,
        openai_organization,
        openai_temperature,
        openai_max_output_tokens,
        openai_top_p,
        openai_request_timeout,
        openai_parallel_requests,
    )
    documents, _ = _load_documents(input_path)
    estimator = build_estimator_from_config(cfg)

    baseline_cfg = dc_replace(cfg, rewrite_enabled=False)
    baseline_results = process_corpus(
        documents, baseline_cfg, estimator, NoOpRewriter()
    )

    rewrite_cfg = dc_replace(cfg, rewrite_enabled=True)
    rewriter = _build_rewriter(rewrite_cfg)
    final_results = process_corpus(documents, rewrite_cfg, estimator, rewriter)

    output_path.mkdir(parents=True, exist_ok=True)
    for doc_id, (final_doc, _, _) in final_results.items():
        dest = output_path / _relative_output_path(doc_id)
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
    summary_path.write_text(
        json.dumps({"documents": summary_items}, indent=2), encoding="utf-8"
    )
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


def _apply_rewriter_overrides(
    config: LexileTunerConfig,
    rewrite_enabled: bool | None,
    rewrite_model: str | None,
    openai_enabled: bool | None,
    openai_model: str | None,
    openai_api_key: str | None,
    openai_api_key_env: str | None,
    openai_base_url: str | None,
    openai_organization: str | None,
    openai_temperature: float | None,
    openai_max_output_tokens: int | None,
    openai_top_p: float | None,
    openai_request_timeout: float | None,
    openai_parallel_requests: int | None,
) -> None:
    if rewrite_enabled is not None:
        config.rewrite_enabled = rewrite_enabled
    if rewrite_model is not None:
        config.rewrite_model = rewrite_model
    settings = config.openai
    if openai_enabled is not None:
        settings.enabled = openai_enabled
    if openai_model:
        settings.model = openai_model
    if openai_api_key:
        settings.api_key = openai_api_key
    if openai_api_key_env:
        settings.api_key_env = openai_api_key_env
    if openai_base_url:
        settings.base_url = openai_base_url
    if openai_organization:
        settings.organization = openai_organization
    if openai_temperature is not None:
        settings.temperature = openai_temperature
    if openai_max_output_tokens is not None:
        settings.max_output_tokens = openai_max_output_tokens
    if openai_top_p is not None:
        settings.top_p = openai_top_p
    if openai_request_timeout is not None:
        settings.request_timeout = openai_request_timeout
    if openai_parallel_requests is not None:
        settings.parallel_requests = openai_parallel_requests


def _build_rewriter(config: LexileTunerConfig) -> Rewriter:
    if not config.rewrite_enabled:
        return NoOpRewriter()
    if config.openai.enabled:
        api_key = _resolve_openai_api_key(config.openai)
        client = OpenAIRewriteClient(config.openai, api_key=api_key)
        return OpenAIRewriter(client)
    typer.echo(
        "Rewriting enabled but no LLM client configured; returning original text.",
        err=True,
    )
    return NoOpRewriter()


SUPPORTED_INPUT_EXTENSIONS = {".txt", ".epub"}


def _load_documents(input_path: Path) -> Tuple[List[Document], Dict[str, Path]]:
    if input_path.is_file():
        doc = _document_from_file(input_path, input_path.name)
        return [doc], {doc.doc_id: input_path}

    files = sorted(
        p
        for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_INPUT_EXTENSIONS
    )
    documents: List[Document] = []
    mapping: Dict[str, Path] = {}
    for file in files:
        relative_id = str(file.relative_to(input_path))
        documents.append(_document_from_file(file, relative_id))
        mapping[relative_id] = file
    return documents, mapping


def _document_from_file(path: Path, doc_id: str) -> Document:
    suffix = path.suffix.lower()
    try:
        if suffix == ".epub":
            text = extract_text_from_epub(path)
        else:
            text = path.read_text(encoding="utf-8")
    except EPUBParseError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return Document(doc_id=doc_id, text=text)


def _relative_output_path(doc_id: str) -> Path:
    target = Path(doc_id)
    if target.suffix.lower() == ".epub":
        return target.with_suffix(".txt")
    return target


def _build_summary(
    results: Dict[str, Tuple[Document, DocumentLexileStats, List[ConstraintViolation]]],
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


def _stats_dict(
    stats: DocumentLexileStats, violations: List[ConstraintViolation]
) -> dict:
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


def _resolve_openai_api_key(settings: OpenAISettings) -> str:
    if settings.api_key:
        return settings.api_key
    env_name = settings.api_key_env or "OPENAI_API_KEY"
    if env_name and env_name in os.environ:
        return os.environ[env_name]
    raise RuntimeError(
        "OpenAI API key not provided. Use --openai-api-key or set the configured environment variable."
    )
