from __future__ import annotations

import json
from pathlib import Path

import click

from .adjustments import adjust_for_special_cases
from .features import compute_document_features
from .model import estimate_lexile_from_features
from .slices import build_slices


@click.group(name="analyze")
def analyze_group() -> None:
    """Lexile-style text analysis commands."""


@analyze_group.command("text")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--json-output", type=click.Path(), default=None)
@click.option(
    "--picture-book",
    is_flag=True,
    default=False,
    help="Apply picture-book adjustment (-120L).",
)
@click.option(
    "--emergent-nonfiction",
    is_flag=True,
    default=False,
    help="Apply emergent nonfiction adjustment (-120L).",
)
def analyze_text(
    input_file: str,
    json_output: str | None,
    picture_book: bool,
    emergent_nonfiction: bool,
) -> None:
    """Analyze a text file and print Lexile-style features + estimated score."""
    path = Path(input_file)
    text = path.read_text(encoding="utf-8")
    slices = build_slices(text)
    doc_features = compute_document_features(slices)
    raw_lexile = estimate_lexile_from_features(doc_features)
    adjusted_lexile = adjust_for_special_cases(
        raw_lexile,
        is_picture_book=picture_book,
        is_emergent_nonfiction=emergent_nonfiction,
    )

    click.echo(f"File: {input_file}")
    click.echo(f"Num slices: {doc_features.num_slices}")
    click.echo(f"Total tokens: {doc_features.total_tokens}")
    click.echo(f"Overall MSL: {doc_features.overall_mean_sentence_length:.2f}")
    click.echo(f"Overall MLF: {doc_features.overall_mean_log_word_freq:.4f}")
    click.echo(f"Estimated Lexile (raw): {raw_lexile:.1f}L")
    click.echo(f"Estimated Lexile (adjusted): {adjusted_lexile:.1f}L")

    if json_output is not None:
        payload = {
            "file": str(path),
            "raw_lexile": raw_lexile,
            "adjusted_lexile": adjusted_lexile,
            "document_features": {
                "num_slices": doc_features.num_slices,
                "total_tokens": doc_features.total_tokens,
                "overall_msl": doc_features.overall_mean_sentence_length,
                "overall_mlf": doc_features.overall_mean_log_word_freq,
            },
            "slice_features": [
                {
                    "slice_id": sf.slice_id,
                    "num_tokens": sf.num_tokens,
                    "num_sentences": sf.num_sentences,
                    "mean_sentence_length": sf.mean_sentence_length,
                    "mean_log_word_freq": sf.mean_log_word_freq,
                }
                for sf in doc_features.slice_features
            ],
        }
        json_path = Path(json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        click.echo(f"Wrote detailed JSON to {json_output}")
