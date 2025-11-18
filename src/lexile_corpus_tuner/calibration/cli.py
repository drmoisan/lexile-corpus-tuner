from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, cast

import click
import pandas as pd
import requests
from bs4 import BeautifulSoup

from ..analyzer.features import compute_document_features
from ..analyzer.slices import build_slices
from ..textutils import normalize_text
from .featureset import make_regression_features
from .model_store import save_model
from .train import FEATURE_COLS, train_regression_model

LOGGER = logging.getLogger(__name__)


@click.group(name="calibration")
def calibration_group() -> None:
    """Commands for Lexile regression calibration."""


@calibration_group.command("fetch-texts")
@click.option("--catalog", type=click.Path(exists=True), required=True)
@click.option("--texts-root", type=click.Path(), required=True)
def fetch_texts(catalog: str, texts_root: str) -> None:
    """Ensure calibration catalog texts exist locally."""
    rows = _read_catalog(Path(catalog))
    texts_dir = Path(texts_root)
    texts_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    pending_manual = 0
    failures = 0

    for row in rows:
        text_id = row.get("text_id")
        if not text_id:
            continue
        dest = texts_dir / f"{text_id}.txt"
        if dest.exists():
            skipped += 1
            continue

        acquisition_type = (row.get("acquisition_type") or "manual").lower()
        acquisition_key = row.get("acquisition_key") or ""

        try:
            if acquisition_type in {"local", "manual"}:
                click.echo(
                    f"[pending] {text_id}: place text at {dest} manually.",
                    err=True,
                )
                pending_manual += 1
            elif acquisition_type == "gutenberg":
                _fetch_gutenberg_text(acquisition_key, dest)
                downloaded += 1
            elif acquisition_type == "http":
                _fetch_http_text(acquisition_key, dest)
                downloaded += 1
            else:
                click.echo(
                    f"[warn] {text_id}: unsupported acquisition type {acquisition_type}",
                    err=True,
                )
                failures += 1
        except Exception as exc:  # noqa: broad-except
            failures += 1
            click.echo(f"[error] {text_id}: {exc}", err=True)

    click.echo(
        f"Downloaded {downloaded} texts, {skipped} already existed, "
        f"{pending_manual} pending manual placement, {failures} failures."
    )


@calibration_group.command("build-dataset")
@click.option("--catalog", type=click.Path(exists=True), required=True)
@click.option("--texts-root", type=click.Path(exists=True), required=True)
@click.option("--output", type=click.Path(), required=True)
@click.option("--strict/--no-strict", default=False, show_default=True)
def build_dataset(
    catalog: str,
    texts_root: str,
    output: str,
    strict: bool,
) -> None:
    """Analyze catalog texts and emit a calibration dataset."""
    rows = _read_catalog(Path(catalog))
    texts_dir = Path(texts_root)
    dataset_rows: list[dict[str, Any]] = []
    missing = 0

    for row in rows:
        text_id = row.get("text_id")
        if not text_id:
            continue
        text_path = texts_dir / f"{text_id}.txt"
        if not text_path.exists():
            msg = f"Missing text for {text_id} at {text_path}"
            if strict:
                raise click.ClickException(msg)
            click.echo(f"[warn] {msg}", err=True)
            missing += 1
            continue

        lexile_value = _parse_lexile_value(row.get("lexile_official"))
        if lexile_value is None:
            click.echo(
                f"[warn] Skipping {text_id}: lexile_official missing or invalid.",
                err=True,
            )
            continue

        text = text_path.read_text(encoding="utf-8")
        slices = build_slices(text)
        doc_features = compute_document_features(slices)
        feature_dict = make_regression_features(doc_features)

        record: dict[str, Any] = {
            "text_id": text_id,
            "title": row.get("title", ""),
            "author": row.get("author", ""),
            "grade_band": row.get("grade_band", ""),
            "lexile_source": row.get("lexile_source", ""),
            "lexile_official": float(lexile_value),
        }
        record.update(feature_dict)
        record["num_tokens"] = doc_features.total_tokens
        record["num_slices"] = doc_features.num_slices
        record["overall_msl"] = doc_features.overall_mean_sentence_length
        record["overall_mlf"] = doc_features.overall_mean_log_word_freq
        dataset_rows.append(record)

    if not dataset_rows:
        raise click.ClickException("No rows were added to the calibration dataset.")

    df = cast(Any, pd.DataFrame(dataset_rows))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        df.to_csv(output_path, index=False)
    else:
        df.to_parquet(output_path, index=False)

    click.echo(
        f"Wrote {len(dataset_rows)} calibration rows to {output_path} "
        f"(missing texts skipped: {missing})."
    )


@calibration_group.command("fit")
@click.argument("dataset", type=click.Path(exists=True))
@click.option(
    "--out",
    type=click.Path(),
    default="data/model/lexile_regression_model.json",
    show_default=True,
)
def fit(dataset: str, out: str) -> None:
    """Fit regression model on a calibration dataset and save JSON spec."""
    dataset_path = Path(dataset)
    out_path = Path(out)
    if dataset_path.suffix.lower() == ".csv":
        df = cast(
            Any,
            pd.read_csv(dataset_path),  # pyright: ignore[reportUnknownMemberType]
        )
    else:
        df = cast(
            Any,
            pd.read_parquet(dataset_path),  # pyright: ignore[reportUnknownMemberType]
        )
    model, metrics = train_regression_model(df)
    save_model(model, metrics, FEATURE_COLS, out_path)
    click.echo(
        f"Saved model to {out_path}. "
        f"Validation RMSE: {metrics['rmse']:.1f}L, "
        f"MAE: {metrics['mae']:.1f}L, r: {metrics['r']:.3f}"
    )


def _read_catalog(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader]


def _fetch_gutenberg_text(ebook_id_value: str, dest: Path) -> None:
    ebook_id = int(ebook_id_value)
    candidates = [
        f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt",
        f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}-0.txt",
        f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}.txt",
    ]
    text = None
    for url in candidates:
        try:
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                text = response.text
                break
        except requests.RequestException:
            continue
    if text is None:
        raise RuntimeError(f"Could not download Gutenberg book {ebook_id}.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")


def _fetch_http_text(url: str, dest: Path) -> None:
    if not url:
        raise ValueError("HTTP acquisition requires a URL.")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    text = response.text
    if "<html" in text.lower():
        text = _strip_html(text)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")


def _strip_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ")
    return normalize_text(text)


def _parse_lexile_value(raw_value: str | None) -> float | None:
    if raw_value is None:
        return None
    cleaned = raw_value.strip().upper().replace("L", "")
    if not cleaned:
        return None
    if cleaned.startswith("BR"):
        remainder = cleaned.removeprefix("BR")
        try:
            magnitude = float(remainder) if remainder else 0.0
        except ValueError:
            magnitude = 0.0
        return -magnitude
    try:
        return float(cleaned)
    except ValueError:
        return None
