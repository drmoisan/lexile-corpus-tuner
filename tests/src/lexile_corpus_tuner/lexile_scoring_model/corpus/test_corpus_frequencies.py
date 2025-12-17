"""Tests for corpus/frequencies.py module.

Comprehensive tests for the frequency computation functions following
unit-test-policy.md. All file system operations use tmp_path for isolation.
"""

from __future__ import annotations

import csv
import json
import logging
from typing import TYPE_CHECKING, Any

from lexile_corpus_tuner.lexile_scoring_model.corpus import frequencies

if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from pytest import MonkeyPatch


class TestCurrentVersion:
    """Tests for _current_version function."""

    def test_returns_iso_date_format(self) -> None:
        """Test that version is in ISO date format (YYYY-MM-DD)."""
        # Act
        version_fn = frequencies._current_version  # pyright: ignore[reportPrivateUsage]
        version = version_fn()

        # Assert
        assert len(version) == 10
        assert version[4] == "-"
        assert version[7] == "-"


class TestLoadSourceWeights:
    """Tests for _load_source_weights function."""

    def test_loads_weights_from_valid_json(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that source weights are loaded correctly from valid JSON."""
        # Arrange
        meta_path = tmp_path / "corpus_sources.json"
        meta_content = {
            "sources": [
                {"id": "gutenberg", "weight": 0.5},
                {"id": "simple_wiki", "weight": 0.3},
                {"id": "openstax", "weight": 0.2},
            ]
        }
        meta_path.write_text(json.dumps(meta_content))
        monkeypatch.setattr(frequencies, "CORPUS_META_PATH", meta_path)

        # Act
        load_fn = (
            frequencies._load_source_weights  # pyright: ignore[reportPrivateUsage]
        )
        weights = load_fn()

        # Assert
        assert weights == {"gutenberg": 0.5, "simple_wiki": 0.3, "openstax": 0.2}

    def test_returns_empty_dict_when_file_missing(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that empty dict is returned when file doesn't exist."""
        # Arrange
        meta_path = tmp_path / "nonexistent.json"
        monkeypatch.setattr(frequencies, "CORPUS_META_PATH", meta_path)

        # Act
        load_fn = (
            frequencies._load_source_weights  # pyright: ignore[reportPrivateUsage]
        )
        weights = load_fn()

        # Assert
        assert weights == {}

    def test_returns_empty_dict_on_invalid_json(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that empty dict is returned for invalid JSON."""
        # Arrange
        meta_path = tmp_path / "invalid.json"
        meta_path.write_text("not valid json {{{")
        monkeypatch.setattr(frequencies, "CORPUS_META_PATH", meta_path)

        # Act
        load_fn = (
            frequencies._load_source_weights  # pyright: ignore[reportPrivateUsage]
        )
        weights = load_fn()

        # Assert
        assert weights == {}

    def test_uses_default_weight_when_missing(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that default weight of 1.0 is used when not specified."""
        # Arrange
        meta_path = tmp_path / "corpus_sources.json"
        meta_content = {"sources": [{"id": "gutenberg"}]}  # No weight specified
        meta_path.write_text(json.dumps(meta_content))
        monkeypatch.setattr(frequencies, "CORPUS_META_PATH", meta_path)

        # Act
        load_fn = (
            frequencies._load_source_weights  # pyright: ignore[reportPrivateUsage]
        )
        weights = load_fn()

        # Assert
        assert weights == {"gutenberg": 1.0}

    def test_handles_invalid_weight_value(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that invalid weight values default to 1.0."""
        # Arrange
        meta_path = tmp_path / "corpus_sources.json"
        meta_content = {"sources": [{"id": "gutenberg", "weight": "not_a_number"}]}
        meta_path.write_text(json.dumps(meta_content))
        monkeypatch.setattr(frequencies, "CORPUS_META_PATH", meta_path)

        # Act
        load_fn = (
            frequencies._load_source_weights  # pyright: ignore[reportPrivateUsage]
        )
        weights = load_fn()

        # Assert
        assert weights == {"gutenberg": 1.0}

    def test_skips_entries_without_id(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that entries without id field are skipped."""
        # Arrange
        meta_path = tmp_path / "corpus_sources.json"
        meta_content: dict[str, list[dict[str, Any]]] = {
            "sources": [
                {"weight": 0.5},  # No id
                {"id": "valid", "weight": 0.3},
            ]
        }
        meta_path.write_text(json.dumps(meta_content))
        monkeypatch.setattr(frequencies, "CORPUS_META_PATH", meta_path)

        # Act
        load_fn = (
            frequencies._load_source_weights  # pyright: ignore[reportPrivateUsage]
        )
        weights = load_fn()

        # Assert
        assert weights == {"valid": 0.3}

    def test_returns_empty_dict_when_no_sources(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that empty dict is returned when sources list is empty."""
        # Arrange
        meta_path = tmp_path / "corpus_sources.json"
        meta_content: dict[str, list[Any]] = {"sources": []}
        meta_path.write_text(json.dumps(meta_content))
        monkeypatch.setattr(frequencies, "CORPUS_META_PATH", meta_path)

        # Act
        load_fn = (
            frequencies._load_source_weights  # pyright: ignore[reportPrivateUsage]
        )
        weights = load_fn()

        # Assert
        assert weights == {}

    def test_skips_non_string_ids(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that entries with non-string id are skipped."""
        # Arrange
        meta_path = tmp_path / "corpus_sources.json"
        meta_content = {
            "sources": [
                {"id": 123, "weight": 0.5},  # Non-string id
                {"id": "valid", "weight": 0.3},
            ]
        }
        meta_path.write_text(json.dumps(meta_content))
        monkeypatch.setattr(frequencies, "CORPUS_META_PATH", meta_path)

        # Act
        load_fn = (
            frequencies._load_source_weights  # pyright: ignore[reportPrivateUsage]
        )
        weights = load_fn()

        # Assert
        assert weights == {"valid": 0.3}


class TestComputeGlobalFrequencies:
    """Tests for compute_global_frequencies function."""

    def test_computes_frequencies_from_shards(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that frequencies are computed correctly from shard files."""
        # Arrange
        shards_root = tmp_path / "shards"
        shards_root.mkdir()

        # Create a shard file with token data
        shard_data = [
            {"source_id": "gutenberg", "tokens": ["the", "quick", "brown", "fox"]},
            {"source_id": "gutenberg", "tokens": ["the", "lazy", "dog"]},
        ]
        shard_file = shards_root / "shard-000001-gutenberg.jsonl"
        with shard_file.open("w") as f:
            for record in shard_data:
                f.write(json.dumps(record) + "\n")

        freq_root = tmp_path / "freq"
        freq_tsv = freq_root / "word_frequencies.tsv"
        freq_meta = freq_root / "meta.json"
        corpus_meta = tmp_path / "corpus_meta.json"

        monkeypatch.setattr(frequencies, "SHARDS_ROOT", shards_root)
        monkeypatch.setattr(frequencies, "FREQ_ROOT", freq_root)
        monkeypatch.setattr(frequencies, "FREQ_TSV", freq_tsv)
        monkeypatch.setattr(frequencies, "FREQ_META", freq_meta)
        monkeypatch.setattr(frequencies, "CORPUS_META_PATH", corpus_meta)

        # Act
        frequencies.compute_global_frequencies()

        # Assert
        assert freq_tsv.exists()
        assert freq_meta.exists()

    def test_writes_tsv_with_correct_columns(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that TSV file has correct column headers."""
        # Arrange
        shards_root = tmp_path / "shards"
        shards_root.mkdir()

        shard_data = [{"source_id": "test", "tokens": ["word"]}]
        shard_file = shards_root / "shard.jsonl"
        with shard_file.open("w") as f:
            f.write(json.dumps(shard_data[0]) + "\n")

        freq_root = tmp_path / "freq"
        freq_tsv = freq_root / "word_frequencies.tsv"
        freq_meta = freq_root / "meta.json"
        corpus_meta = tmp_path / "corpus_meta.json"

        monkeypatch.setattr(frequencies, "SHARDS_ROOT", shards_root)
        monkeypatch.setattr(frequencies, "FREQ_ROOT", freq_root)
        monkeypatch.setattr(frequencies, "FREQ_TSV", freq_tsv)
        monkeypatch.setattr(frequencies, "FREQ_META", freq_meta)
        monkeypatch.setattr(frequencies, "CORPUS_META_PATH", corpus_meta)

        # Act
        frequencies.compute_global_frequencies()

        # Assert
        with freq_tsv.open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            fieldnames = reader.fieldnames or []
            assert "token" in fieldnames
            assert "count" in fieldnames
            assert "freq_per_5m" in fieldnames
            assert "log_freq_per_5m" in fieldnames
            assert "rank" in fieldnames

    def test_tokens_sorted_by_frequency(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that tokens are sorted by frequency (highest first)."""
        # Arrange
        shards_root = tmp_path / "shards"
        shards_root.mkdir()

        shard_data: list[dict[str, str | list[str]]] = [
            {"source_id": "test", "tokens": ["rare"]},
            {"source_id": "test", "tokens": ["frequent", "frequent", "frequent"]},
            {"source_id": "test", "tokens": ["medium", "medium"]},
        ]
        shard_file = shards_root / "shard.jsonl"
        with shard_file.open("w") as f:
            for record in shard_data:
                f.write(json.dumps(record) + "\n")

        freq_root = tmp_path / "freq"
        freq_tsv = freq_root / "word_frequencies.tsv"
        freq_meta = freq_root / "meta.json"
        corpus_meta = tmp_path / "corpus_meta.json"

        monkeypatch.setattr(frequencies, "SHARDS_ROOT", shards_root)
        monkeypatch.setattr(frequencies, "FREQ_ROOT", freq_root)
        monkeypatch.setattr(frequencies, "FREQ_TSV", freq_tsv)
        monkeypatch.setattr(frequencies, "FREQ_META", freq_meta)
        monkeypatch.setattr(frequencies, "CORPUS_META_PATH", corpus_meta)

        # Act
        frequencies.compute_global_frequencies()

        # Assert
        with freq_tsv.open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)
            # Most frequent token should be first - "frequent" appears 3 times
            top_token = "frequent"  # noqa: S105
            assert rows[0]["token"] == top_token
            assert rows[0]["rank"] == "1"

    def test_applies_source_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that source weights are applied to frequency computation."""
        # Arrange
        shards_root = tmp_path / "shards"
        shards_root.mkdir()

        shard_data = [
            {"source_id": "high_weight", "tokens": ["word"]},
            {"source_id": "low_weight", "tokens": ["word"]},
        ]
        shard_file = shards_root / "shard.jsonl"
        with shard_file.open("w") as f:
            for record in shard_data:
                f.write(json.dumps(record) + "\n")

        freq_root = tmp_path / "freq"
        freq_tsv = freq_root / "word_frequencies.tsv"
        freq_meta = freq_root / "meta.json"

        corpus_meta = tmp_path / "corpus_meta.json"
        corpus_meta.write_text(
            json.dumps(
                {
                    "sources": [
                        {"id": "high_weight", "weight": 2.0},
                        {"id": "low_weight", "weight": 0.5},
                    ]
                }
            )
        )

        monkeypatch.setattr(frequencies, "SHARDS_ROOT", shards_root)
        monkeypatch.setattr(frequencies, "FREQ_ROOT", freq_root)
        monkeypatch.setattr(frequencies, "FREQ_TSV", freq_tsv)
        monkeypatch.setattr(frequencies, "FREQ_META", freq_meta)
        monkeypatch.setattr(frequencies, "CORPUS_META_PATH", corpus_meta)

        # Act
        frequencies.compute_global_frequencies()

        # Assert
        meta = json.loads(freq_meta.read_text())
        # Total weighted tokens should be 2*1 + 0.5*1 = 2.5
        assert meta["weighted_total_tokens"] == 2.5

    def test_writes_meta_file(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        """Test that metadata file is written correctly."""
        # Arrange
        shards_root = tmp_path / "shards"
        shards_root.mkdir()

        shard_data = [{"source_id": "test", "tokens": ["a", "b", "c"]}]
        shard_file = shards_root / "shard.jsonl"
        with shard_file.open("w") as f:
            f.write(json.dumps(shard_data[0]) + "\n")

        freq_root = tmp_path / "freq"
        freq_tsv = freq_root / "word_frequencies.tsv"
        freq_meta = freq_root / "meta.json"
        corpus_meta = tmp_path / "corpus_meta.json"

        monkeypatch.setattr(frequencies, "SHARDS_ROOT", shards_root)
        monkeypatch.setattr(frequencies, "FREQ_ROOT", freq_root)
        monkeypatch.setattr(frequencies, "FREQ_TSV", freq_tsv)
        monkeypatch.setattr(frequencies, "FREQ_META", freq_meta)
        monkeypatch.setattr(frequencies, "CORPUS_META_PATH", corpus_meta)

        # Act
        frequencies.compute_global_frequencies()

        # Assert
        meta = json.loads(freq_meta.read_text())
        assert "version" in meta
        assert "total_tokens" in meta
        assert "weighted_total_tokens" in meta
        assert "num_types" in meta
        assert meta["total_tokens"] == 3
        assert meta["num_types"] == 3

    def test_warns_when_no_shards(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that warning is logged when no shard files exist."""
        # Arrange

        shards_root = tmp_path / "shards"
        shards_root.mkdir()  # Empty directory

        freq_root = tmp_path / "freq"
        freq_tsv = freq_root / "word_frequencies.tsv"
        freq_meta = freq_root / "meta.json"
        corpus_meta = tmp_path / "corpus_meta.json"

        monkeypatch.setattr(frequencies, "SHARDS_ROOT", shards_root)
        monkeypatch.setattr(frequencies, "FREQ_ROOT", freq_root)
        monkeypatch.setattr(frequencies, "FREQ_TSV", freq_tsv)
        monkeypatch.setattr(frequencies, "FREQ_META", freq_meta)
        monkeypatch.setattr(frequencies, "CORPUS_META_PATH", corpus_meta)

        # Act
        with caplog.at_level(logging.WARNING):
            frequencies.compute_global_frequencies()

        # Assert
        assert "No shard files found" in caplog.text

    def test_warns_when_zero_tokens(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that warning is logged when shards contain zero tokens."""
        # Arrange

        shards_root = tmp_path / "shards"
        shards_root.mkdir()

        shard_data: list[dict[str, str | list[str]]] = [
            {"source_id": "test", "tokens": []}
        ]  # Empty tokens
        shard_file = shards_root / "shard.jsonl"
        with shard_file.open("w") as f:
            f.write(json.dumps(shard_data[0]) + "\n")

        freq_root = tmp_path / "freq"
        freq_tsv = freq_root / "word_frequencies.tsv"
        freq_meta = freq_root / "meta.json"
        corpus_meta = tmp_path / "corpus_meta.json"

        monkeypatch.setattr(frequencies, "SHARDS_ROOT", shards_root)
        monkeypatch.setattr(frequencies, "FREQ_ROOT", freq_root)
        monkeypatch.setattr(frequencies, "FREQ_TSV", freq_tsv)
        monkeypatch.setattr(frequencies, "FREQ_META", freq_meta)
        monkeypatch.setattr(frequencies, "CORPUS_META_PATH", corpus_meta)

        # Act
        with caplog.at_level(logging.WARNING):
            frequencies.compute_global_frequencies()

        # Assert
        assert "zero tokens" in caplog.text

    def test_skips_empty_lines_in_shards(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that empty lines in shard files are skipped."""
        # Arrange
        shards_root = tmp_path / "shards"
        shards_root.mkdir()

        shard_file = shards_root / "shard.jsonl"
        with shard_file.open("w") as f:
            f.write('{"source_id": "test", "tokens": ["word"]}\n')
            f.write("\n")  # Empty line
            f.write("   \n")  # Whitespace-only line

        freq_root = tmp_path / "freq"
        freq_tsv = freq_root / "word_frequencies.tsv"
        freq_meta = freq_root / "meta.json"
        corpus_meta = tmp_path / "corpus_meta.json"

        monkeypatch.setattr(frequencies, "SHARDS_ROOT", shards_root)
        monkeypatch.setattr(frequencies, "FREQ_ROOT", freq_root)
        monkeypatch.setattr(frequencies, "FREQ_TSV", freq_tsv)
        monkeypatch.setattr(frequencies, "FREQ_META", freq_meta)
        monkeypatch.setattr(frequencies, "CORPUS_META_PATH", corpus_meta)

        # Act
        frequencies.compute_global_frequencies()

        # Assert - should succeed without error
        assert freq_tsv.exists()

    def test_skips_invalid_json_lines(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that invalid JSON lines in shard files are skipped."""
        # Arrange
        shards_root = tmp_path / "shards"
        shards_root.mkdir()

        shard_file = shards_root / "shard.jsonl"
        with shard_file.open("w") as f:
            f.write('{"source_id": "test", "tokens": ["valid"]}\n')
            f.write("invalid json\n")
            f.write('{"source_id": "test", "tokens": ["also_valid"]}\n')

        freq_root = tmp_path / "freq"
        freq_tsv = freq_root / "word_frequencies.tsv"
        freq_meta = freq_root / "meta.json"
        corpus_meta = tmp_path / "corpus_meta.json"

        monkeypatch.setattr(frequencies, "SHARDS_ROOT", shards_root)
        monkeypatch.setattr(frequencies, "FREQ_ROOT", freq_root)
        monkeypatch.setattr(frequencies, "FREQ_TSV", freq_tsv)
        monkeypatch.setattr(frequencies, "FREQ_META", freq_meta)
        monkeypatch.setattr(frequencies, "CORPUS_META_PATH", corpus_meta)

        # Act
        frequencies.compute_global_frequencies()

        # Assert
        meta = json.loads(freq_meta.read_text())
        assert meta["total_tokens"] == 2  # Only valid tokens counted

    def test_handles_missing_tokens_field(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that records without tokens field are handled."""
        # Arrange
        shards_root = tmp_path / "shards"
        shards_root.mkdir()

        shard_file = shards_root / "shard.jsonl"
        with shard_file.open("w") as f:
            f.write('{"source_id": "test"}\n')  # No tokens
            f.write('{"source_id": "test", "tokens": ["word"]}\n')

        freq_root = tmp_path / "freq"
        freq_tsv = freq_root / "word_frequencies.tsv"
        freq_meta = freq_root / "meta.json"
        corpus_meta = tmp_path / "corpus_meta.json"

        monkeypatch.setattr(frequencies, "SHARDS_ROOT", shards_root)
        monkeypatch.setattr(frequencies, "FREQ_ROOT", freq_root)
        monkeypatch.setattr(frequencies, "FREQ_TSV", freq_tsv)
        monkeypatch.setattr(frequencies, "FREQ_META", freq_meta)
        monkeypatch.setattr(frequencies, "CORPUS_META_PATH", corpus_meta)

        # Act
        frequencies.compute_global_frequencies()

        # Assert
        meta = json.loads(freq_meta.read_text())
        assert meta["total_tokens"] == 1

    def test_uses_default_weight_for_unknown_source(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that default weight 1.0 is used for unknown sources."""
        # Arrange
        shards_root = tmp_path / "shards"
        shards_root.mkdir()

        shard_data = [{"source_id": "unknown_source", "tokens": ["word"]}]
        shard_file = shards_root / "shard.jsonl"
        with shard_file.open("w") as f:
            f.write(json.dumps(shard_data[0]) + "\n")

        freq_root = tmp_path / "freq"
        freq_tsv = freq_root / "word_frequencies.tsv"
        freq_meta = freq_root / "meta.json"
        corpus_meta = tmp_path / "corpus_meta.json"
        corpus_meta_content: dict[str, list[Any]] = {"sources": []}
        corpus_meta.write_text(json.dumps(corpus_meta_content))

        monkeypatch.setattr(frequencies, "SHARDS_ROOT", shards_root)
        monkeypatch.setattr(frequencies, "FREQ_ROOT", freq_root)
        monkeypatch.setattr(frequencies, "FREQ_TSV", freq_tsv)
        monkeypatch.setattr(frequencies, "FREQ_META", freq_meta)
        monkeypatch.setattr(frequencies, "CORPUS_META_PATH", corpus_meta)

        # Act
        frequencies.compute_global_frequencies()

        # Assert
        meta = json.loads(freq_meta.read_text())
        # With default weight 1.0, weighted = unweighted
        assert meta["weighted_total_tokens"] == meta["total_tokens"]

    def test_creates_freq_directory(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that frequency directory is created if it doesn't exist."""
        # Arrange
        shards_root = tmp_path / "shards"
        shards_root.mkdir()

        shard_data = [{"source_id": "test", "tokens": ["word"]}]
        shard_file = shards_root / "shard.jsonl"
        with shard_file.open("w") as f:
            f.write(json.dumps(shard_data[0]) + "\n")

        freq_root = tmp_path / "freq"  # Does not exist yet
        freq_tsv = freq_root / "word_frequencies.tsv"
        freq_meta = freq_root / "meta.json"
        corpus_meta = tmp_path / "corpus_meta.json"

        monkeypatch.setattr(frequencies, "SHARDS_ROOT", shards_root)
        monkeypatch.setattr(frequencies, "FREQ_ROOT", freq_root)
        monkeypatch.setattr(frequencies, "FREQ_TSV", freq_tsv)
        monkeypatch.setattr(frequencies, "FREQ_META", freq_meta)
        monkeypatch.setattr(frequencies, "CORPUS_META_PATH", corpus_meta)

        # Act
        frequencies.compute_global_frequencies()

        # Assert
        assert freq_root.exists()

    def test_computes_log_frequency(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that log frequency is computed correctly."""
        # Arrange
        shards_root = tmp_path / "shards"
        shards_root.mkdir()

        shard_data = [{"source_id": "test", "tokens": ["word"]}]
        shard_file = shards_root / "shard.jsonl"
        with shard_file.open("w") as f:
            f.write(json.dumps(shard_data[0]) + "\n")

        freq_root = tmp_path / "freq"
        freq_tsv = freq_root / "word_frequencies.tsv"
        freq_meta = freq_root / "meta.json"
        corpus_meta = tmp_path / "corpus_meta.json"

        monkeypatch.setattr(frequencies, "SHARDS_ROOT", shards_root)
        monkeypatch.setattr(frequencies, "FREQ_ROOT", freq_root)
        monkeypatch.setattr(frequencies, "FREQ_TSV", freq_tsv)
        monkeypatch.setattr(frequencies, "FREQ_META", freq_meta)
        monkeypatch.setattr(frequencies, "CORPUS_META_PATH", corpus_meta)

        # Act
        frequencies.compute_global_frequencies()

        # Assert
        with freq_tsv.open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            row = next(reader)
            log_freq = float(row["log_freq_per_5m"])
            # Log frequency should be a reasonable number
            assert log_freq > 0

    def test_handles_multiple_shard_files(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that multiple shard files are processed."""
        # Arrange
        shards_root = tmp_path / "shards"
        shards_root.mkdir()

        shard1 = shards_root / "shard-001.jsonl"
        shard2 = shards_root / "shard-002.jsonl"

        with shard1.open("w") as f:
            f.write('{"source_id": "test", "tokens": ["word1", "word2"]}\n')
        with shard2.open("w") as f:
            f.write('{"source_id": "test", "tokens": ["word3", "word4"]}\n')

        freq_root = tmp_path / "freq"
        freq_tsv = freq_root / "word_frequencies.tsv"
        freq_meta = freq_root / "meta.json"
        corpus_meta = tmp_path / "corpus_meta.json"

        monkeypatch.setattr(frequencies, "SHARDS_ROOT", shards_root)
        monkeypatch.setattr(frequencies, "FREQ_ROOT", freq_root)
        monkeypatch.setattr(frequencies, "FREQ_TSV", freq_tsv)
        monkeypatch.setattr(frequencies, "FREQ_META", freq_meta)
        monkeypatch.setattr(frequencies, "CORPUS_META_PATH", corpus_meta)

        # Act
        frequencies.compute_global_frequencies()

        # Assert
        meta = json.loads(freq_meta.read_text())
        assert meta["total_tokens"] == 4
        assert meta["num_types"] == 4

    def test_logs_completion_info(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that completion info is logged."""
        # Arrange

        shards_root = tmp_path / "shards"
        shards_root.mkdir()

        shard_data = [{"source_id": "test", "tokens": ["word"]}]
        shard_file = shards_root / "shard.jsonl"
        with shard_file.open("w") as f:
            f.write(json.dumps(shard_data[0]) + "\n")

        freq_root = tmp_path / "freq"
        freq_tsv = freq_root / "word_frequencies.tsv"
        freq_meta = freq_root / "meta.json"
        corpus_meta = tmp_path / "corpus_meta.json"

        monkeypatch.setattr(frequencies, "SHARDS_ROOT", shards_root)
        monkeypatch.setattr(frequencies, "FREQ_ROOT", freq_root)
        monkeypatch.setattr(frequencies, "FREQ_TSV", freq_tsv)
        monkeypatch.setattr(frequencies, "FREQ_META", freq_meta)
        monkeypatch.setattr(frequencies, "CORPUS_META_PATH", corpus_meta)

        # Act
        with caplog.at_level(logging.INFO):
            frequencies.compute_global_frequencies()

        # Assert
        assert "Computed global frequencies" in caplog.text

    def test_handles_missing_source_id_in_record(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that records without source_id field are handled."""
        # Arrange
        shards_root = tmp_path / "shards"
        shards_root.mkdir()

        shard_file = shards_root / "shard.jsonl"
        with shard_file.open("w") as f:
            f.write('{"tokens": ["word"]}\n')  # No source_id

        freq_root = tmp_path / "freq"
        freq_tsv = freq_root / "word_frequencies.tsv"
        freq_meta = freq_root / "meta.json"
        corpus_meta = tmp_path / "corpus_meta.json"

        monkeypatch.setattr(frequencies, "SHARDS_ROOT", shards_root)
        monkeypatch.setattr(frequencies, "FREQ_ROOT", freq_root)
        monkeypatch.setattr(frequencies, "FREQ_TSV", freq_tsv)
        monkeypatch.setattr(frequencies, "FREQ_META", freq_meta)
        monkeypatch.setattr(frequencies, "CORPUS_META_PATH", corpus_meta)

        # Act
        frequencies.compute_global_frequencies()

        # Assert - should succeed using "unknown" as source_id
        assert freq_tsv.exists()
