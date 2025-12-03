# pyright: reportUnknownMemberType=false, reportPrivateUsage=false
"""Tests for analyzer features module.

This module tests the compute_document_features function and related
helper functions for computing Lexile-style features from text slices.
"""

from unittest.mock import patch

import pytest
from lexile_corpus_tuner.analyzer.features import (
    DocumentFeatures,
    SliceFeatures,
    _compute_unseen_floor,
    compute_document_features,
)
from lexile_corpus_tuner.analyzer.slices import Slice
from lexile_corpus_tuner.frequency_loader import WordFrequency


@pytest.fixture
def sample_frequency_table() -> dict[str, WordFrequency]:
    """Create a sample frequency table for testing."""
    return {
        "the": WordFrequency(
            count=100000, freq_per_5m=50000.0, log_freq_per_5m=4.7, rank=1
        ),
        "a": WordFrequency(
            count=50000, freq_per_5m=25000.0, log_freq_per_5m=4.4, rank=2
        ),
        "test": WordFrequency(
            count=1000, freq_per_5m=500.0, log_freq_per_5m=2.7, rank=100
        ),
        "word": WordFrequency(
            count=500, freq_per_5m=250.0, log_freq_per_5m=2.4, rank=200
        ),
        "hello": WordFrequency(
            count=200, freq_per_5m=100.0, log_freq_per_5m=2.0, rank=500
        ),
        "world": WordFrequency(
            count=150, freq_per_5m=75.0, log_freq_per_5m=1.9, rank=600
        ),
    }


@pytest.fixture
def simple_slice() -> Slice:
    """Create a simple slice for testing."""
    return Slice(
        slice_id=0,
        text="The test word.",
        tokens=["the", "test", "word"],
        sentence_lengths=[3],
    )


@pytest.fixture
def multi_sentence_slice() -> Slice:
    """Create a slice with multiple sentences."""
    return Slice(
        slice_id=0,
        text="Hello world. The test.",
        tokens=["hello", "world", "the", "test"],
        sentence_lengths=[2, 2],
    )


class TestComputeDocumentFeatures:
    """Tests for compute_document_features function."""

    def test_single_slice_basic_features(
        self, sample_frequency_table: dict[str, WordFrequency], simple_slice: Slice
    ):
        """Test computing features for a single slice with known tokens."""
        with patch(
            "lexile_corpus_tuner.analyzer.features.load_frequency_table",
            return_value=sample_frequency_table,
        ):
            result = compute_document_features([simple_slice])

        assert isinstance(result, DocumentFeatures)
        assert result.num_slices == 1
        assert result.total_tokens == 3
        assert len(result.slice_features) == 1

    def test_slice_features_populated(
        self, sample_frequency_table: dict[str, WordFrequency], simple_slice: Slice
    ):
        """Test that slice features are correctly populated."""
        with patch(
            "lexile_corpus_tuner.analyzer.features.load_frequency_table",
            return_value=sample_frequency_table,
        ):
            result = compute_document_features([simple_slice])

        sf = result.slice_features[0]
        assert isinstance(sf, SliceFeatures)
        assert sf.slice_id == 0
        assert sf.num_tokens == 3
        assert sf.num_sentences == 1
        assert sf.mean_sentence_length == 3.0

    def test_mean_log_word_freq_calculated(
        self, sample_frequency_table: dict[str, WordFrequency], simple_slice: Slice
    ):
        """Test that mean log word frequency is correctly calculated."""
        with patch(
            "lexile_corpus_tuner.analyzer.features.load_frequency_table",
            return_value=sample_frequency_table,
        ):
            result = compute_document_features([simple_slice])

        # Expected: mean of [4.7, 2.7, 2.4] = 9.8 / 3 ≈ 3.267
        sf = result.slice_features[0]
        expected_mlf = (4.7 + 2.7 + 2.4) / 3
        assert sf.mean_log_word_freq == pytest.approx(expected_mlf, rel=0.01)

    def test_multiple_slices(self, sample_frequency_table: dict[str, WordFrequency]):
        """Test computing features for multiple slices."""
        slice1 = Slice(
            slice_id=0,
            text="The test.",
            tokens=["the", "test"],
            sentence_lengths=[2],
        )
        slice2 = Slice(
            slice_id=1,
            text="Hello world.",
            tokens=["hello", "world"],
            sentence_lengths=[2],
        )

        with patch(
            "lexile_corpus_tuner.analyzer.features.load_frequency_table",
            return_value=sample_frequency_table,
        ):
            result = compute_document_features([slice1, slice2])

        assert result.num_slices == 2
        assert result.total_tokens == 4
        assert len(result.slice_features) == 2
        assert result.slice_features[0].slice_id == 0
        assert result.slice_features[1].slice_id == 1

    def test_overall_mean_sentence_length(
        self, sample_frequency_table: dict[str, WordFrequency]
    ):
        """Test that overall mean sentence length is calculated correctly."""
        slice1 = Slice(
            slice_id=0,
            text="One two three. Four five.",
            tokens=["one", "two", "three", "four", "five"],
            sentence_lengths=[3, 2],  # Avg = 2.5
        )
        slice2 = Slice(
            slice_id=1,
            text="Six seven eight nine.",
            tokens=["six", "seven", "eight", "nine"],
            sentence_lengths=[4],  # Avg = 4.0
        )

        with patch(
            "lexile_corpus_tuner.analyzer.features.load_frequency_table",
            return_value=sample_frequency_table,
        ):
            result = compute_document_features([slice1, slice2])

        # Overall mean: (3 + 2 + 4) / 3 = 3.0
        assert result.overall_mean_sentence_length == pytest.approx(3.0)

    def test_empty_slices_list(self, sample_frequency_table: dict[str, WordFrequency]):
        """Test computing features with empty slices list."""
        with patch(
            "lexile_corpus_tuner.analyzer.features.load_frequency_table",
            return_value=sample_frequency_table,
        ):
            result = compute_document_features([])

        assert result.num_slices == 0
        assert result.total_tokens == 0
        assert result.slice_features == []
        assert result.overall_mean_sentence_length == 0.0

    def test_slice_with_no_sentence_lengths(
        self, sample_frequency_table: dict[str, WordFrequency]
    ):
        """Test handling of slice with empty sentence_lengths."""
        sl = Slice(
            slice_id=0,
            text="No sentences",
            tokens=["no", "sentences"],
            sentence_lengths=[],  # Empty sentence lengths
        )

        with patch(
            "lexile_corpus_tuner.analyzer.features.load_frequency_table",
            return_value=sample_frequency_table,
        ):
            result = compute_document_features([sl])

        # When no sentence_lengths, mean_sentence_length should default to num_tokens
        sf = result.slice_features[0]
        assert sf.mean_sentence_length == 2.0  # num_tokens when no sentences
        assert sf.num_sentences == 1  # Defaults to 1

    def test_slice_with_empty_tokens(
        self, sample_frequency_table: dict[str, WordFrequency]
    ):
        """Test handling of slice with empty tokens."""
        sl = Slice(
            slice_id=0,
            text="",
            tokens=[],
            sentence_lengths=[],
        )

        with patch(
            "lexile_corpus_tuner.analyzer.features.load_frequency_table",
            return_value=sample_frequency_table,
        ):
            result = compute_document_features([sl])

        sf = result.slice_features[0]
        assert sf.num_tokens == 0
        assert sf.mean_sentence_length == 0.0

    def test_unknown_words_use_unseen_floor(
        self, sample_frequency_table: dict[str, WordFrequency]
    ):
        """Test that unknown words use the unseen floor frequency."""
        sl = Slice(
            slice_id=0,
            text="Unknown words here",
            tokens=["unknownword", "anotherunk"],  # Not in frequency table
            sentence_lengths=[2],
        )

        with patch(
            "lexile_corpus_tuner.analyzer.features.load_frequency_table",
            return_value=sample_frequency_table,
        ):
            result = compute_document_features([sl])

        # Unseen floor should be min_log_freq - 1.0
        # Min log freq in table is 1.9, so unseen floor is 0.9
        expected_unseen = 1.9 - 1.0
        sf = result.slice_features[0]
        assert sf.mean_log_word_freq == pytest.approx(expected_unseen, rel=0.01)

    def test_mixed_known_unknown_tokens(
        self, sample_frequency_table: dict[str, WordFrequency]
    ):
        """Test handling of mix of known and unknown tokens."""
        sl = Slice(
            slice_id=0,
            text="The unknownword test",
            tokens=["the", "unknownword", "test"],
            sentence_lengths=[3],
        )

        with patch(
            "lexile_corpus_tuner.analyzer.features.load_frequency_table",
            return_value=sample_frequency_table,
        ):
            result = compute_document_features([sl])

        # Expected: mean of [4.7, 0.9, 2.7] = 8.3 / 3 ≈ 2.767
        unseen_floor = 1.9 - 1.0  # 0.9
        expected_mlf = (4.7 + unseen_floor + 2.7) / 3
        sf = result.slice_features[0]
        assert sf.mean_log_word_freq == pytest.approx(expected_mlf, rel=0.01)

    def test_overall_mean_log_word_freq(
        self, sample_frequency_table: dict[str, WordFrequency]
    ):
        """Test that overall mean log word frequency is calculated correctly."""
        slice1 = Slice(
            slice_id=0,
            text="The test.",
            tokens=["the", "test"],  # log freqs: 4.7, 2.7
            sentence_lengths=[2],
        )
        slice2 = Slice(
            slice_id=1,
            text="Hello world.",
            tokens=["hello", "world"],  # log freqs: 2.0, 1.9
            sentence_lengths=[2],
        )

        with patch(
            "lexile_corpus_tuner.analyzer.features.load_frequency_table",
            return_value=sample_frequency_table,
        ):
            result = compute_document_features([slice1, slice2])

        # Overall mean: (4.7 + 2.7 + 2.0 + 1.9) / 4 = 11.3 / 4 = 2.825
        expected = (4.7 + 2.7 + 2.0 + 1.9) / 4
        assert result.overall_mean_log_word_freq == pytest.approx(expected, rel=0.01)


class TestComputeUnseenFloor:
    """Tests for _compute_unseen_floor function."""

    def test_with_valid_frequency_table(
        self, sample_frequency_table: dict[str, WordFrequency]
    ):
        """Test unseen floor calculation with valid table."""
        result = _compute_unseen_floor(sample_frequency_table)
        # Min log freq is 1.9 (world), so floor is 0.9
        assert result == pytest.approx(0.9, rel=0.01)

    def test_with_empty_frequency_table(self):
        """Test unseen floor calculation with empty table returns default."""
        result = _compute_unseen_floor({})
        assert result == -20.0

    def test_with_single_entry_table(self):
        """Test unseen floor calculation with single entry table."""
        table = {
            "only": WordFrequency(
                count=100, freq_per_5m=50.0, log_freq_per_5m=1.7, rank=1
            )
        }
        result = _compute_unseen_floor(table)
        assert result == pytest.approx(0.7, rel=0.01)

    def test_floor_is_below_minimum(self):
        """Test that unseen floor is always below the minimum frequency."""
        table = {
            "high": WordFrequency(
                count=1000, freq_per_5m=500.0, log_freq_per_5m=5.0, rank=1
            ),
            "low": WordFrequency(
                count=10, freq_per_5m=5.0, log_freq_per_5m=0.5, rank=100
            ),
        }
        result = _compute_unseen_floor(table)
        # Should be 0.5 - 1.0 = -0.5
        assert result == pytest.approx(-0.5, rel=0.01)


class TestSliceFeaturesDataclass:
    """Tests for SliceFeatures dataclass."""

    def test_slice_features_fields(self):
        """Test that SliceFeatures has all required fields."""
        sf = SliceFeatures(
            slice_id=0,
            num_tokens=10,
            num_sentences=2,
            mean_sentence_length=5.0,
            mean_log_word_freq=3.5,
        )
        assert sf.slice_id == 0
        assert sf.num_tokens == 10
        assert sf.num_sentences == 2
        assert sf.mean_sentence_length == 5.0
        assert sf.mean_log_word_freq == 3.5


class TestDocumentFeaturesDataclass:
    """Tests for DocumentFeatures dataclass."""

    def test_document_features_fields(self):
        """Test that DocumentFeatures has all required fields."""
        sf = SliceFeatures(
            slice_id=0,
            num_tokens=10,
            num_sentences=2,
            mean_sentence_length=5.0,
            mean_log_word_freq=3.5,
        )
        df = DocumentFeatures(
            num_slices=1,
            total_tokens=10,
            overall_mean_sentence_length=5.0,
            overall_mean_log_word_freq=3.5,
            slice_features=[sf],
        )
        assert df.num_slices == 1
        assert df.total_tokens == 10
        assert df.overall_mean_sentence_length == 5.0
        assert df.overall_mean_log_word_freq == 3.5
        assert len(df.slice_features) == 1
