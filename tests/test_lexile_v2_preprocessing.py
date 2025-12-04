"""Tests for lexile_v2_preprocessing module.

This module tests the preprocessing pipeline for the Lexile v2 model adapter.
All NLTK dependencies are mocked to ensure tests run without downloads.

Note: This file tests some private functions (prefixed with _) because they
contain significant logic that benefits from direct testing.
"""

import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# Import module for access to private members through helpers
import lexile_corpus_tuner.estimators.lexile_v2_preprocessing as preproc_module
import pytest


def _mock_lower(word: Any) -> str:
    """Mock lemmatizer that lowercases input."""
    return str(word).lower()


def _mock_identity(word: Any) -> str:
    """Mock lemmatizer that returns string as-is."""
    return str(word)


def _get_nltk_cache() -> Any:
    """Access protected cache for testing only."""
    return preproc_module._nltk_cache  # pyright: ignore[reportPrivateUsage]


def _set_nltk_cache(value: Any) -> None:
    """Set protected cache for testing only."""
    preproc_module._nltk_cache = value  # pyright: ignore[reportPrivateUsage]


def _ensure_nltk_dependencies() -> (
    tuple[type[Any], type[Any], Callable[..., list[str]], Callable[..., list[str]]]
):
    """Access protected function for testing only."""
    # Accessing protected function for test purposes
    fn = preproc_module._ensure_nltk_dependencies  # pyright: ignore[reportPrivateUsage]
    return fn()


class TestLoadStopwords:
    """Tests for load_stopwords function."""

    def test_load_stopwords_from_list(self, tmp_path: Path) -> None:
        """Test loading stopwords from a pickled list."""
        # Arrange
        stopwords_file = tmp_path / "stopwords.pkl"
        stopwords_data = ["the", "a", "an", "and", "or"]
        with stopwords_file.open("wb") as f:
            pickle.dump(stopwords_data, f)

        # Act
        from lexile_corpus_tuner.estimators.lexile_v2_preprocessing import (
            load_stopwords,
        )

        result = load_stopwords(stopwords_file)

        # Assert
        assert result == ["the", "a", "an", "and", "or"]

    def test_load_stopwords_from_tuple(self, tmp_path: Path) -> None:
        """Test loading stopwords from a pickled tuple."""
        # Arrange
        stopwords_file = tmp_path / "stopwords.pkl"
        stopwords_data = ("the", "a", "an")
        with stopwords_file.open("wb") as f:
            pickle.dump(stopwords_data, f)

        # Act
        from lexile_corpus_tuner.estimators.lexile_v2_preprocessing import (
            load_stopwords,
        )

        result = load_stopwords(stopwords_file)

        # Assert
        assert result == ["the", "a", "an"]

    def test_load_stopwords_from_set(self, tmp_path: Path) -> None:
        """Test loading stopwords from a pickled set."""
        # Arrange
        stopwords_file = tmp_path / "stopwords.pkl"
        stopwords_data = {"the", "a", "an"}
        with stopwords_file.open("wb") as f:
            pickle.dump(stopwords_data, f)

        # Act
        from lexile_corpus_tuner.estimators.lexile_v2_preprocessing import (
            load_stopwords,
        )

        result = load_stopwords(stopwords_file)

        # Assert
        assert set(result) == {"the", "a", "an"}

    def test_load_stopwords_from_dict(self, tmp_path: Path) -> None:
        """Test loading stopwords from a pickled dict (uses keys)."""
        # Arrange
        stopwords_file = tmp_path / "stopwords.pkl"
        stopwords_data = {"the": 1, "a": 2, "an": 3}
        with stopwords_file.open("wb") as f:
            pickle.dump(stopwords_data, f)

        # Act
        from lexile_corpus_tuner.estimators.lexile_v2_preprocessing import (
            load_stopwords,
        )

        result = load_stopwords(stopwords_file)

        # Assert
        assert set(result) == {"the", "a", "an"}

    def test_load_stopwords_from_string(self, tmp_path: Path) -> None:
        """Test loading stopwords from a pickled string (splits by lines)."""
        # Arrange
        stopwords_file = tmp_path / "stopwords.pkl"
        stopwords_data = "the\na\nan\nand"
        with stopwords_file.open("wb") as f:
            pickle.dump(stopwords_data, f)

        # Act
        from lexile_corpus_tuner.estimators.lexile_v2_preprocessing import (
            load_stopwords,
        )

        result = load_stopwords(stopwords_file)

        # Assert
        assert result == ["the", "a", "an", "and"]

    def test_load_stopwords_unsupported_type_raises_error(self, tmp_path: Path) -> None:
        """Test that unsupported payload type raises TypeError."""
        # Arrange
        stopwords_file = tmp_path / "stopwords.pkl"
        stopwords_data = 12345  # integer is not supported
        with stopwords_file.open("wb") as f:
            pickle.dump(stopwords_data, f)

        # Act & Assert
        from lexile_corpus_tuner.estimators.lexile_v2_preprocessing import (
            load_stopwords,
        )

        with pytest.raises(TypeError, match="Unsupported stopword payload type"):
            load_stopwords(stopwords_file)

    def test_load_stopwords_missing_file_raises_error(self, tmp_path: Path) -> None:
        """Test that missing file raises FileNotFoundError."""
        # Arrange
        nonexistent_file = tmp_path / "nonexistent.pkl"

        # Act & Assert
        from lexile_corpus_tuner.estimators.lexile_v2_preprocessing import (
            load_stopwords,
        )

        with pytest.raises(FileNotFoundError):
            load_stopwords(nonexistent_file)

    def test_load_stopwords_converts_items_to_strings(self, tmp_path: Path) -> None:
        """Test that items in list are converted to strings."""
        # Arrange
        stopwords_file = tmp_path / "stopwords.pkl"
        stopwords_data = [1, 2, 3]  # integers in list
        with stopwords_file.open("wb") as f:
            pickle.dump(stopwords_data, f)

        # Act
        from lexile_corpus_tuner.estimators.lexile_v2_preprocessing import (
            load_stopwords,
        )

        result = load_stopwords(stopwords_file)

        # Assert
        assert result == ["1", "2", "3"]

    def test_load_stopwords_accepts_string_path(self, tmp_path: Path) -> None:
        """Test that load_stopwords accepts a string path."""
        # Arrange
        stopwords_file = tmp_path / "stopwords.pkl"
        stopwords_data = ["the", "a"]
        with stopwords_file.open("wb") as f:
            pickle.dump(stopwords_data, f)

        # Act
        from lexile_corpus_tuner.estimators.lexile_v2_preprocessing import (
            load_stopwords,
        )

        result = load_stopwords(str(stopwords_file))

        # Assert
        assert result == ["the", "a"]


class TestLoadPickle:
    """Tests for _load_pickle function."""

    def test_load_pickle_returns_object(self, tmp_path: Path) -> None:
        """Test that _load_pickle returns the pickled object."""
        # Arrange
        pickle_file = tmp_path / "data.pkl"
        data = {"key": "value", "number": 42}
        with pickle_file.open("wb") as f:
            pickle.dump(data, f)

        # Act
        from lexile_corpus_tuner.estimators.lexile_v2_preprocessing import (
            _load_pickle,  # pyright: ignore[reportPrivateUsage]
        )

        result = _load_pickle(pickle_file)

        # Assert
        assert result == {"key": "value", "number": 42}

    def test_load_pickle_missing_file_raises_error(self, tmp_path: Path) -> None:
        """Test that missing file raises FileNotFoundError."""
        # Arrange
        nonexistent_file = tmp_path / "nonexistent.pkl"

        # Act & Assert
        from lexile_corpus_tuner.estimators.lexile_v2_preprocessing import (
            _load_pickle,  # pyright: ignore[reportPrivateUsage]
        )

        with pytest.raises(FileNotFoundError):
            _load_pickle(nonexistent_file)

    def test_load_pickle_corrupted_file_raises_error(self, tmp_path: Path) -> None:
        """Test that corrupted pickle file raises error."""
        # Arrange
        corrupted_file = tmp_path / "corrupted.pkl"
        corrupted_file.write_bytes(b"not a valid pickle")

        # Act & Assert
        from lexile_corpus_tuner.estimators.lexile_v2_preprocessing import (
            _load_pickle,  # pyright: ignore[reportPrivateUsage]
        )

        with pytest.raises(pickle.UnpicklingError):
            _load_pickle(corrupted_file)

    def test_load_pickle_accepts_string_path(self, tmp_path: Path) -> None:
        """Test that _load_pickle accepts a string path."""
        # Arrange
        pickle_file = tmp_path / "data.pkl"
        data = [1, 2, 3]
        with pickle_file.open("wb") as f:
            pickle.dump(data, f)

        # Act
        from lexile_corpus_tuner.estimators.lexile_v2_preprocessing import (
            _load_pickle,  # pyright: ignore[reportPrivateUsage]
        )

        result = _load_pickle(str(pickle_file))

        # Assert
        assert result == [1, 2, 3]


class TestEnsureNltkDependencies:
    """Tests for _ensure_nltk_dependencies function."""

    def test_ensure_nltk_dependencies_returns_tuple(self) -> None:
        """Test that _ensure_nltk_dependencies returns expected tuple."""
        # Arrange
        mock_stem_module = MagicMock()
        mock_tokenize_module = MagicMock()
        mock_lemmatizer = MagicMock()
        mock_tiling = MagicMock()
        mock_sent_tok = MagicMock()
        mock_word_tok = MagicMock()

        mock_stem_module.WordNetLemmatizer = mock_lemmatizer
        mock_tokenize_module.TextTilingTokenizer = mock_tiling
        mock_tokenize_module.sent_tokenize = mock_sent_tok
        mock_tokenize_module.word_tokenize = mock_word_tok

        # Act
        with patch(
            "lexile_corpus_tuner.estimators.lexile_v2_preprocessing.import_module"
        ) as mock_import:
            mock_import.side_effect = [mock_stem_module, mock_tokenize_module]

            # Reset the cache before testing
            original_cache = _get_nltk_cache()
            _set_nltk_cache(None)

            try:
                result = _ensure_nltk_dependencies()

                # Assert
                assert len(result) == 4
                assert result[0] == mock_lemmatizer
                assert result[1] == mock_tiling
                assert result[2] == mock_sent_tok
                assert result[3] == mock_word_tok
            finally:
                _set_nltk_cache(original_cache)

    def test_ensure_nltk_dependencies_caches_result(self) -> None:
        """Test that _ensure_nltk_dependencies caches the result."""
        # Arrange
        mock_stem_module = MagicMock()
        mock_tokenize_module = MagicMock()
        mock_lemmatizer = MagicMock()
        mock_tiling = MagicMock()
        mock_sent_tok = MagicMock()
        mock_word_tok = MagicMock()

        mock_stem_module.WordNetLemmatizer = mock_lemmatizer
        mock_tokenize_module.TextTilingTokenizer = mock_tiling
        mock_tokenize_module.sent_tokenize = mock_sent_tok
        mock_tokenize_module.word_tokenize = mock_word_tok

        # Act
        with patch(
            "lexile_corpus_tuner.estimators.lexile_v2_preprocessing.import_module"
        ) as mock_import:
            mock_import.side_effect = [mock_stem_module, mock_tokenize_module]

            original_cache = _get_nltk_cache()
            _set_nltk_cache(None)

            try:
                # First call should import
                result1 = _ensure_nltk_dependencies()
                # Second call should use cache
                result2 = _ensure_nltk_dependencies()

                # Assert
                assert result1 is result2
                assert mock_import.call_count == 2  # Only called during first call
            finally:
                _set_nltk_cache(original_cache)

    def test_ensure_nltk_dependencies_missing_nltk_raises_import_error(self) -> None:
        """Test that missing nltk raises ImportError with helpful message."""
        # Arrange & Act
        with patch(
            "lexile_corpus_tuner.estimators.lexile_v2_preprocessing.import_module"
        ) as mock_import:
            mock_import.side_effect = ModuleNotFoundError("No module named 'nltk'")

            original_cache = _get_nltk_cache()
            _set_nltk_cache(None)

            try:
                with pytest.raises(ImportError, match="nltk is required"):
                    _ensure_nltk_dependencies()
            finally:
                _set_nltk_cache(original_cache)


class TestSegmentText:
    """Tests for _segment_text function."""

    def test_segment_text_returns_list(self) -> None:
        """Test that _segment_text returns a list of strings."""
        # Arrange
        mock_lemmatizer = MagicMock()
        mock_tiling = MagicMock()
        mock_tiling_instance = MagicMock()
        mock_tiling.return_value = mock_tiling_instance
        mock_tiling_instance.tokenize.return_value = ["segment 1", "segment 2"]
        mock_sent_tok = MagicMock()
        mock_word_tok = MagicMock()

        # Act
        with patch(
            "lexile_corpus_tuner.estimators.lexile_v2_preprocessing._ensure_nltk_dependencies"
        ) as mock_deps:
            mock_deps.return_value = (
                mock_lemmatizer,
                mock_tiling,
                mock_sent_tok,
                mock_word_tok,
            )

            from lexile_corpus_tuner.estimators.lexile_v2_preprocessing import (
                _segment_text,  # pyright: ignore[reportPrivateUsage]
            )

            result = _segment_text("Sample text for segmentation.", [])

        # Assert
        assert isinstance(result, list)
        assert result == ["segment 1", "segment 2"]

    def test_segment_text_handles_value_error(self) -> None:
        """Test that _segment_text handles ValueError from tokenizer."""
        # Arrange
        mock_lemmatizer = MagicMock()
        mock_tiling = MagicMock()
        mock_tiling_instance = MagicMock()
        mock_tiling.return_value = mock_tiling_instance
        mock_tiling_instance.tokenize.side_effect = ValueError("Text too short")
        mock_sent_tok = MagicMock()
        mock_word_tok = MagicMock()

        # Act
        with patch(
            "lexile_corpus_tuner.estimators.lexile_v2_preprocessing._ensure_nltk_dependencies"
        ) as mock_deps:
            mock_deps.return_value = (
                mock_lemmatizer,
                mock_tiling,
                mock_sent_tok,
                mock_word_tok,
            )

            from lexile_corpus_tuner.estimators.lexile_v2_preprocessing import (
                _segment_text,  # pyright: ignore[reportPrivateUsage]
            )

            result = _segment_text("Short text.", [])

        # Assert
        assert result == ["Short text."]

    def test_segment_text_handles_tuple_result(self) -> None:
        """Test that _segment_text handles tuple result from tokenizer."""
        # Arrange
        mock_lemmatizer = MagicMock()
        mock_tiling = MagicMock()
        mock_tiling_instance = MagicMock()
        mock_tiling.return_value = mock_tiling_instance
        # Some tokenizers return a tuple where first element contains segments
        mock_tiling_instance.tokenize.return_value = (["seg1", "seg2"], "metadata")
        mock_sent_tok = MagicMock()
        mock_word_tok = MagicMock()

        # Act
        with patch(
            "lexile_corpus_tuner.estimators.lexile_v2_preprocessing._ensure_nltk_dependencies"
        ) as mock_deps:
            mock_deps.return_value = (
                mock_lemmatizer,
                mock_tiling,
                mock_sent_tok,
                mock_word_tok,
            )

            from lexile_corpus_tuner.estimators.lexile_v2_preprocessing import (
                _segment_text,  # pyright: ignore[reportPrivateUsage]
            )

            result = _segment_text("Sample text.", [])

        # Assert
        assert result == ["seg1", "seg2"]

    def test_segment_text_handles_string_result(self) -> None:
        """Test that _segment_text handles string result from tokenizer."""
        # Arrange
        mock_lemmatizer = MagicMock()
        mock_tiling = MagicMock()
        mock_tiling_instance = MagicMock()
        mock_tiling.return_value = mock_tiling_instance
        mock_tiling_instance.tokenize.return_value = "Single segment as string"
        mock_sent_tok = MagicMock()
        mock_word_tok = MagicMock()

        # Act
        with patch(
            "lexile_corpus_tuner.estimators.lexile_v2_preprocessing._ensure_nltk_dependencies"
        ) as mock_deps:
            mock_deps.return_value = (
                mock_lemmatizer,
                mock_tiling,
                mock_sent_tok,
                mock_word_tok,
            )

            from lexile_corpus_tuner.estimators.lexile_v2_preprocessing import (
                _segment_text,  # pyright: ignore[reportPrivateUsage]
            )

            result = _segment_text("Sample text.", [])

        # Assert
        assert result == ["Single segment as string"]

    def test_segment_text_passes_stopwords(self) -> None:
        """Test that _segment_text passes stopwords to tokenizer."""
        # Arrange
        mock_lemmatizer = MagicMock()
        mock_tiling = MagicMock()
        mock_tiling_instance = MagicMock()
        mock_tiling.return_value = mock_tiling_instance
        mock_tiling_instance.tokenize.return_value = ["segment"]
        mock_sent_tok = MagicMock()
        mock_word_tok = MagicMock()

        stopwords = ["the", "a", "an"]

        # Act
        with patch(
            "lexile_corpus_tuner.estimators.lexile_v2_preprocessing._ensure_nltk_dependencies"
        ) as mock_deps:
            mock_deps.return_value = (
                mock_lemmatizer,
                mock_tiling,
                mock_sent_tok,
                mock_word_tok,
            )

            from lexile_corpus_tuner.estimators.lexile_v2_preprocessing import (
                _segment_text,  # pyright: ignore[reportPrivateUsage]
            )

            _segment_text("Sample text.", stopwords)

        # Assert
        mock_tiling.assert_called_once_with(stopwords=["the", "a", "an"])


class TestLemmatizeSegments:
    """Tests for _lemmatize_segments function."""

    def test_lemmatize_segments_returns_list(self) -> None:
        """Test that _lemmatize_segments returns a list of lemmas."""
        # Arrange
        mock_lemmatizer_class = MagicMock()
        mock_lemmatizer_instance = MagicMock()
        mock_lemmatizer_class.return_value = mock_lemmatizer_instance
        mock_lemmatizer_instance.lemmatize.side_effect = _mock_lower

        mock_tiling = MagicMock()
        mock_sent_tok = MagicMock(return_value=["Hello world."])
        mock_word_tok = MagicMock(return_value=["Hello", "world"])

        # Act
        with patch(
            "lexile_corpus_tuner.estimators.lexile_v2_preprocessing._ensure_nltk_dependencies"
        ) as mock_deps:
            mock_deps.return_value = (
                mock_lemmatizer_class,
                mock_tiling,
                mock_sent_tok,
                mock_word_tok,
            )

            from lexile_corpus_tuner.estimators.lexile_v2_preprocessing import (
                _lemmatize_segments,  # pyright: ignore[reportPrivateUsage]
            )

            result = _lemmatize_segments(["Hello world."])

        # Assert
        assert isinstance(result, list)
        assert "hello" in result
        assert "world" in result

    def test_lemmatize_segments_handles_multiple_sentences(self) -> None:
        """Test that _lemmatize_segments handles multiple sentences."""
        # Arrange
        mock_lemmatizer_class = MagicMock()
        mock_lemmatizer_instance = MagicMock()
        mock_lemmatizer_class.return_value = mock_lemmatizer_instance
        mock_lemmatizer_instance.lemmatize.side_effect = _mock_lower

        mock_tiling = MagicMock()
        mock_sent_tok = MagicMock(return_value=["First sentence.", "Second sentence."])
        mock_word_tok = MagicMock(
            side_effect=[["First", "sentence"], ["Second", "sentence"]]
        )

        # Act
        with patch(
            "lexile_corpus_tuner.estimators.lexile_v2_preprocessing._ensure_nltk_dependencies"
        ) as mock_deps:
            mock_deps.return_value = (
                mock_lemmatizer_class,
                mock_tiling,
                mock_sent_tok,
                mock_word_tok,
            )

            from lexile_corpus_tuner.estimators.lexile_v2_preprocessing import (
                _lemmatize_segments,  # pyright: ignore[reportPrivateUsage]
            )

            result = _lemmatize_segments(["First sentence. Second sentence."])

        # Assert
        assert "first" in result
        assert "second" in result

    def test_lemmatize_segments_keeps_non_alpha_tokens(self) -> None:
        """Test that non-alphabetic tokens are kept as-is."""
        # Arrange
        mock_lemmatizer_class = MagicMock()
        mock_lemmatizer_instance = MagicMock()
        mock_lemmatizer_class.return_value = mock_lemmatizer_instance
        mock_lemmatizer_instance.lemmatize.side_effect = _mock_lower

        mock_tiling = MagicMock()
        mock_sent_tok = MagicMock(return_value=["Text 123."])
        mock_word_tok = MagicMock(return_value=["Text", "123", "."])

        # Act
        with patch(
            "lexile_corpus_tuner.estimators.lexile_v2_preprocessing._ensure_nltk_dependencies"
        ) as mock_deps:
            mock_deps.return_value = (
                mock_lemmatizer_class,
                mock_tiling,
                mock_sent_tok,
                mock_word_tok,
            )

            from lexile_corpus_tuner.estimators.lexile_v2_preprocessing import (
                _lemmatize_segments,  # pyright: ignore[reportPrivateUsage]
            )

            result = _lemmatize_segments(["Text 123."])

        # Assert
        assert "text" in result
        assert "123" in result
        assert "." in result

    def test_lemmatize_segments_empty_input(self) -> None:
        """Test that empty input returns empty list."""
        # Arrange
        mock_lemmatizer_class = MagicMock()
        mock_lemmatizer_instance = MagicMock()
        mock_lemmatizer_class.return_value = mock_lemmatizer_instance

        mock_tiling = MagicMock()
        mock_sent_tok = MagicMock(return_value=[])
        mock_word_tok = MagicMock()

        # Act
        with patch(
            "lexile_corpus_tuner.estimators.lexile_v2_preprocessing._ensure_nltk_dependencies"
        ) as mock_deps:
            mock_deps.return_value = (
                mock_lemmatizer_class,
                mock_tiling,
                mock_sent_tok,
                mock_word_tok,
            )

            from lexile_corpus_tuner.estimators.lexile_v2_preprocessing import (
                _lemmatize_segments,  # pyright: ignore[reportPrivateUsage]
            )

            result = _lemmatize_segments([])

        # Assert
        assert result == []


class TestVectorizeWithLexilePipeline:
    """Tests for vectorize_with_lexile_pipeline function."""

    def test_vectorize_with_lexile_pipeline_returns_matrix(self) -> None:
        """Test that vectorize_with_lexile_pipeline returns tokenizer output."""
        # Arrange
        mock_tokenizer = MagicMock()
        mock_matrix: list[list[float]] = [[0.1, 0.2, 0.3]]
        mock_tokenizer.texts_to_matrix.return_value = mock_matrix

        mock_lemmatizer_class = MagicMock()
        mock_lemmatizer_instance = MagicMock()
        mock_lemmatizer_class.return_value = mock_lemmatizer_instance
        mock_lemmatizer_instance.lemmatize.side_effect = _mock_lower

        mock_tiling = MagicMock()
        mock_tiling_instance = MagicMock()
        mock_tiling.return_value = mock_tiling_instance
        mock_tiling_instance.tokenize.return_value = ["segment"]

        mock_sent_tok = MagicMock(return_value=["Sample text."])
        mock_word_tok = MagicMock(return_value=["Sample", "text"])

        # Act
        with patch(
            "lexile_corpus_tuner.estimators.lexile_v2_preprocessing._ensure_nltk_dependencies"
        ) as mock_deps:
            mock_deps.return_value = (
                mock_lemmatizer_class,
                mock_tiling,
                mock_sent_tok,
                mock_word_tok,
            )

            from lexile_corpus_tuner.estimators.lexile_v2_preprocessing import (
                vectorize_with_lexile_pipeline,
            )

            result = vectorize_with_lexile_pipeline(
                "Sample text.", mock_tokenizer, ["the", "a"]
            )

        # Assert
        assert result == mock_matrix
        mock_tokenizer.texts_to_matrix.assert_called_once()

    def test_vectorize_with_lexile_pipeline_uses_tfidf_mode(self) -> None:
        """Test that vectorize uses tfidf mode."""
        # Arrange
        mock_tokenizer = MagicMock()
        mock_tokenizer.texts_to_matrix.return_value = [[0.0]]

        mock_lemmatizer_class = MagicMock()
        mock_lemmatizer_instance = MagicMock()
        mock_lemmatizer_class.return_value = mock_lemmatizer_instance
        mock_lemmatizer_instance.lemmatize.side_effect = _mock_identity

        mock_tiling = MagicMock()
        mock_tiling_instance = MagicMock()
        mock_tiling.return_value = mock_tiling_instance
        mock_tiling_instance.tokenize.return_value = ["text"]

        mock_sent_tok = MagicMock(return_value=["text"])
        mock_word_tok = MagicMock(return_value=["text"])

        # Act
        with patch(
            "lexile_corpus_tuner.estimators.lexile_v2_preprocessing._ensure_nltk_dependencies"
        ) as mock_deps:
            mock_deps.return_value = (
                mock_lemmatizer_class,
                mock_tiling,
                mock_sent_tok,
                mock_word_tok,
            )

            from lexile_corpus_tuner.estimators.lexile_v2_preprocessing import (
                vectorize_with_lexile_pipeline,
            )

            vectorize_with_lexile_pipeline("text", mock_tokenizer, [])

        # Assert - check that tfidf mode was passed
        call_kwargs: dict[str, Any] = mock_tokenizer.texts_to_matrix.call_args[1]
        assert call_kwargs["mode"] == "tfidf"
