"""Tests for textutils module."""

from lexile_corpus_tuner.corpus_tuning_pipeline.textutils import (
    iter_tokens,
    normalize_text,
)


class TestNormalizeText:
    """Tests for normalize_text function."""

    def test_normalize_text_basic_string(self):
        """Test basic string normalization."""
        assert normalize_text("Hello World") == "hello world"

    def test_normalize_text_type_coercion_int(self):
        """Test integer input is converted to string."""
        assert normalize_text(123) == "123"

    def test_normalize_text_type_coercion_float(self):
        """Test float input is converted to string."""
        assert normalize_text(123.45) == "123.45"

    def test_normalize_text_type_coercion_none(self):
        """Test None input is converted to string 'None'."""
        # Note: str(None) is "None", which lowercases to "none"
        assert normalize_text(None) == "none"

    def test_normalize_text_type_coercion_object(self):
        """Test object with __str__ is converted correctly."""

        class CustomObj:
            def __str__(self):
                return "Custom Object"

        assert normalize_text(CustomObj()) == "custom object"

    def test_normalize_text_whitespace_multiple_spaces(self):
        """Test multiple spaces are collapsed to single space."""
        assert normalize_text("Hello   World") == "hello world"

    def test_normalize_text_whitespace_tabs_newlines(self):
        """Test tabs and newlines are replaced by space."""
        assert normalize_text("Hello\tWorld\nTest") == "hello world test"

    def test_normalize_text_whitespace_stripping(self):
        """Test leading and trailing whitespace is stripped."""
        assert normalize_text("  Hello World  ") == "hello world"

    def test_normalize_text_unicode_normalization(self):
        """Test Unicode NFKC normalization."""
        # \u2121 is TEL symbol, normalizes to TEL
        # \u00BD is 1/2 fraction, normalizes to 1 2044 2
        # Let's use a simpler one: \uFB01 is 'fi' ligature
        assert normalize_text("ﬁle") == "file"

    def test_normalize_text_unicode_case_folding(self):
        """Test Unicode case folding."""
        assert normalize_text("Strauß") == "strauß"

    def test_normalize_text_empty_string(self):
        """Test empty string returns empty string."""
        assert normalize_text("") == ""

    def test_normalize_text_whitespace_only(self):
        """Test whitespace-only string returns empty string."""
        assert normalize_text("   \t\n   ") == ""


class TestIterTokens:
    """Tests for iter_tokens function."""

    def test_iter_tokens_simple_text(self):
        """Test tokenization of simple text."""
        tokens = list(iter_tokens("Hello World"))
        assert tokens == ["hello", "world"]

    def test_iter_tokens_punctuation(self):
        """Test tokenization handles punctuation correctly."""
        tokens = list(iter_tokens("Hello, World! This is a test."))
        assert tokens == ["hello", "world", "this", "is", "a", "test"]

    def test_iter_tokens_hyphens_apostrophes(self):
        """Test tokenization preserves internal hyphens and apostrophes."""
        tokens = list(iter_tokens("don't self-driving"))
        assert tokens == ["don't", "self-driving"]

    def test_iter_tokens_empty_text(self):
        """Test tokenization of empty text yields nothing."""
        tokens = list(iter_tokens(""))
        assert tokens == []

    def test_iter_tokens_unicode_text(self):
        """Test tokenization of Unicode text."""
        tokens = list(iter_tokens("Café résumé"))
        assert tokens == ["café", "résumé"]

    def test_iter_tokens_generator_behavior(self):
        """Test that iter_tokens returns a generator."""
        gen = iter_tokens("test")
        assert iter(gen) is gen
        assert list(gen) == ["test"]
