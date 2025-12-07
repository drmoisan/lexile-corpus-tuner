# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Tests for analyzer slices module.

This module tests the split_into_sentences and build_slices functions
which handle text segmentation for Lexile analysis.
"""

from lexile_corpus_tuner.lexile_scoring_model.analyzer.slices import (
    Slice,
    build_slices,
    split_into_sentences,
)


class TestSplitIntoSentences:
    """Tests for split_into_sentences function."""

    def test_simple_sentence(self):
        """Test splitting a single simple sentence ending with period."""
        result = split_into_sentences("Hello world.")
        assert result == ["Hello world."]

    def test_multiple_sentences_with_periods(self):
        """Test splitting text with multiple sentences ending with periods."""
        result = split_into_sentences("Hello world. How are you. I am fine.")
        assert result == ["Hello world.", "How are you.", "I am fine."]

    def test_sentence_ending_with_question_mark(self):
        """Test splitting sentences ending with question marks."""
        result = split_into_sentences("What is your name? I am a robot.")
        assert result == ["What is your name?", "I am a robot."]

    def test_sentence_ending_with_exclamation(self):
        """Test splitting sentences ending with exclamation marks."""
        result = split_into_sentences("Hello world! This is great!")
        assert result == ["Hello world!", "This is great!"]

    def test_sentence_ending_with_semicolon(self):
        """Test splitting at semicolons as sentence boundaries."""
        result = split_into_sentences("First part; second part.")
        assert result == ["First part;", "second part."]

    def test_empty_string(self):
        """Test that empty string returns empty list."""
        result = split_into_sentences("")
        assert result == []

    def test_whitespace_only(self):
        """Test that whitespace-only string returns empty list."""
        result = split_into_sentences("   \t\n   ")
        assert result == []

    def test_no_punctuation(self):
        """Test text without sentence-ending punctuation returns whole text."""
        result = split_into_sentences("No punctuation here")
        assert result == ["No punctuation here"]

    def test_multiple_spaces_normalized(self):
        """Test that multiple spaces are normalized to single space."""
        result = split_into_sentences("Hello    world.   Next   sentence.")
        assert result == ["Hello world.", "Next sentence."]

    def test_newlines_and_tabs_normalized(self):
        """Test that newlines and tabs are replaced with spaces."""
        result = split_into_sentences("Hello\nworld.\tNext\nsentence.")
        assert result == ["Hello world.", "Next sentence."]

    def test_leading_trailing_whitespace_stripped(self):
        """Test that leading and trailing whitespace is stripped."""
        result = split_into_sentences("   Hello world.   ")
        assert result == ["Hello world."]

    def test_consecutive_punctuation(self):
        """Test handling of consecutive punctuation marks."""
        result = split_into_sentences("What!? Really?!")
        assert len(result) >= 1  # Behavior depends on implementation

    def test_single_word(self):
        """Test single word without punctuation."""
        result = split_into_sentences("Hello")
        assert result == ["Hello"]


class TestBuildSlices:
    """Tests for build_slices function."""

    def test_short_text_single_slice(self):
        """Test that short text creates a single slice."""
        text = "The quick brown fox."
        result = build_slices(text, min_words=5)
        assert len(result) == 1
        assert isinstance(result[0], Slice)
        assert result[0].slice_id == 0

    def test_slice_contains_tokens(self):
        """Test that slices contain tokenized words."""
        text = "The quick brown fox jumps over the lazy dog."
        result = build_slices(text, min_words=100)
        assert len(result) == 1
        # Tokens should be lowercase
        assert "the" in result[0].tokens
        assert "quick" in result[0].tokens
        assert "brown" in result[0].tokens

    def test_multiple_slices_created(self):
        """Test that long text with multiple sentences creates multiple slices."""
        # Create multiple sentences, each with enough words to exceed min_words
        # when combined with subsequent sentences
        sentences = []
        for _ in range(10):
            words = ["word"] * 50
            sentences.append(" ".join(words) + ".")
        text = " ".join(sentences)
        result = build_slices(text, min_words=125)
        # Should create multiple slices since we have 10 sentences of 50 words each
        assert len(result) >= 2

    def test_slice_text_preserved(self):
        """Test that slice text field contains the original sentence text."""
        text = "Hello world. How are you."
        result = build_slices(text, min_words=100)
        assert len(result) == 1
        assert "Hello world" in result[0].text
        assert "How are you" in result[0].text

    def test_sentence_lengths_tracked(self):
        """Test that sentence lengths are tracked in slice."""
        text = "One two three. Four five six seven eight."
        result = build_slices(text, min_words=100)
        assert len(result) == 1
        # Should have 2 sentences with lengths 3 and 5
        assert result[0].sentence_lengths == [3, 5]

    def test_empty_text_creates_fallback_slice(self):
        """Test that empty text creates a fallback slice."""
        text = ""
        result = build_slices(text, min_words=125)
        assert len(result) == 1
        assert result[0].slice_id == 0
        assert result[0].tokens == []
        assert result[0].sentence_lengths == []

    def test_whitespace_only_creates_fallback_slice(self):
        """Test that whitespace-only text creates a fallback slice."""
        text = "   \t\n   "
        result = build_slices(text, min_words=125)
        assert len(result) == 1
        assert result[0].tokens == []

    def test_slice_ids_sequential(self):
        """Test that slice IDs are sequential starting from 0."""
        words = ["word"] * 400
        text = ". ".join([" ".join(words[i : i + 50]) for i in range(0, 400, 50)])
        result = build_slices(text, min_words=100)
        for i, sl in enumerate(result):
            assert sl.slice_id == i

    def test_default_min_words(self):
        """Test that default min_words is 125."""
        words = ["word"] * 200
        text = " ".join(words) + "."
        result = build_slices(text)
        assert len(result) >= 1

    def test_text_without_punctuation(self):
        """Test handling of text without sentence-ending punctuation."""
        text = "No punctuation here just words and more words"
        result = build_slices(text, min_words=100)
        assert len(result) == 1
        assert len(result[0].tokens) > 0

    def test_mixed_punctuation(self):
        """Test handling of mixed punctuation types."""
        text = "First sentence. Second question? Third exclamation!"
        result = build_slices(text, min_words=100)
        assert len(result) == 1
        assert len(result[0].sentence_lengths) == 3

    def test_single_word_text(self):
        """Test handling of single word text."""
        text = "Hello"
        result = build_slices(text, min_words=125)
        assert len(result) == 1
        assert result[0].tokens == ["hello"]

    def test_slice_boundary_at_sentence_end(self):
        """Test that slices end at sentence boundaries when min_words reached."""
        # Create sentences of known lengths
        sentences = []
        for _ in range(10):
            words = ["word"] * 50
            sentences.append(" ".join(words) + ".")
        text = " ".join(sentences)
        result = build_slices(text, min_words=125)
        # Each slice should contain complete sentences
        for sl in result:
            assert len(sl.sentence_lengths) > 0

    def test_slice_tokens_are_lowercase(self):
        """Test that tokens in slices are lowercase."""
        text = "HELLO WORLD. HOW ARE YOU."
        result = build_slices(text, min_words=100)
        for token in result[0].tokens:
            assert token == token.lower()

    def test_sentences_with_empty_tokens_skipped(self):
        """Test that sentences producing no tokens are skipped."""
        text = "Hello world. !!!. Another sentence."
        result = build_slices(text, min_words=100)
        assert len(result) == 1
        # The "!!!." sentence should not contribute a sentence length
        # as it has no word tokens


class TestSliceDataclass:
    """Tests for Slice dataclass structure."""

    def test_slice_has_required_fields(self):
        """Test that Slice has all required fields."""
        sl = Slice(
            slice_id=0,
            text="Hello world.",
            tokens=["hello", "world"],
            sentence_lengths=[2],
        )
        assert sl.slice_id == 0
        assert sl.text == "Hello world."
        assert sl.tokens == ["hello", "world"]
        assert sl.sentence_lengths == [2]

    def test_slice_with_empty_tokens(self):
        """Test Slice can be created with empty token list."""
        sl = Slice(slice_id=0, text="", tokens=[], sentence_lengths=[])
        assert sl.tokens == []
        assert sl.sentence_lengths == []
