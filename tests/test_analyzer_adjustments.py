"""Tests for analyzer adjustments module.

This module tests the adjust_for_special_cases function which applies
Lexile-style adjustments for picture books and emergent nonfiction.
"""

from lexile_corpus_tuner.analyzer.adjustments import adjust_for_special_cases


class TestAdjustForSpecialCases:
    """Tests for adjust_for_special_cases function."""

    def test_no_adjustments_applied(self):
        """Test that raw lexile is returned unchanged when no flags are set."""
        raw_lexile = 500.0
        result = adjust_for_special_cases(raw_lexile)
        assert result == 500.0

    def test_picture_book_adjustment(self):
        """Test that picture book flag applies -120 adjustment."""
        raw_lexile = 500.0
        result = adjust_for_special_cases(raw_lexile, is_picture_book=True)
        assert result == 380.0

    def test_emergent_nonfiction_adjustment(self):
        """Test that emergent nonfiction flag applies -120 adjustment."""
        raw_lexile = 500.0
        result = adjust_for_special_cases(raw_lexile, is_emergent_nonfiction=True)
        assert result == 380.0

    def test_both_adjustments_applied(self):
        """Test that both flags together apply -240 adjustment."""
        raw_lexile = 500.0
        result = adjust_for_special_cases(
            raw_lexile, is_picture_book=True, is_emergent_nonfiction=True
        )
        assert result == 260.0

    def test_zero_raw_lexile_with_adjustments(self):
        """Test adjustments applied to zero lexile value."""
        result = adjust_for_special_cases(
            0.0, is_picture_book=True, is_emergent_nonfiction=True
        )
        assert result == -240.0

    def test_negative_raw_lexile(self):
        """Test adjustments with negative input lexile."""
        result = adjust_for_special_cases(-100.0, is_picture_book=True)
        assert result == -220.0

    def test_large_raw_lexile(self):
        """Test adjustments with large input lexile."""
        raw_lexile = 1500.0
        result = adjust_for_special_cases(raw_lexile, is_picture_book=True)
        assert result == 1380.0

    def test_float_precision_maintained(self):
        """Test that float precision is maintained in calculations."""
        raw_lexile = 500.5
        result = adjust_for_special_cases(raw_lexile, is_picture_book=True)
        assert result == 380.5

    def test_explicit_false_flags_no_adjustment(self):
        """Test that explicitly setting flags to False applies no adjustment."""
        raw_lexile = 500.0
        result = adjust_for_special_cases(
            raw_lexile, is_picture_book=False, is_emergent_nonfiction=False
        )
        assert result == 500.0
