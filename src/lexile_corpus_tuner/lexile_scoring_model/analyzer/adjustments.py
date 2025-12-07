from __future__ import annotations


def adjust_for_special_cases(
    raw_lexile: float,
    *,
    is_picture_book: bool = False,
    is_emergent_nonfiction: bool = False,
) -> float:
    """Apply Lexile-style adjustments for picture books and emergent nonfiction."""
    adjustment = 0.0
    if is_picture_book:
        adjustment -= 120.0
    if is_emergent_nonfiction:
        adjustment -= 120.0
    return raw_lexile + adjustment
