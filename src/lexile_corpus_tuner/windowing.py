from __future__ import annotations

from typing import List

from .models import Document, Token, Window


def create_windows(
    doc: Document,
    tokens: List[Token],
    window_size: int,
    stride: int,
) -> List[Window]:
    """Create overlapping windows of tokens for a document."""
    if not tokens:
        return []

    stride = max(1, stride)
    windows: List[Window] = []
    start_idx = 0
    window_id = 0

    while start_idx < len(tokens):
        end_idx = min(start_idx + max(1, window_size), len(tokens))
        start_char = tokens[start_idx].start_char
        end_char = tokens[end_idx - 1].end_char
        text = doc.text[start_char:end_char]
        windows.append(
            Window(
                doc_id=doc.doc_id,
                window_id=window_id,
                start_token_idx=start_idx,
                end_token_idx=end_idx,
                text=text,
            )
        )
        window_id += 1
        if end_idx == len(tokens):
            break
        start_idx += stride

    return windows
