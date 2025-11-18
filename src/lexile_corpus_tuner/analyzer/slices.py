from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from ..textutils import iter_tokens

SENTENCE_END_RE = re.compile(r"[.?!;]")


@dataclass(slots=True)
class Slice:
    slice_id: int
    text: str
    tokens: list[str]
    sentence_lengths: list[int]


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using a simple punctuation-based heuristic."""
    if not text:
        return []
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    sentences: list[str] = []
    current_chars: list[str] = []
    for ch in normalized:
        current_chars.append(ch)
        if SENTENCE_END_RE.match(ch):
            sentence = "".join(current_chars).strip()
            if sentence:
                sentences.append(sentence)
            current_chars = []

    remainder = "".join(current_chars).strip()
    if remainder:
        sentences.append(remainder)
    return sentences


def build_slices(text: str, min_words: int = 125) -> list[Slice]:
    """Build Lexile-style slices of >= min_words extended to sentence boundary."""
    sentences = split_into_sentences(text)
    if not sentences:
        sentences = [text]

    slices: list[Slice] = []
    current_tokens: list[str] = []
    current_sentence_lengths: list[int] = []
    current_text_parts: list[str] = []
    slice_id = 0

    for sentence in sentences:
        sent_tokens = list(iter_tokens(sentence))
        if not sent_tokens:
            continue

        current_text_parts.append(sentence.strip())
        current_tokens.extend(sent_tokens)
        current_sentence_lengths.append(len(sent_tokens))

        if len(current_tokens) >= min_words:
            slices.append(
                Slice(
                    slice_id=slice_id,
                    text=" ".join(current_text_parts).strip(),
                    tokens=list(current_tokens),
                    sentence_lengths=list(current_sentence_lengths),
                )
            )
            slice_id += 1
            current_tokens = []
            current_sentence_lengths = []
            current_text_parts = []

    if current_tokens:
        slices.append(
            Slice(
                slice_id=slice_id,
                text=" ".join(current_text_parts).strip(),
                tokens=list(current_tokens),
                sentence_lengths=list(current_sentence_lengths),
            )
        )

    if not slices:
        fallback_tokens = list(iter_tokens(text))
        sentence_lengths = [len(fallback_tokens)] if fallback_tokens else []
        slices.append(
            Slice(
                slice_id=0,
                text=text.strip(),
                tokens=fallback_tokens,
                sentence_lengths=sentence_lengths,
            )
        )

    return slices
