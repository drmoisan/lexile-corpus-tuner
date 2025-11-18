from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import List

from ..frequency_loader import WordFrequency, load_frequency_table
from .slices import Slice


@dataclass(slots=True)
class SliceFeatures:
    slice_id: int
    num_tokens: int
    num_sentences: int
    mean_sentence_length: float
    mean_log_word_freq: float


@dataclass(slots=True)
class DocumentFeatures:
    num_slices: int
    total_tokens: int
    overall_mean_sentence_length: float
    overall_mean_log_word_freq: float
    slice_features: List[SliceFeatures]


def compute_document_features(slices: list[Slice]) -> DocumentFeatures:
    """Compute Lexile-style features for a list of slices."""
    freq_table = load_frequency_table()
    unseen_log_freq = _compute_unseen_floor(freq_table)

    slice_feats: list[SliceFeatures] = []
    total_tokens = 0
    all_sentence_lengths: list[int] = []
    all_token_log_freqs: list[float] = []

    for sl in slices:
        num_tokens = len(sl.tokens)
        total_tokens += num_tokens
        num_sentences = len(sl.sentence_lengths) or 1

        if sl.sentence_lengths:
            msl = float(statistics.mean(sl.sentence_lengths))
        elif num_tokens:
            msl = float(num_tokens)
        else:
            msl = 0.0

        token_log_freqs: list[float] = []
        for token in sl.tokens:
            wf = freq_table.get(token)
            token_log_freqs.append(wf.log_freq_per_5m if wf else unseen_log_freq)
        mlf = (
            float(statistics.mean(token_log_freqs))
            if token_log_freqs
            else unseen_log_freq
        )

        slice_feats.append(
            SliceFeatures(
                slice_id=sl.slice_id,
                num_tokens=num_tokens,
                num_sentences=num_sentences,
                mean_sentence_length=msl,
                mean_log_word_freq=mlf,
            )
        )

        all_sentence_lengths.extend(
            sl.sentence_lengths or ([num_tokens] if num_tokens else [])
        )
        all_token_log_freqs.extend(token_log_freqs)

    num_slices = len(slice_feats)
    overall_msl = (
        float(statistics.mean(all_sentence_lengths)) if all_sentence_lengths else 0.0
    )
    overall_mlf = (
        float(statistics.mean(all_token_log_freqs))
        if all_token_log_freqs
        else unseen_log_freq
    )

    return DocumentFeatures(
        num_slices=num_slices,
        total_tokens=total_tokens,
        overall_mean_sentence_length=overall_msl,
        overall_mean_log_word_freq=overall_mlf,
        slice_features=slice_feats,
    )


def _compute_unseen_floor(freq_table: dict[str, WordFrequency]) -> float:
    if not freq_table:
        return -20.0
    min_log_freq = min(entry.log_freq_per_5m for entry in freq_table.values())
    return min_log_freq - 1.0
