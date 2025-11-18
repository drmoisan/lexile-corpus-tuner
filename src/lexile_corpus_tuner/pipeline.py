from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Tuple

from .config import LexileTunerConfig
from .constraints import find_violations, has_hard_window_violations
from .estimators import LexileEstimator
from .models import (
    ConstraintViolation,
    Document,
    DocumentLexileStats,
    Token,
    Window,
    WindowScore,
)
from .rewriting import Rewriter, RewriteRequest
from .scoring import compute_document_stats, score_windows, smooth_window_lexiles
from .tokenization import tokenize_words
from .windowing import create_windows


def process_document(
    doc: Document,
    config: LexileTunerConfig,
    estimator: LexileEstimator,
    rewriter: Rewriter,
) -> Tuple[Document, DocumentLexileStats, List[ConstraintViolation]]:
    """Run the Lexile tuning loop for a single document."""
    current_doc = Document(doc.doc_id, doc.text)
    last_stats = DocumentLexileStats(
        doc_id=doc.doc_id, avg_lexile=0.0, max_lexile=0.0, window_scores=[]
    )
    last_violations: List[ConstraintViolation] = []
    passes = max(1, config.max_passes)

    for _ in range(passes):
        tokens = tokenize_words(current_doc.text)
        windows = create_windows(current_doc, tokens, config.window_size, config.stride)
        window_scores = score_windows(windows, estimator)
        if config.smoothing_kernel_size > 1:
            smoothed = smooth_window_lexiles(
                window_scores, config.smoothing_kernel_size
            )
            window_scores = [
                WindowScore(window=score.window, lexile=value)
                for score, value in zip(window_scores, smoothed)
            ]
        stats = _build_document_stats(current_doc, window_scores)
        violations = find_violations(stats, config)
        last_stats, last_violations = stats, violations

        if not config.rewrite_enabled or not has_hard_window_violations(violations):
            return current_doc, stats, violations

        worst_violation = max(
            (v for v in violations if v.window_id >= 0),
            key=lambda v: v.lexile,
        )
        window_lookup = {w.window_id: w for w in windows}
        target_window = window_lookup.get(worst_violation.window_id)
        if target_window is None:
            break
        span_text = get_window_span_text(current_doc, target_window, tokens)
        rewrite_request = RewriteRequest(
            doc_id=current_doc.doc_id,
            window_id=target_window.window_id,
            text=span_text,
            target_lexile=config.target_avg_lexile,
            violation=worst_violation,
            metadata={
                "max_window_lexile": config.max_window_lexile,
                "target_avg_lexile": config.target_avg_lexile,
                "avg_tolerance": config.avg_tolerance,
            },
        )
        rewritten = rewriter.rewrite(rewrite_request)
        current_doc = replace(
            current_doc,
            text=replace_window_span(current_doc, target_window, tokens, rewritten),
        )

    return current_doc, last_stats, last_violations


def process_corpus(
    documents: List[Document],
    config: LexileTunerConfig,
    estimator: LexileEstimator,
    rewriter: Rewriter,
) -> Dict[str, Tuple[Document, DocumentLexileStats, List[ConstraintViolation]]]:
    """Process all documents and return the per-document outputs."""
    results: Dict[
        str, Tuple[Document, DocumentLexileStats, List[ConstraintViolation]]
    ] = {}
    for document in documents:
        results[document.doc_id] = process_document(
            document, config, estimator, rewriter
        )
    return results


def get_window_span_text(doc: Document, window: Window, tokens: List[Token]) -> str:
    """Extract the substring corresponding to a window."""
    if not tokens or window.end_token_idx <= window.start_token_idx:
        return ""
    start_token = tokens[window.start_token_idx]
    end_token = tokens[window.end_token_idx - 1]
    return doc.text[start_token.start_char : end_token.end_char]


def replace_window_span(
    doc: Document, window: Window, tokens: List[Token], new_text: str
) -> str:
    """Return the updated document text with the window replaced by new_text."""
    if not tokens or window.end_token_idx <= window.start_token_idx:
        return doc.text
    start_token = tokens[window.start_token_idx]
    end_token = tokens[window.end_token_idx - 1]
    start_char = start_token.start_char
    end_char = end_token.end_char
    return doc.text[:start_char] + new_text + doc.text[end_char:]


def _build_document_stats(
    doc: Document, window_scores: List[WindowScore]
) -> DocumentLexileStats:
    if not window_scores:
        return DocumentLexileStats(
            doc_id=doc.doc_id, avg_lexile=0.0, max_lexile=0.0, window_scores=[]
        )
    stats = compute_document_stats(window_scores)
    for stat in stats:
        if stat.doc_id == doc.doc_id:
            return stat
    return DocumentLexileStats(
        doc_id=doc.doc_id, avg_lexile=0.0, max_lexile=0.0, window_scores=[]
    )
