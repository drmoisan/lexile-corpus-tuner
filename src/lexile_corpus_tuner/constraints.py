from __future__ import annotations

from .config import LexileTunerConfig
from .models import ConstraintViolation, DocumentLexileStats


def find_violations(
    doc_stats: DocumentLexileStats, config: LexileTunerConfig
) -> list[ConstraintViolation]:
    """Identify window and document-level Lexile violations."""
    violations: list[ConstraintViolation] = []

    for score in doc_stats.window_scores:
        if score.lexile > config.max_window_lexile:
            violations.append(
                ConstraintViolation(
                    doc_id=score.window.doc_id,
                    window_id=score.window.window_id,
                    lexile=score.lexile,
                    reason=f"Window Lexile {score.lexile:.1f} exceeds max {config.max_window_lexile:.1f}",
                    start_token_idx=score.window.start_token_idx,
                    end_token_idx=score.window.end_token_idx,
                )
            )

    lower = config.target_avg_lexile - config.avg_tolerance
    upper = config.target_avg_lexile + config.avg_tolerance
    if not (lower <= doc_stats.avg_lexile <= upper):
        violations.append(
            ConstraintViolation(
                doc_id=doc_stats.doc_id,
                window_id=-1,
                lexile=doc_stats.avg_lexile,
                reason=(
                    f"Document average {doc_stats.avg_lexile:.1f} outside "
                    f"target range [{lower:.1f}, {upper:.1f}]"
                ),
                start_token_idx=-1,
                end_token_idx=-1,
            )
        )

    return violations


def has_hard_window_violations(violations: list[ConstraintViolation]) -> bool:
    """Return True if any violation references a specific window."""
    return any(v.window_id >= 0 for v in violations)
