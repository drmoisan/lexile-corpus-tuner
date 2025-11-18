from __future__ import annotations

import math
import statistics

from ..analyzer.features import DocumentFeatures


def make_regression_features(doc: DocumentFeatures) -> dict[str, float]:
    """Map DocumentFeatures into a flat feature dict suitable for regression."""
    features: dict[str, float] = {
        "overall_msl": float(doc.overall_mean_sentence_length),
        "overall_mlf": float(doc.overall_mean_log_word_freq),
        "num_tokens": float(doc.total_tokens),
        "num_slices": float(doc.num_slices),
    }

    features["log_num_tokens"] = math.log(max(1.0, float(doc.total_tokens)))

    msl_values = [s.mean_sentence_length for s in doc.slice_features]
    mlf_values = [s.mean_log_word_freq for s in doc.slice_features]
    features["msl_std"] = (
        float(statistics.pstdev(msl_values)) if len(msl_values) > 1 else 0.0
    )
    features["mlf_std"] = (
        float(statistics.pstdev(mlf_values)) if len(mlf_values) > 1 else 0.0
    )

    features["overall_msl_sq"] = features["overall_msl"] ** 2
    features["overall_mlf_sq"] = features["overall_mlf"] ** 2
    features["msl_times_mlf"] = features["overall_msl"] * features["overall_mlf"]

    return features
