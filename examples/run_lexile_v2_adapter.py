"""
Tiny helper script to sanity check the lexile_v2 estimator wiring.
Update the placeholder paths before running.
"""

from __future__ import annotations

from lexile_corpus_tuner.config import LexileTunerConfig
from lexile_corpus_tuner.estimators import build_estimator_from_config


def main() -> None:
    config = LexileTunerConfig(
        estimator_name="lexile_v2",
        lexile_v2_model_path="/absolute/path/to/model.h5",
        lexile_v2_vectorizer_path="/absolute/path/to/vectorizer.pkl",
        lexile_v2_label_encoder_path="/absolute/path/to/label_encoder.pkl",
        lexile_v2_band_to_midpoint={
            "300-399": 350.0,
            "700-799": 750.0,
        },
    )

    estimator = build_estimator_from_config(config)
    samples = [
        "The cat sat on the mat. It was raining outside, but the cat was warm and happy.",
        "Quantum entanglement is a physical phenomenon that occurs when particles share proximity in ways such that their states cannot be described independently.",
    ]

    for sample in samples:
        score = estimator.predict_scalar(sample)
        print("-" * 40)
        print(sample)
        print(f"Estimated Lexile: {score:.2f}")


if __name__ == "__main__":
    main()
