"""
lexile_corpus_tuner package exports convenience helpers for library consumers.
"""

from __future__ import annotations

from .config import LexileTunerConfig, config_from_dict, config_from_yaml, load_config
from .estimators import build_estimator_from_config, create_estimator
from .pipeline import process_corpus, process_document

__all__ = [
    "LexileTunerConfig",
    "config_from_dict",
    "config_from_yaml",
    "load_config",
    "create_estimator",
    "build_estimator_from_config",
    "process_corpus",
    "process_document",
]

__version__ = "0.1.0"
