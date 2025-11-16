from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping

import yaml


@dataclass(slots=True)
class OpenAISettings:
    """Configuration block for OpenAI-powered rewriting."""

    enabled: bool = False
    model: str = "gpt-4.1-mini"
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    organization: str | None = None
    temperature: float = 0.3
    max_output_tokens: int = 400
    top_p: float = 0.95
    request_timeout: float = 60.0
    parallel_requests: int = 1


@dataclass(slots=True)
class LexileTunerConfig:
    """Configuration options for the corpus tuner pipeline."""

    window_size: int = 500
    stride: int = 250
    max_window_lexile: float = 450.0
    target_avg_lexile: float = 350.0
    avg_tolerance: float = 20.0
    max_passes: int = 3
    smoothing_kernel_size: int = 3
    estimator_name: str = "dummy"
    rewrite_enabled: bool = False
    rewrite_model: str | None = None
    lexile_v2_model_path: str | None = None
    lexile_v2_vectorizer_path: str | None = None
    lexile_v2_label_encoder_path: str | None = None
    lexile_v2_band_to_midpoint: Dict[str, float] = field(default_factory=dict)
    openai: OpenAISettings = field(default_factory=OpenAISettings)

    def to_dict(self) -> dict[str, Any]:
        """Return a mutable dictionary representation of the configuration."""
        return dict(asdict(self))


def _build_kwargs(data: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {field.name for field in fields(LexileTunerConfig)}
    kwargs = {key: data[key] for key in data if key in allowed}
    if "openai" in data:
        openai_value = data["openai"]
        if isinstance(openai_value, OpenAISettings):
            kwargs["openai"] = openai_value
        elif isinstance(openai_value, Mapping):
            kwargs["openai"] = _build_openai_settings(openai_value)
    return kwargs


def _build_openai_settings(data: Mapping[str, Any]) -> OpenAISettings:
    openai_allowed = {field.name for field in fields(OpenAISettings)}
    filtered = {key: data[key] for key in data if key in openai_allowed}
    return OpenAISettings(**filtered)


def config_from_dict(data: Mapping[str, Any] | None) -> LexileTunerConfig:
    """Build a LexileTunerConfig from a dictionary-like input."""
    if data is None:
        return LexileTunerConfig()
    return LexileTunerConfig(**_build_kwargs(data))


def config_from_yaml(path: str | Path) -> LexileTunerConfig:
    """Load configuration from a YAML file."""
    contents = Path(path).read_text(encoding="utf-8")
    parsed = yaml.safe_load(contents) or {}
    if not isinstance(parsed, MutableMapping):
        raise ValueError("Configuration YAML must define a mapping.")
    return config_from_dict(parsed)


def load_config(path: str | Path | None = None) -> LexileTunerConfig:
    """Load configuration from YAML when provided, otherwise return defaults."""
    if path is None:
        return LexileTunerConfig()
    return config_from_yaml(path)
