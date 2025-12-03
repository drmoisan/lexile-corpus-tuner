"""Advanced tests for config module covering edge cases."""

from pathlib import Path

import pytest
from lexile_corpus_tuner.config import (
    LexileTunerConfig,
    OpenAISettings,
    config_from_dict,
    config_from_yaml,
    load_config,
)


class TestConfigAdvanced:
    """Advanced tests for configuration loading."""

    def test_config_from_dict_extra_keys(self):
        """Test that extra keys in dict are ignored."""
        data = {
            "window_size": 100,
            "extra_key": "should be ignored",
            "openai": {"model": "gpt-4", "extra_openai_key": "ignored"},
        }
        config = config_from_dict(data)
        assert config.window_size == 100
        assert config.openai.model == "gpt-4"
        # Verify no error raised

    def test_config_from_dict_openai_object(self):
        """Test passing OpenAISettings object directly."""
        openai_settings = OpenAISettings(model="gpt-3.5-turbo")
        data = {"window_size": 200, "openai": openai_settings}
        config = config_from_dict(data)
        assert config.window_size == 200
        assert config.openai.model == "gpt-3.5-turbo"

    def test_config_from_yaml_empty_file(self, tmp_path: Path):
        """Test loading an empty YAML file returns default config."""
        config_file = tmp_path / "empty.yaml"
        config_file.touch()
        config = config_from_yaml(config_file)
        assert config == LexileTunerConfig()

    def test_config_from_yaml_invalid_structure_list(self, tmp_path: Path):
        """Test loading a YAML list raises ValueError."""
        config_file = tmp_path / "list.yaml"
        config_file.write_text("- item1\n- item2", encoding="utf-8")
        with pytest.raises(
            ValueError, match="Configuration YAML must define a mapping"
        ):
            config_from_yaml(config_file)

    def test_config_from_yaml_invalid_structure_scalar(self, tmp_path: Path):
        """Test loading a YAML scalar raises ValueError."""
        config_file = tmp_path / "scalar.yaml"
        config_file.write_text("just a string", encoding="utf-8")
        with pytest.raises(
            ValueError, match="Configuration YAML must define a mapping"
        ):
            config_from_yaml(config_file)

    def test_config_from_yaml_non_string_keys(self, tmp_path: Path):
        """Test loading a YAML with non-string keys raises ValueError."""
        config_file = tmp_path / "bad_keys.yaml"
        # YAML allows non-string keys
        config_file.write_text("123: value", encoding="utf-8")
        with pytest.raises(ValueError, match="Configuration keys must be strings"):
            config_from_yaml(config_file)

    def test_load_config_with_path(self, tmp_path: Path):
        """Test load_config with a valid path."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("window_size: 300", encoding="utf-8")
        config = load_config(config_file)
        assert config.window_size == 300

    def test_load_config_none(self):
        """Test load_config with None returns default."""
        config = load_config(None)
        assert config == LexileTunerConfig()

    def test_config_from_dict_none(self):
        """Test config_from_dict with None returns default."""
        config = config_from_dict(None)
        assert config == LexileTunerConfig()
