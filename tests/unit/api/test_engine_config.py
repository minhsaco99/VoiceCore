"""Unit tests for engine configuration loading

Focus on dynamic import error handling and YAML parsing.
"""

from unittest.mock import MagicMock, patch

import pytest
import yaml
from pydantic import ValidationError

from app.api.engine_config import (
    EngineConfig,
    EngineConfigEntry,
    create_engine_instance,
    load_engine_config,
)


class TestLoadEngineConfig:
    """YAML loading and parsing tests"""

    def test_load_valid_yaml(self, tmp_path):
        """Load valid engines.yaml"""
        yaml_content = """
stt:
  whisper:
    enabled: true
    engine_class: "app.engines.stt.whisper.engine.WhisperSTTEngine"
    config:
      model_name: "base"
      device: "cpu"

tts: {}
"""
        yaml_file = tmp_path / "engines.yaml"
        yaml_file.write_text(yaml_content)

        config = load_engine_config(yaml_file)

        assert isinstance(config, EngineConfig)
        assert "whisper" in config.stt
        assert config.stt["whisper"].enabled is True
        assert config.stt["whisper"].config["model_name"] == "base"
        assert config.tts == {}

    def test_missing_yaml_returns_empty(self, tmp_path):
        """Missing engines.yaml returns EngineConfig()"""
        nonexistent_file = tmp_path / "nonexistent.yaml"

        config = load_engine_config(nonexistent_file)

        assert isinstance(config, EngineConfig)
        assert config.stt == {}
        assert config.tts == {}

    def test_malformed_yaml_raises(self, tmp_path):
        """Malformed YAML raises yaml.YAMLError"""
        yaml_content = """
stt:
  whisper:
    enabled: true
    config: [invalid
"""
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text(yaml_content)

        with pytest.raises(yaml.YAMLError):
            load_engine_config(yaml_file)

    def test_invalid_structure_ignored_extra_keys(self, tmp_path):
        """Valid YAML with extra keys - pydantic ignores by default"""
        yaml_content = """
invalid_key: value
stt: {}
tts: {}
"""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text(yaml_content)

        # pydantic v2 ignores extra fields by default
        config = load_engine_config(yaml_file)
        assert isinstance(config, EngineConfig)
        assert config.stt == {}
        assert config.tts == {}

    def test_empty_yaml_file_raises_type_error(self, tmp_path):
        """Empty YAML file raises TypeError (yaml returns None)"""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")

        # Note: This is a potential bug - empty YAML returns None
        # which can't be unpacked as **kwargs
        with pytest.raises(TypeError, match="must be a mapping"):
            load_engine_config(yaml_file)

    def test_missing_required_fields_raises(self, tmp_path):
        """Missing required fields raises ValidationError"""
        yaml_content = """
stt:
  whisper:
    enabled: true
    # Missing engine_class and config
"""
        yaml_file = tmp_path / "missing.yaml"
        yaml_file.write_text(yaml_content)

        with pytest.raises(ValidationError):
            load_engine_config(yaml_file)

    def test_disabled_engine_loaded(self, tmp_path):
        """Disabled engines are loaded in config but marked disabled"""
        yaml_content = """
stt:
  whisper:
    enabled: false
    engine_class: "app.engines.stt.whisper.engine.WhisperSTTEngine"
    config: {}

tts: {}
"""
        yaml_file = tmp_path / "disabled.yaml"
        yaml_file.write_text(yaml_content)

        config = load_engine_config(yaml_file)

        assert "whisper" in config.stt
        assert config.stt["whisper"].enabled is False


class TestCreateEngineInstanceDynamicImport:
    """CRITICAL: Dynamic import error handling tests"""

    def test_invalid_class_path_no_dot(self):
        """CRITICAL: Class path without dot raises error"""
        config_entry = EngineConfigEntry(
            enabled=True, engine_class="NoDotsHere", config={}
        )

        with pytest.raises(ValueError, match="not enough values to unpack"):
            create_engine_instance(config_entry)

    def test_module_not_found(self):
        """CRITICAL: Non-existent module raises ImportError"""
        config_entry = EngineConfigEntry(
            enabled=True,
            engine_class="nonexistent.module.Engine",
            config={},
        )

        with pytest.raises(ModuleNotFoundError, match="nonexistent"):
            create_engine_instance(config_entry)

    @patch("app.api.engine_config.importlib.import_module")
    def test_class_not_found_in_module(self, mock_import):
        """CRITICAL: Class doesn't exist in module"""
        # Mock module without the expected class
        mock_module = MagicMock()
        del mock_module.NonExistentClass  # Ensure attribute doesn't exist

        def import_side_effect(module_name):
            return mock_module

        mock_import.side_effect = import_side_effect

        config_entry = EngineConfigEntry(
            enabled=True,
            engine_class="some.module.NonExistentClass",
            config={},
        )

        with pytest.raises(AttributeError):
            create_engine_instance(config_entry)

    @patch("app.api.engine_config.importlib.import_module")
    def test_find_correct_config_class(self, mock_import):
        """Config class discovery skips base EngineConfig"""
        # Track calls to config class
        config_calls = []

        # Create a real class that tracks instantiation
        class MockWhisperConfig:
            def __init__(self, **kwargs):
                config_calls.append(kwargs)

        # Mock engine module
        mock_engine_module = MagicMock()
        mock_engine_instance = MagicMock()
        mock_engine_module.MockEngine = MagicMock(return_value=mock_engine_instance)

        # Create fake config module with real class (isinstance(x, type) works)
        class FakeConfigModule:
            WhisperConfig = MockWhisperConfig
            EngineConfig = type  # Should be skipped (name == "EngineConfig")

        mock_config_module = FakeConfigModule()

        def import_side_effect(module_name):
            if "config" in module_name:
                return mock_config_module
            return mock_engine_module

        mock_import.side_effect = import_side_effect

        config_entry = EngineConfigEntry(
            enabled=True,
            engine_class="app.engines.stt.whisper.engine.MockEngine",
            config={"model_name": "base", "device": "cpu"},
        )

        engine = create_engine_instance(config_entry)

        # Verify engine was created
        assert engine is not None
        # Verify config class was used (WhisperConfig, not EngineConfig)
        assert len(config_calls) == 1
        assert config_calls[0] == {"model_name": "base", "device": "cpu"}

    @patch("app.api.engine_config.importlib.import_module")
    def test_multiple_config_classes(self, mock_import):
        """Multiple *Config classes in module - first one chosen"""
        # Mock the module with multiple config classes
        mock_module = MagicMock()
        mock_module.WhisperSTTEngine = MagicMock(return_value=MagicMock())

        mock_config_module = MagicMock()
        # Simulate multiple config classes
        mock_config_module.WhisperConfig = MagicMock
        mock_config_module.AlternativeConfig = MagicMock
        mock_config_module.EngineConfig = MagicMock  # Should be skipped

        def import_side_effect(module_name):
            if "config" in module_name:
                return mock_config_module
            return mock_module

        mock_import.side_effect = import_side_effect

        config_entry = EngineConfigEntry(
            enabled=True,
            engine_class="app.engines.stt.whisper.engine.WhisperSTTEngine",
            config={},
        )

        # Should not raise, picks first non-base config
        engine = create_engine_instance(config_entry)
        assert engine is not None

    @patch("app.api.engine_config.importlib.import_module")
    @patch("app.models.engine.EngineConfig")
    def test_missing_config_module_fallback(self, mock_base_config, mock_import):
        """Missing config.py falls back to BaseEngineConfig"""
        # Mock engine module
        mock_engine_module = MagicMock()
        mock_engine_instance = MagicMock()
        mock_engine_module.MockEngine = MagicMock(return_value=mock_engine_instance)
        mock_base_config.return_value = MagicMock()

        def import_side_effect(module_name):
            if "config" in module_name:
                raise ImportError("No config module")
            return mock_engine_module

        mock_import.side_effect = import_side_effect

        config_entry = EngineConfigEntry(
            enabled=True,
            engine_class="app.engines.fake.engine.MockEngine",
            config={},
        )

        # Should still work, falling back to BaseEngineConfig
        engine = create_engine_instance(config_entry)
        assert engine is not None

    @patch("app.api.engine_config.importlib.import_module")
    def test_config_instantiation_failure(self, mock_import):
        """Config class raises ValidationError on invalid data"""
        # Mock engine module
        mock_engine_module = MagicMock()
        mock_engine_module.MockEngine = MagicMock()

        # Mock config module with config class that raises ValidationError
        mock_config_module = MagicMock()
        mock_config_module.MockConfig = MagicMock(
            side_effect=ValidationError.from_exception_data("MockConfig", [])
        )

        def import_side_effect(module_name):
            if "config" in module_name:
                return mock_config_module
            return mock_engine_module

        mock_import.side_effect = import_side_effect

        config_entry = EngineConfigEntry(
            enabled=True,
            engine_class="app.engines.fake.engine.MockEngine",
            config={"invalid_field": "value"},
        )

        with pytest.raises(ValidationError):
            create_engine_instance(config_entry)

    @patch("app.api.engine_config.importlib.import_module")
    @patch("app.models.engine.EngineConfig")
    def test_engine_instantiation_failure(self, mock_base_config, mock_import):
        """Engine class constructor raises error"""
        # Mock engine module with constructor that raises error
        mock_engine_module = MagicMock()
        mock_engine_module.MockEngine = MagicMock(
            side_effect=RuntimeError("Engine init failed")
        )
        mock_base_config.return_value = MagicMock()

        def import_side_effect(module_name):
            if "config" in module_name:
                raise ImportError("No config module")
            return mock_engine_module

        mock_import.side_effect = import_side_effect

        config_entry = EngineConfigEntry(
            enabled=True,
            engine_class="app.engines.fake.engine.MockEngine",
            config={},
        )

        with pytest.raises(RuntimeError, match="Engine init failed"):
            create_engine_instance(config_entry)


class TestEngineConfigEntry:
    """EngineConfigEntry validation tests"""

    def test_valid_entry(self):
        """Valid EngineConfigEntry"""
        entry = EngineConfigEntry(
            enabled=True,
            engine_class="app.engines.stt.whisper.engine.WhisperSTTEngine",
            config={"model_name": "base"},
        )

        assert entry.enabled is True
        assert entry.engine_class == "app.engines.stt.whisper.engine.WhisperSTTEngine"
        assert entry.config == {"model_name": "base"}

    def test_disabled_entry(self):
        """Disabled EngineConfigEntry"""
        entry = EngineConfigEntry(
            enabled=False,
            engine_class="app.engines.stt.whisper.engine.WhisperSTTEngine",
            config={},
        )

        assert entry.enabled is False

    def test_empty_config(self):
        """EngineConfigEntry with empty config dict"""
        entry = EngineConfigEntry(
            enabled=True,
            engine_class="app.engines.stt.whisper.engine.WhisperSTTEngine",
            config={},
        )

        assert entry.config == {}

    def test_missing_enabled_field(self):
        """Missing enabled field raises ValidationError"""
        with pytest.raises(ValidationError):
            EngineConfigEntry(
                engine_class="app.engines.stt.whisper.engine.WhisperSTTEngine",
                config={},
            )

    def test_invalid_engine_class_type(self):
        """engine_class must be string"""
        with pytest.raises(ValidationError):
            EngineConfigEntry(enabled=True, engine_class=123, config={})


class TestEngineConfigModel:
    """EngineConfig model tests"""

    def test_empty_config(self):
        """Empty EngineConfig"""
        config = EngineConfig()

        assert config.stt == {}
        assert config.tts == {}

    def test_config_with_stt_only(self):
        """EngineConfig with only STT engines"""
        config = EngineConfig(
            stt={
                "whisper": EngineConfigEntry(
                    enabled=True,
                    engine_class="app.engines.stt.whisper.engine.WhisperSTTEngine",
                    config={},
                )
            }
        )

        assert "whisper" in config.stt
        assert config.tts == {}

    def test_config_with_both_stt_and_tts(self):
        """EngineConfig with both STT and TTS engines"""
        config = EngineConfig(
            stt={
                "whisper": EngineConfigEntry(
                    enabled=True,
                    engine_class="app.engines.stt.whisper.engine.WhisperSTTEngine",
                    config={},
                )
            },
            tts={
                "coqui": EngineConfigEntry(
                    enabled=True,
                    engine_class="app.engines.tts.coqui.engine.CoquiTTSEngine",
                    config={},
                )
            },
        )

        assert "whisper" in config.stt
        assert "coqui" in config.tts
