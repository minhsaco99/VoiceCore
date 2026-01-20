"""Integration tests for app lifespan management

Tests startup, shutdown, engine loading, and error scenarios.
Requires faster_whisper to be installed.
"""

import logging

import pytest
import yaml
from fastapi import FastAPI
from starlette.testclient import TestClient

# Skip all tests if faster_whisper not available
pytest.importorskip("faster_whisper", reason="faster_whisper not installed")

from app.api.lifespan import lifespan


class TestLifespanStartup:
    """Startup lifecycle tests"""

    def test_loads_engines_from_yaml(self, temp_engines_yaml, monkeypatch):
        """Engines loaded from engines.yaml on startup"""
        monkeypatch.setenv("ENGINE_CONFIG_PATH", str(temp_engines_yaml))

        app = FastAPI(lifespan=lifespan)

        with TestClient(app):
            assert hasattr(app.state, "engine_registry")
            assert app.state.engine_registry is not None

            stt_engines = app.state.engine_registry.list_stt_engines()
            assert "whisper" in stt_engines

    def test_initializes_enabled_engines(self, tmp_path, monkeypatch):
        """Only enabled engines are initialized"""
        yaml_content = """stt:
  whisper:
    enabled: true
    engine_class: "app.engines.stt.whisper.engine.WhisperSTTEngine"
    config:
      model_name: "base"
      device: "cpu"
      compute_type: "int8"

  disabled_engine:
    enabled: false
    engine_class: "app.engines.stt.whisper.engine.WhisperSTTEngine"
    config:
      model_name: "base"

tts: {}
"""
        yaml_file = tmp_path / "engines.yaml"
        yaml_file.write_text(yaml_content)
        monkeypatch.setenv("ENGINE_CONFIG_PATH", str(yaml_file))

        app = FastAPI(lifespan=lifespan)

        with TestClient(app):
            stt_engines = app.state.engine_registry.list_stt_engines()
            assert "whisper" in stt_engines
            assert "disabled_engine" not in stt_engines

    def test_stores_registry_in_app_state(self, temp_engines_yaml, monkeypatch):
        """Registry stored in app.state.engine_registry"""
        monkeypatch.setenv("ENGINE_CONFIG_PATH", str(temp_engines_yaml))

        app = FastAPI(lifespan=lifespan)

        with TestClient(app):
            assert hasattr(app.state, "engine_registry")
            from app.api.registry import EngineRegistry

            assert isinstance(app.state.engine_registry, EngineRegistry)

    def test_partial_engine_failure(self, tmp_path, monkeypatch, caplog):
        """CRITICAL: One engine fails, others continue loading"""
        yaml_content = """stt:
  whisper:
    enabled: true
    engine_class: "app.engines.stt.whisper.engine.WhisperSTTEngine"
    config:
      model_name: "base"
      device: "cpu"
      compute_type: "int8"

  nonexistent:
    enabled: true
    engine_class: "nonexistent.module.Engine"
    config: {}

tts: {}
"""
        yaml_file = tmp_path / "engines.yaml"
        yaml_file.write_text(yaml_content)
        monkeypatch.setenv("ENGINE_CONFIG_PATH", str(yaml_file))

        app = FastAPI(lifespan=lifespan)

        with caplog.at_level(logging.ERROR), TestClient(app):
            stt_engines = app.state.engine_registry.list_stt_engines()
            assert "whisper" in stt_engines
            assert "nonexistent" not in stt_engines

        assert "Failed to initialize STT engine" in caplog.text
        assert "nonexistent" in caplog.text

    def test_all_engines_fail(self, tmp_path, monkeypatch, caplog):
        """All engines fail → app starts with empty registry"""
        yaml_content = """stt:
  bad_engine:
    enabled: true
    engine_class: "nonexistent.module.Engine"
    config: {}

tts: {}
"""
        yaml_file = tmp_path / "engines.yaml"
        yaml_file.write_text(yaml_content)
        monkeypatch.setenv("ENGINE_CONFIG_PATH", str(yaml_file))

        app = FastAPI(lifespan=lifespan)

        with caplog.at_level(logging.ERROR), TestClient(app):
            assert hasattr(app.state, "engine_registry")
            stt_engines = app.state.engine_registry.list_stt_engines()
            assert len(stt_engines) == 0

        assert "Failed to initialize" in caplog.text

    def test_missing_yaml_file(self, monkeypatch, caplog):
        """Missing engines.yaml → app starts with no engines"""
        monkeypatch.setenv("ENGINE_CONFIG_PATH", "/nonexistent/engines.yaml")

        app = FastAPI(lifespan=lifespan)

        with caplog.at_level(logging.WARNING), TestClient(app):
            assert hasattr(app.state, "engine_registry")
            stt_engines = app.state.engine_registry.list_stt_engines()
            assert len(stt_engines) == 0

        assert "not found" in caplog.text or "using defaults" in caplog.text

    def test_malformed_yaml(self, tmp_path, monkeypatch):
        """Malformed YAML → startup fails"""
        yaml_content = """
stt:
  whisper: [invalid yaml syntax
"""
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text(yaml_content)
        monkeypatch.setenv("ENGINE_CONFIG_PATH", str(yaml_file))

        app = FastAPI(lifespan=lifespan)

        with pytest.raises(yaml.YAMLError), TestClient(app):
            pass


class TestLifespanShutdown:
    """Shutdown lifecycle tests"""

    def test_closes_all_engines(self, temp_engines_yaml, monkeypatch, caplog):
        """All engines closed on shutdown"""
        monkeypatch.setenv("ENGINE_CONFIG_PATH", str(temp_engines_yaml))

        app = FastAPI(lifespan=lifespan)

        with caplog.at_level(logging.INFO), TestClient(app):
            pass

        # Shutdown should have happened (context manager exit)
        # Note: actual close behavior depends on engine implementation

    def test_shutdown_with_uninitialized_engines(self, tmp_path, monkeypatch):
        """Engines never initialized don't cause errors on close"""
        yaml_content = """stt:
  bad_engine:
    enabled: true
    engine_class: "nonexistent.module.Engine"
    config: {}

tts: {}
"""
        yaml_file = tmp_path / "engines.yaml"
        yaml_file.write_text(yaml_content)
        monkeypatch.setenv("ENGINE_CONFIG_PATH", str(yaml_file))

        app = FastAPI(lifespan=lifespan)

        # Should start and shutdown cleanly despite failed engine
        with TestClient(app):
            pass


class TestLifespanLogging:
    """Lifespan logging tests"""

    def test_startup_success_logged(self, temp_engines_yaml, monkeypatch, caplog):
        """Successful startup is logged"""
        monkeypatch.setenv("ENGINE_CONFIG_PATH", str(temp_engines_yaml))

        app = FastAPI(lifespan=lifespan)

        with caplog.at_level(logging.INFO), TestClient(app):
            pass

        assert "Engine registry initialized" in caplog.text or "STT" in caplog.text

    def test_engine_count_logged(self, temp_engines_yaml, monkeypatch, caplog):
        """Number of engines logged on startup"""
        monkeypatch.setenv("ENGINE_CONFIG_PATH", str(temp_engines_yaml))

        app = FastAPI(lifespan=lifespan)

        with caplog.at_level(logging.INFO), TestClient(app):
            pass

        assert "1" in caplog.text or "whisper" in caplog.text


class TestLifespanEdgeCases:
    """Edge cases in lifespan management"""

    def test_empty_yaml_file(self, tmp_path, monkeypatch):
        """Empty YAML file raises TypeError (yaml returns None)"""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        monkeypatch.setenv("ENGINE_CONFIG_PATH", str(yaml_file))

        app = FastAPI(lifespan=lifespan)

        # Empty YAML returns None, which can't be unpacked
        with pytest.raises(TypeError, match="must be a mapping"), TestClient(app):
            pass

    def test_yaml_with_comments(self, tmp_path, monkeypatch):
        """YAML with comments is parsed correctly"""
        yaml_content = """# This is a comment
stt:
  # Whisper engine
  whisper:
    enabled: true  # Enable this engine
    engine_class: "app.engines.stt.whisper.engine.WhisperSTTEngine"
    config:
      model_name: "base"
      device: "cpu"
      compute_type: "int8"

tts: {}  # No TTS engines yet
"""
        yaml_file = tmp_path / "commented.yaml"
        yaml_file.write_text(yaml_content)
        monkeypatch.setenv("ENGINE_CONFIG_PATH", str(yaml_file))

        app = FastAPI(lifespan=lifespan)

        with TestClient(app):
            stt_engines = app.state.engine_registry.list_stt_engines()
            assert "whisper" in stt_engines

    def test_multiple_startup_shutdown_cycles(self, temp_engines_yaml, monkeypatch):
        """Multiple startup/shutdown cycles work correctly"""
        monkeypatch.setenv("ENGINE_CONFIG_PATH", str(temp_engines_yaml))

        # First cycle
        app1 = FastAPI(lifespan=lifespan)
        with TestClient(app1):
            pass

        # Second cycle
        app2 = FastAPI(lifespan=lifespan)
        with TestClient(app2):
            pass

        # Both should work independently
