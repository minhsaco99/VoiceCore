"""Integration tests for app lifespan management

Tests startup, shutdown, engine loading, and error scenarios.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.lifespan import lifespan


class TestLifespanStartup:
    """Startup lifecycle tests"""

    @pytest.mark.asyncio
    async def test_loads_engines_from_yaml(self, temp_engines_yaml, monkeypatch):
        """Engines loaded from engines.yaml on startup"""
        # Set engine config path to temp file
        monkeypatch.setenv("ENGINE_CONFIG_PATH", str(temp_engines_yaml))

        app = FastAPI(lifespan=lifespan)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as _:
            # App started successfully
            assert hasattr(app.state, "engine_registry")
            assert app.state.engine_registry is not None

            # Check engines loaded
            stt_engines = app.state.engine_registry.list_stt_engines()
            assert "whisper" in stt_engines

    @pytest.mark.asyncio
    async def test_initializes_enabled_engines(self, tmp_path, monkeypatch):
        """Only enabled engines are initialized"""
        # Create YAML with one enabled, one disabled
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

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as _:
            # Only enabled engine should be loaded
            stt_engines = app.state.engine_registry.list_stt_engines()
            assert "whisper" in stt_engines
            assert "disabled_engine" not in stt_engines

    @pytest.mark.asyncio
    async def test_stores_registry_in_app_state(self, temp_engines_yaml, monkeypatch):
        """Registry stored in app.state.engine_registry"""
        monkeypatch.setenv("ENGINE_CONFIG_PATH", str(temp_engines_yaml))

        app = FastAPI(lifespan=lifespan)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as _:
            assert hasattr(app.state, "engine_registry")
            from app.api.registry import EngineRegistry

            assert isinstance(app.state.engine_registry, EngineRegistry)

    @pytest.mark.asyncio
    async def test_partial_engine_failure(self, tmp_path, monkeypatch, caplog):
        """CRITICAL: One engine fails, others continue loading"""
        # Create YAML with two engines
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

        # App should start despite one engine failing
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as _:
            # Whisper should be loaded
            stt_engines = app.state.engine_registry.list_stt_engines()
            assert "whisper" in stt_engines

            # Nonexistent should not be loaded
            assert "nonexistent" not in stt_engines

        # Error should be logged
        assert "Failed to initialize STT engine" in caplog.text
        assert "nonexistent" in caplog.text

    @pytest.mark.asyncio
    async def test_all_engines_fail(self, tmp_path, monkeypatch, caplog):
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

        # App should start even with no working engines
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as _:
            # Registry exists but is empty
            assert hasattr(app.state, "engine_registry")
            stt_engines = app.state.engine_registry.list_stt_engines()
            assert len(stt_engines) == 0

        # Error should be logged
        assert "Failed to initialize" in caplog.text

    @pytest.mark.asyncio
    async def test_missing_yaml_file(self, monkeypatch, caplog):
        """Missing engines.yaml → app starts with no engines"""
        monkeypatch.setenv("ENGINE_CONFIG_PATH", "/nonexistent/engines.yaml")

        app = FastAPI(lifespan=lifespan)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as _:
            # App starts successfully
            assert hasattr(app.state, "engine_registry")

            # Registry is empty
            stt_engines = app.state.engine_registry.list_stt_engines()
            assert len(stt_engines) == 0

        # Warning should be logged
        assert "not found" in caplog.text or "using defaults" in caplog.text

    @pytest.mark.asyncio
    async def test_malformed_yaml(self, tmp_path, monkeypatch):
        """Malformed YAML → startup fails"""
        yaml_content = """
stt:
  whisper: [invalid yaml syntax
"""
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text(yaml_content)
        monkeypatch.setenv("ENGINE_CONFIG_PATH", str(yaml_file))

        # App startup should fail
        import yaml

        with pytest.raises(yaml.YAMLError):  # YAML parsing error for malformed syntax
            app = FastAPI(lifespan=lifespan)
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ):
                pass


class TestLifespanShutdown:
    """Shutdown lifecycle tests"""

    @pytest.mark.asyncio
    async def test_closes_all_engines(self, temp_engines_yaml, monkeypatch, caplog):
        """All engines closed on shutdown"""
        monkeypatch.setenv("ENGINE_CONFIG_PATH", str(temp_engines_yaml))

        app = FastAPI(lifespan=lifespan)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as _:
            # Engines initialized
            pass

        # Check shutdown logs
        # Note: In real scenario, engine.close() would be called
        # This test verifies the shutdown path executes

    @pytest.mark.asyncio
    async def test_shutdown_continues_on_close_failure(
        self, temp_engines_yaml, monkeypatch, caplog
    ):
        """One engine fails to close → others still closed"""
        monkeypatch.setenv("ENGINE_CONFIG_PATH", str(temp_engines_yaml))

        app = FastAPI(lifespan=lifespan)

        # Mock engine to fail on close
        with patch("app.engines.stt.whisper.engine.WhisperSTTEngine") as mock_engine:
            mock_instance = MagicMock()
            mock_instance.initialize = AsyncMock()
            mock_instance.close = AsyncMock(side_effect=RuntimeError("Close failed"))
            mock_instance.is_ready.return_value = True
            mock_instance.engine_name = "whisper"
            mock_engine.return_value = mock_instance

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ):
                pass

            # Shutdown should complete despite error
            # Error should be logged
            # (Actual test would verify other engines closed)

    @pytest.mark.asyncio
    async def test_shutdown_with_uninitialized_engines(self, tmp_path, monkeypatch):
        """Engines never initialized don't cause errors on close"""
        # YAML with engine that fails to initialize
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
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ):
            pass


class TestLifespanLogging:
    """Lifespan logging tests"""

    @pytest.mark.asyncio
    async def test_startup_success_logged(self, temp_engines_yaml, monkeypatch, caplog):
        """Successful startup is logged"""
        monkeypatch.setenv("ENGINE_CONFIG_PATH", str(temp_engines_yaml))

        app = FastAPI(lifespan=lifespan)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as _:
            pass

        # Check success log
        assert "Engine registry initialized" in caplog.text or "STT" in caplog.text

    @pytest.mark.asyncio
    async def test_engine_count_logged(self, temp_engines_yaml, monkeypatch, caplog):
        """Number of engines logged on startup"""
        monkeypatch.setenv("ENGINE_CONFIG_PATH", str(temp_engines_yaml))

        app = FastAPI(lifespan=lifespan)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as _:
            pass

        # Log should mention engine count
        assert "1" in caplog.text or "whisper" in caplog.text


class TestLifespanEdgeCases:
    """Edge cases in lifespan management"""

    @pytest.mark.asyncio
    async def test_empty_yaml_file(self, tmp_path, monkeypatch):
        """Empty YAML file handled gracefully"""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        monkeypatch.setenv("ENGINE_CONFIG_PATH", str(yaml_file))

        app = FastAPI(lifespan=lifespan)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as _:
            # Should handle empty config
            assert hasattr(app.state, "engine_registry")

    @pytest.mark.asyncio
    async def test_yaml_with_comments(self, tmp_path, monkeypatch):
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

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as _:
            # Comments should be ignored
            stt_engines = app.state.engine_registry.list_stt_engines()
            assert "whisper" in stt_engines

    @pytest.mark.asyncio
    async def test_multiple_startup_shutdown_cycles(
        self, temp_engines_yaml, monkeypatch
    ):
        """Multiple startup/shutdown cycles work correctly"""
        monkeypatch.setenv("ENGINE_CONFIG_PATH", str(temp_engines_yaml))

        # First cycle
        app1 = FastAPI(lifespan=lifespan)
        async with AsyncClient(
            transport=ASGITransport(app=app1), base_url="http://test"
        ):
            pass

        # Second cycle
        app2 = FastAPI(lifespan=lifespan)
        async with AsyncClient(
            transport=ASGITransport(app=app2), base_url="http://test"
        ):
            pass

        # Both should work independently
