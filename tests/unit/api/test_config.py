"""Unit tests for Settings configuration

Tests environment variable overrides and validation.
"""

import pytest
from pydantic import ValidationError

from app.api.config import Settings


class TestSettingsDefaults:
    """Test default values"""

    def test_default_values(self):
        """Settings load with default values"""
        settings = Settings()

        assert settings.app_name == "Voice Engine API"
        assert settings.version == "1.0.0"
        assert settings.debug is False
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.cors_origins == ["*"]
        assert settings.engine_config_path == "engines.yaml"
        assert settings.max_audio_size_mb == 25
        assert settings.request_timeout_seconds == 300

    def test_cors_origins_default_list(self):
        """CORS origins is a list by default"""
        settings = Settings()

        assert isinstance(settings.cors_origins, list)
        assert len(settings.cors_origins) == 1
        assert settings.cors_origins[0] == "*"


class TestSettingsEnvOverride:
    """Test environment variable overrides"""

    def test_env_override_app_name(self, monkeypatch):
        """Environment variables override defaults"""
        monkeypatch.setenv("APP_NAME", "Custom API")
        monkeypatch.setenv("VERSION", "2.0.0")
        monkeypatch.setenv("DEBUG", "true")

        settings = Settings()

        assert settings.app_name == "Custom API"
        assert settings.version == "2.0.0"
        assert settings.debug is True

    def test_env_override_server_config(self, monkeypatch):
        """Server config can be overridden"""
        monkeypatch.setenv("HOST", "127.0.0.1")
        monkeypatch.setenv("PORT", "9000")

        settings = Settings()

        assert settings.host == "127.0.0.1"
        assert settings.port == 9000

    def test_env_override_cors_origins(self, monkeypatch):
        """CORS origins can be overridden"""
        # pydantic-settings parses JSON for list fields
        monkeypatch.setenv("CORS_ORIGINS", '["http://localhost:3000"]')

        settings = Settings()

        assert settings.cors_origins == ["http://localhost:3000"]

    def test_env_override_cors_multiple_origins(self, monkeypatch):
        """Multiple CORS origins"""
        monkeypatch.setenv(
            "CORS_ORIGINS", '["http://localhost:3000", "http://localhost:8080"]'
        )

        settings = Settings()

        assert len(settings.cors_origins) == 2
        assert "http://localhost:3000" in settings.cors_origins
        assert "http://localhost:8080" in settings.cors_origins

    def test_env_override_limits(self, monkeypatch):
        """Audio size and timeout limits can be overridden"""
        monkeypatch.setenv("MAX_AUDIO_SIZE_MB", "50")
        monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "600")

        settings = Settings()

        assert settings.max_audio_size_mb == 50
        assert settings.request_timeout_seconds == 600

    def test_env_override_engine_config_path(self, monkeypatch):
        """Engine config path can be overridden"""
        monkeypatch.setenv("ENGINE_CONFIG_PATH", "/custom/path/engines.yaml")

        settings = Settings()

        assert settings.engine_config_path == "/custom/path/engines.yaml"


class TestSettingsValidation:
    """Test settings validation"""

    def test_invalid_port_negative(self, monkeypatch):
        """Invalid port raises validation error"""
        monkeypatch.setenv("PORT", "-1")

        with pytest.raises(ValidationError):
            Settings()

    def test_invalid_port_too_large(self, monkeypatch):
        """Port too large raises validation error"""
        monkeypatch.setenv("PORT", "99999")

        with pytest.raises(ValidationError):
            Settings()

    def test_invalid_port_not_number(self, monkeypatch):
        """Port must be a number"""
        monkeypatch.setenv("PORT", "not_a_number")

        with pytest.raises(ValidationError):
            Settings()

    def test_max_audio_size_zero(self, monkeypatch):
        """Max audio size of zero is technically valid"""
        monkeypatch.setenv("MAX_AUDIO_SIZE_MB", "0")

        # pydantic allows 0 by default
        settings = Settings()
        assert settings.max_audio_size_mb == 0

    def test_max_audio_size_negative(self, monkeypatch):
        """Negative max audio size should be invalid"""
        monkeypatch.setenv("MAX_AUDIO_SIZE_MB", "-1")

        with pytest.raises(ValidationError):
            Settings()

    def test_request_timeout_zero(self, monkeypatch):
        """Zero timeout is technically valid"""
        monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "0")

        settings = Settings()
        assert settings.request_timeout_seconds == 0

    def test_request_timeout_negative(self, monkeypatch):
        """Negative timeout should be invalid"""
        monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "-100")

        with pytest.raises(ValidationError):
            Settings()


class TestSettingsCORSParsing:
    """CORS origins parsing edge cases"""

    def test_cors_empty_list(self, monkeypatch):
        """Empty CORS list is valid"""
        monkeypatch.setenv("CORS_ORIGINS", "[]")

        settings = Settings()

        assert settings.cors_origins == []

    def test_cors_wildcard(self, monkeypatch):
        """Wildcard CORS origin"""
        monkeypatch.setenv("CORS_ORIGINS", '["*"]')

        settings = Settings()

        assert settings.cors_origins == ["*"]

    def test_cors_invalid_json(self, monkeypatch):
        """Invalid JSON for CORS raises error"""
        monkeypatch.setenv("CORS_ORIGINS", "not valid json")

        with pytest.raises(ValidationError):
            Settings()

    def test_cors_not_list(self, monkeypatch):
        """CORS_ORIGINS must be a list"""
        monkeypatch.setenv("CORS_ORIGINS", '"http://localhost:3000"')  # String not list

        with pytest.raises(ValidationError):
            Settings()


class TestSettingsDebugMode:
    """Debug mode toggle tests"""

    def test_debug_false_by_default(self):
        """Debug is False by default"""
        settings = Settings()
        assert settings.debug is False

    def test_debug_true(self, monkeypatch):
        """Debug can be enabled"""
        monkeypatch.setenv("DEBUG", "true")

        settings = Settings()
        assert settings.debug is True

    def test_debug_various_truthy_values(self, monkeypatch):
        """Various truthy values for debug"""
        for value in ["1", "yes", "True", "TRUE"]:
            monkeypatch.setenv("DEBUG", value)
            settings = Settings()
            assert settings.debug is True

    def test_debug_various_falsy_values(self, monkeypatch):
        """Various falsy values for debug"""
        for value in ["0", "no", "False", "FALSE", "false"]:
            monkeypatch.setenv("DEBUG", value)
            settings = Settings()
            assert settings.debug is False


class TestSettingsImmutability:
    """Settings behavior tests"""

    def test_settings_instance_created(self):
        """Settings instance can be created multiple times"""
        settings1 = Settings()
        settings2 = Settings()

        # Different instances
        assert settings1 is not settings2
        # Same values
        assert settings1.port == settings2.port

    def test_settings_values_can_be_accessed(self):
        """All settings values are accessible"""
        settings = Settings()

        # Should not raise AttributeError
        _ = settings.app_name
        _ = settings.version
        _ = settings.debug
        _ = settings.host
        _ = settings.port
        _ = settings.cors_origins
        _ = settings.engine_config_path
        _ = settings.max_audio_size_mb
        _ = settings.request_timeout_seconds


class TestSettingsEdgeCases:
    """Edge cases and boundary conditions"""

    def test_port_at_min_valid(self, monkeypatch):
        """Port 1 is valid"""
        monkeypatch.setenv("PORT", "1")

        settings = Settings()
        assert settings.port == 1

    def test_port_at_max_valid(self, monkeypatch):
        """Port 65535 is valid"""
        monkeypatch.setenv("PORT", "65535")

        settings = Settings()
        assert settings.port == 65535

    def test_max_audio_size_large_value(self, monkeypatch):
        """Large max audio size is valid"""
        monkeypatch.setenv("MAX_AUDIO_SIZE_MB", "1000")

        settings = Settings()
        assert settings.max_audio_size_mb == 1000

    def test_empty_engine_config_path(self, monkeypatch):
        """Empty engine config path is technically valid"""
        monkeypatch.setenv("ENGINE_CONFIG_PATH", "")

        settings = Settings()
        assert settings.engine_config_path == ""

    def test_very_long_app_name(self, monkeypatch):
        """Very long app name is valid"""
        long_name = "A" * 1000
        monkeypatch.setenv("APP_NAME", long_name)

        settings = Settings()
        assert settings.app_name == long_name
        assert len(settings.app_name) == 1000
