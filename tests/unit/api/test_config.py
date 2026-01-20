"""Unit tests for Settings configuration"""

import pytest
from pydantic import ValidationError

from app.api.config import Settings


class TestSettingsBasic:
    """Basic Settings functionality"""

    def test_creates_instance(self):
        """Can create Settings instance"""
        settings = Settings()
        assert settings is not None

    def test_has_required_attributes(self):
        """Has all required attributes"""
        settings = Settings()

        assert hasattr(settings, "app_name")
        assert hasattr(settings, "version")
        assert hasattr(settings, "debug")
        assert hasattr(settings, "host")
        assert hasattr(settings, "port")
        assert hasattr(settings, "cors_origins")
        assert hasattr(settings, "engine_config_path")
        assert hasattr(settings, "max_audio_size_mb")
        assert hasattr(settings, "request_timeout_seconds")

    def test_types_are_correct(self):
        """Attributes have correct types"""
        settings = Settings()

        assert isinstance(settings.app_name, str)
        assert isinstance(settings.version, str)
        assert isinstance(settings.debug, bool)
        assert isinstance(settings.host, str)
        assert isinstance(settings.port, int)
        assert isinstance(settings.cors_origins, list)
        assert isinstance(settings.engine_config_path, str)
        assert isinstance(settings.max_audio_size_mb, int)
        assert isinstance(settings.request_timeout_seconds, int)


class TestSettingsEnvOverride:
    """Environment variable overrides"""

    def test_overrides_app_name(self, monkeypatch):
        """APP_NAME env var overrides default"""
        monkeypatch.setenv("APP_NAME", "Custom API")

        settings = Settings()

        assert settings.app_name == "Custom API"

    def test_overrides_version(self, monkeypatch):
        """VERSION env var overrides default"""
        monkeypatch.setenv("VERSION", "2.0.0")

        settings = Settings()

        assert settings.version == "2.0.0"

    def test_overrides_debug(self, monkeypatch):
        """DEBUG env var overrides default"""
        monkeypatch.setenv("DEBUG", "true")

        settings = Settings()

        assert settings.debug is True

    def test_overrides_host(self, monkeypatch):
        """HOST env var overrides default"""
        monkeypatch.setenv("HOST", "127.0.0.1")

        settings = Settings()

        assert settings.host == "127.0.0.1"

    def test_overrides_port(self, monkeypatch):
        """PORT env var overrides default"""
        monkeypatch.setenv("PORT", "9000")

        settings = Settings()

        assert settings.port == 9000

    def test_overrides_cors_origins(self, monkeypatch):
        """CORS_ORIGINS env var overrides default"""
        monkeypatch.setenv("CORS_ORIGINS", '["http://example.com"]')

        settings = Settings()

        assert settings.cors_origins == ["http://example.com"]

    def test_overrides_max_audio_size(self, monkeypatch):
        """MAX_AUDIO_SIZE_MB env var overrides default"""
        monkeypatch.setenv("MAX_AUDIO_SIZE_MB", "50")

        settings = Settings()

        assert settings.max_audio_size_mb == 50

    def test_overrides_request_timeout(self, monkeypatch):
        """REQUEST_TIMEOUT_SECONDS env var overrides default"""
        monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "600")

        settings = Settings()

        assert settings.request_timeout_seconds == 600

    def test_overrides_engine_config_path(self, monkeypatch):
        """ENGINE_CONFIG_PATH env var overrides default"""
        monkeypatch.setenv("ENGINE_CONFIG_PATH", "/custom/path.yaml")

        settings = Settings()

        assert settings.engine_config_path == "/custom/path.yaml"


class TestSettingsValidation:
    """Settings validation"""

    def test_port_must_be_integer(self, monkeypatch):
        """PORT must be valid integer"""
        monkeypatch.setenv("PORT", "not_a_number")

        with pytest.raises(ValidationError):
            Settings()

    def test_max_audio_size_must_be_integer(self, monkeypatch):
        """MAX_AUDIO_SIZE_MB must be valid integer"""
        monkeypatch.setenv("MAX_AUDIO_SIZE_MB", "not_a_number")

        with pytest.raises(ValidationError):
            Settings()

    def test_cors_origins_must_be_list(self, monkeypatch):
        """CORS_ORIGINS must be valid JSON list"""
        monkeypatch.setenv("CORS_ORIGINS", '"not_a_list"')

        with pytest.raises(ValidationError):
            Settings()

    def test_debug_accepts_various_truthy(self, monkeypatch):
        """DEBUG accepts various truthy values"""
        for value in ["1", "true", "True", "yes"]:
            monkeypatch.setenv("DEBUG", value)
            settings = Settings()
            assert settings.debug is True

    def test_debug_accepts_various_falsy(self, monkeypatch):
        """DEBUG accepts various falsy values"""
        for value in ["0", "false", "False", "no"]:
            monkeypatch.setenv("DEBUG", value)
            settings = Settings()
            assert settings.debug is False


class TestSettingsCORSOrigins:
    """CORS origins parsing"""

    def test_accepts_empty_list(self, monkeypatch):
        """Accepts empty CORS origins list"""
        monkeypatch.setenv("CORS_ORIGINS", "[]")

        settings = Settings()

        assert settings.cors_origins == []

    def test_accepts_multiple_origins(self, monkeypatch):
        """Accepts multiple CORS origins"""
        monkeypatch.setenv("CORS_ORIGINS", '["http://a.com", "http://b.com"]')

        settings = Settings()

        assert len(settings.cors_origins) == 2
        assert "http://a.com" in settings.cors_origins
        assert "http://b.com" in settings.cors_origins

    def test_accepts_wildcard(self, monkeypatch):
        """Accepts wildcard CORS origin"""
        monkeypatch.setenv("CORS_ORIGINS", '["*"]')

        settings = Settings()

        assert settings.cors_origins == ["*"]


class TestSettingsEdgeCases:
    """Edge cases"""

    def test_port_at_min(self, monkeypatch):
        """Accepts port 1"""
        monkeypatch.setenv("PORT", "1")

        settings = Settings()

        assert settings.port == 1

    def test_port_at_max(self, monkeypatch):
        """Accepts port 65535"""
        monkeypatch.setenv("PORT", "65535")

        settings = Settings()

        assert settings.port == 65535

    def test_large_max_audio_size(self, monkeypatch):
        """Accepts large max audio size"""
        monkeypatch.setenv("MAX_AUDIO_SIZE_MB", "1000")

        settings = Settings()

        assert settings.max_audio_size_mb == 1000

    def test_zero_max_audio_size(self, monkeypatch):
        """Accepts zero max audio size"""
        monkeypatch.setenv("MAX_AUDIO_SIZE_MB", "0")

        settings = Settings()

        assert settings.max_audio_size_mb == 0

    def test_empty_engine_config_path(self, monkeypatch):
        """Accepts empty engine config path"""
        monkeypatch.setenv("ENGINE_CONFIG_PATH", "")

        settings = Settings()

        assert settings.engine_config_path == ""

    def test_long_app_name(self, monkeypatch):
        """Accepts long app name"""
        long_name = "A" * 500
        monkeypatch.setenv("APP_NAME", long_name)

        settings = Settings()

        assert settings.app_name == long_name
