"""Unit tests for dependency injection functions

Tests for FastAPI dependencies that provide engines and validate uploads.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, Request, UploadFile

from app.api.config import Settings
from app.api.deps import (
    get_engine_registry,
    get_settings,
    get_stt_engine,
    get_tts_engine,
    validate_audio_upload,
)
from app.api.registry import EngineNotFoundError


class TestGetSettings:
    """Test get_settings dependency"""

    def test_get_settings_returns_instance(self):
        """get_settings returns Settings instance"""
        settings = get_settings()

        assert isinstance(settings, Settings)
        assert settings.app_name == "Voice Engine API"

    def test_get_settings_returns_new_instance(self):
        """get_settings returns new instance each time"""
        settings1 = get_settings()
        settings2 = get_settings()

        # Different instances (not cached)
        assert settings1 is not settings2


class TestGetEngineRegistry:
    """Test get_engine_registry dependency"""

    def test_registry_available(self, mock_registry):
        """Get registry from app.state"""
        mock_request = MagicMock(spec=Request)
        mock_request.app.state.engine_registry = mock_registry

        registry = get_engine_registry(mock_request)

        assert registry == mock_registry

    def test_registry_missing_raises_503(self):
        """Missing registry raises HTTPException(503)"""
        mock_request = MagicMock(spec=Request)
        # No engine_registry attribute
        del mock_request.app.state.engine_registry

        with pytest.raises(HTTPException) as exc_info:
            get_engine_registry(mock_request)

        assert exc_info.value.status_code == 503
        assert "not initialized" in exc_info.value.detail.lower()

    def test_registry_none_raises_503(self):
        """Registry set to None raises HTTPException(503)"""
        mock_request = MagicMock(spec=Request)
        mock_request.app.state.engine_registry = None

        with pytest.raises(HTTPException) as exc_info:
            get_engine_registry(mock_request)

        assert exc_info.value.status_code == 503


class TestGetSTTEngine:
    """Test get_stt_engine dependency"""

    def test_get_engine_by_name(self, mock_registry, mock_stt_engine):
        """Get STT engine by name from registry"""
        engine = get_stt_engine(engine_name="whisper", registry=mock_registry)

        assert engine == mock_stt_engine

    def test_engine_not_found_raises_404(self, mock_registry):
        """Non-existent engine raises HTTPException(404)"""
        # Mock registry to raise EngineNotFoundError
        mock_registry.get_stt.side_effect = EngineNotFoundError("Engine not found")

        with pytest.raises(HTTPException) as exc_info:
            get_stt_engine(engine_name="nonexistent", registry=mock_registry)

        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail.lower()

    def test_engine_not_ready_raises_404(self, mock_registry, mock_stt_engine):
        """Engine not ready converted to HTTPException(404)"""
        # Mock engine as not ready
        mock_stt_engine.is_ready.return_value = False
        mock_registry.get_stt.side_effect = EngineNotFoundError("Engine not ready")

        with pytest.raises(HTTPException) as exc_info:
            get_stt_engine(engine_name="whisper", registry=mock_registry)

        assert exc_info.value.status_code == 404

    def test_get_stt_preserves_exception_chain(self, mock_registry):
        """Exception chaining is preserved"""
        original_error = EngineNotFoundError("Original error")
        mock_registry.get_stt.side_effect = original_error

        with pytest.raises(HTTPException) as exc_info:
            get_stt_engine(engine_name="test", registry=mock_registry)

        # Check exception was chained
        assert exc_info.value.__cause__ == original_error


class TestGetTTSEngine:
    """Test get_tts_engine dependency"""

    def test_get_tts_engine_by_name(self, mock_registry, mock_tts_engine):
        """Get TTS engine by name from registry"""
        mock_registry._tts_engines = {"coqui": mock_tts_engine}
        mock_registry.get_tts.return_value = mock_tts_engine

        engine = get_tts_engine(engine_name="coqui", registry=mock_registry)

        assert engine == mock_tts_engine

    def test_tts_engine_not_found_raises_404(self, mock_registry):
        """Non-existent TTS engine raises HTTPException(404)"""
        mock_registry.get_tts.side_effect = EngineNotFoundError("TTS engine not found")

        with pytest.raises(HTTPException) as exc_info:
            get_tts_engine(engine_name="nonexistent", registry=mock_registry)

        assert exc_info.value.status_code == 404


class TestValidateAudioUpload:
    """Test validate_audio_upload dependency"""

    @pytest.mark.asyncio
    async def test_valid_audio(self):
        """Valid audio file passes"""
        mock_file = MagicMock(spec=UploadFile)
        mock_file.read = AsyncMock(return_value=b"audio data")
        mock_file.filename = "test.wav"

        settings = Settings()

        audio_bytes = await validate_audio_upload(mock_file, settings)

        assert audio_bytes == b"audio data"

    @pytest.mark.asyncio
    async def test_empty_file_raises_400(self):
        """Empty file raises HTTPException(400)"""
        mock_file = MagicMock(spec=UploadFile)
        mock_file.read = AsyncMock(return_value=b"")
        mock_file.filename = "empty.wav"

        settings = Settings()

        with pytest.raises(HTTPException) as exc_info:
            await validate_audio_upload(mock_file, settings)

        assert exc_info.value.status_code == 400
        assert "empty" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_oversized_file_raises_413(self):
        """File exceeds max_audio_size_mb raises HTTPException(413)"""
        # Create file larger than default 25MB
        large_audio = b"x" * (26 * 1024 * 1024)  # 26MB

        mock_file = MagicMock(spec=UploadFile)
        mock_file.read = AsyncMock(return_value=large_audio)
        mock_file.filename = "large.wav"

        settings = Settings()

        with pytest.raises(HTTPException) as exc_info:
            await validate_audio_upload(mock_file, settings)

        assert exc_info.value.status_code == 413
        assert "too large" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_file_at_limit(self):
        """File exactly at limit is allowed"""
        # Create file exactly 25MB (default limit)
        audio_at_limit = b"x" * (25 * 1024 * 1024)

        mock_file = MagicMock(spec=UploadFile)
        mock_file.read = AsyncMock(return_value=audio_at_limit)
        mock_file.filename = "at_limit.wav"

        settings = Settings()

        audio_bytes = await validate_audio_upload(mock_file, settings)

        assert len(audio_bytes) == 25 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_file_just_under_limit(self):
        """File just under limit is allowed"""
        # Create file 1 byte under limit
        audio_under = b"x" * (25 * 1024 * 1024 - 1)

        mock_file = MagicMock(spec=UploadFile)
        mock_file.read = AsyncMock(return_value=audio_under)
        mock_file.filename = "under.wav"

        settings = Settings()

        audio_bytes = await validate_audio_upload(mock_file, settings)

        assert len(audio_bytes) == 25 * 1024 * 1024 - 1

    @pytest.mark.asyncio
    async def test_file_just_over_limit(self):
        """File just over limit raises 413"""
        # Create file 1 byte over limit
        audio_over = b"x" * (25 * 1024 * 1024 + 1)

        mock_file = MagicMock(spec=UploadFile)
        mock_file.read = AsyncMock(return_value=audio_over)
        mock_file.filename = "over.wav"

        settings = Settings()

        with pytest.raises(HTTPException) as exc_info:
            await validate_audio_upload(mock_file, settings)

        assert exc_info.value.status_code == 413

    @pytest.mark.asyncio
    async def test_custom_max_audio_size(self):
        """Custom max_audio_size_mb is respected"""
        # Custom limit of 10MB
        settings = Settings(max_audio_size_mb=10)

        # File of 11MB (over custom limit)
        large_audio = b"x" * (11 * 1024 * 1024)

        mock_file = MagicMock(spec=UploadFile)
        mock_file.read = AsyncMock(return_value=large_audio)
        mock_file.filename = "over_custom.wav"

        with pytest.raises(HTTPException) as exc_info:
            await validate_audio_upload(mock_file, settings)

        assert exc_info.value.status_code == 413
        assert "10" in exc_info.value.detail  # Should mention the limit

    @pytest.mark.asyncio
    async def test_file_read_error(self):
        """File read error is propagated"""
        mock_file = MagicMock(spec=UploadFile)
        mock_file.read = AsyncMock(side_effect=OSError("Read failed"))
        mock_file.filename = "error.wav"

        settings = Settings()

        with pytest.raises(IOError, match="Read failed"):
            await validate_audio_upload(mock_file, settings)


class TestDependencyEdgeCases:
    """Edge cases for dependencies"""

    def test_get_registry_with_invalid_request(self):
        """Invalid request object raises error"""
        invalid_request = MagicMock()
        # Missing app attribute
        del invalid_request.app

        with pytest.raises(AttributeError):
            get_engine_registry(invalid_request)

    @pytest.mark.asyncio
    async def test_validate_audio_with_none_file(self):
        """None file raises error"""
        settings = Settings()

        with pytest.raises(AttributeError):
            await validate_audio_upload(None, settings)

    @pytest.mark.asyncio
    async def test_validate_audio_zero_limit(self):
        """Zero size limit rejects all files"""
        settings = Settings(max_audio_size_mb=0)

        mock_file = MagicMock(spec=UploadFile)
        mock_file.read = AsyncMock(return_value=b"x")  # Even 1 byte
        mock_file.filename = "tiny.wav"

        with pytest.raises(HTTPException) as exc_info:
            await validate_audio_upload(mock_file, settings)

        assert exc_info.value.status_code == 413

    def test_get_stt_engine_empty_name(self, mock_registry):
        """Empty engine name still calls registry"""
        mock_registry.get_stt.side_effect = EngineNotFoundError("Not found")

        with pytest.raises(HTTPException):
            get_stt_engine(engine_name="", registry=mock_registry)

        mock_registry.get_stt.assert_called_once_with("")
