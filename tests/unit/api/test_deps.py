"""Unit tests for dependency injection functions"""

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
from app.api.registry import EngineRegistry


class TestGetSettings:
    """Test get_settings dependency"""

    def test_returns_settings_instance(self):
        """Returns Settings instance"""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_returns_new_instance_each_call(self):
        """Returns new instance each time"""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is not s2


class TestGetEngineRegistry:
    """Test get_engine_registry dependency"""

    def test_returns_registry_from_app_state(self, registry_with_stt):
        """Returns registry from app.state"""
        mock_request = MagicMock(spec=Request)
        mock_request.app.state.engine_registry = registry_with_stt

        result = get_engine_registry(mock_request)

        assert result is registry_with_stt

    def test_raises_503_when_registry_is_none(self):
        """Raises 503 when registry is None"""
        mock_request = MagicMock(spec=Request)
        mock_request.app.state.engine_registry = None

        with pytest.raises(HTTPException) as exc:
            get_engine_registry(mock_request)

        assert exc.value.status_code == 503

    def test_raises_503_when_registry_missing(self):
        """Raises 503 when registry attribute missing"""
        mock_request = MagicMock(spec=Request)
        mock_request.app.state = MagicMock(spec=[])  # No engine_registry attr

        with pytest.raises(HTTPException) as exc:
            get_engine_registry(mock_request)

        assert exc.value.status_code == 503


class TestGetSTTEngine:
    """Test get_stt_engine dependency"""

    def test_returns_engine_by_name(self, registry_with_stt, mock_stt_engine):
        """Returns engine by name"""
        engine = get_stt_engine(engine="default", registry=registry_with_stt)
        assert engine is mock_stt_engine

    def test_raises_404_when_engine_not_found(self, empty_registry):
        """Raises 404 when engine not found"""
        with pytest.raises(HTTPException) as exc:
            get_stt_engine(engine="nonexistent", registry=empty_registry)

        assert exc.value.status_code == 404
        assert "not found" in exc.value.detail.lower()

    def test_raises_404_when_engine_not_ready(self, mock_stt_engine):
        """Raises 404 when engine not ready"""
        mock_stt_engine.is_ready.return_value = False
        registry = EngineRegistry()
        registry._stt_engines = {"default": mock_stt_engine}

        with pytest.raises(HTTPException) as exc:
            get_stt_engine(engine="default", registry=registry)

        assert exc.value.status_code == 404
        assert "not ready" in exc.value.detail.lower()

    def test_exception_chaining(self, empty_registry):
        """Exception is chained from original"""
        with pytest.raises(HTTPException) as exc:
            get_stt_engine(engine="missing", registry=empty_registry)

        assert exc.value.__cause__ is not None


class TestGetTTSEngine:
    """Test get_tts_engine dependency"""

    def test_returns_engine_by_name(self, registry_with_both, mock_tts_engine):
        """Returns TTS engine by name"""
        engine = get_tts_engine(engine="default", registry=registry_with_both)
        assert engine is mock_tts_engine

    def test_raises_404_when_not_found(self, empty_registry):
        """Raises 404 when TTS engine not found"""
        with pytest.raises(HTTPException) as exc:
            get_tts_engine(engine="missing", registry=empty_registry)

        assert exc.value.status_code == 404


class TestValidateAudioUpload:
    """Test validate_audio_upload dependency"""

    @pytest.mark.asyncio
    async def test_valid_audio_returns_bytes(self):
        """Valid audio returns bytes"""
        mock_file = MagicMock(spec=UploadFile)
        mock_file.read = AsyncMock(return_value=b"audio data")

        result = await validate_audio_upload(mock_file, Settings())

        assert result == b"audio data"

    @pytest.mark.asyncio
    async def test_raises_400_for_empty_file(self):
        """Raises 400 for empty file"""
        mock_file = MagicMock(spec=UploadFile)
        mock_file.read = AsyncMock(return_value=b"")

        with pytest.raises(HTTPException) as exc:
            await validate_audio_upload(mock_file, Settings())

        assert exc.value.status_code == 400
        assert "empty" in exc.value.detail.lower()

    @pytest.mark.asyncio
    async def test_raises_413_for_oversized_file(self):
        """Raises 413 for oversized file"""
        large_audio = b"x" * (26 * 1024 * 1024)  # 26MB > 25MB default
        mock_file = MagicMock(spec=UploadFile)
        mock_file.read = AsyncMock(return_value=large_audio)

        with pytest.raises(HTTPException) as exc:
            await validate_audio_upload(mock_file, Settings())

        assert exc.value.status_code == 413

    @pytest.mark.asyncio
    async def test_accepts_file_at_limit(self):
        """Accepts file exactly at limit"""
        at_limit = b"x" * (25 * 1024 * 1024)
        mock_file = MagicMock(spec=UploadFile)
        mock_file.read = AsyncMock(return_value=at_limit)

        result = await validate_audio_upload(mock_file, Settings())

        assert len(result) == 25 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_respects_custom_size_limit(self):
        """Respects custom max_audio_size_mb"""
        audio = b"x" * (11 * 1024 * 1024)  # 11MB
        mock_file = MagicMock(spec=UploadFile)
        mock_file.read = AsyncMock(return_value=audio)

        with pytest.raises(HTTPException) as exc:
            await validate_audio_upload(mock_file, Settings(max_audio_size_mb=10))

        assert exc.value.status_code == 413

    @pytest.mark.asyncio
    async def test_propagates_read_error(self):
        """Propagates file read errors"""
        mock_file = MagicMock(spec=UploadFile)
        mock_file.read = AsyncMock(side_effect=OSError("Read failed"))

        with pytest.raises(OSError):
            await validate_audio_upload(mock_file, Settings())
