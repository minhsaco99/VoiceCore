"""Shared fixtures for API unit tests

Tests the generic Voice Engine API framework.
Engine mocks represent ANY STT/TTS engine implementation.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from app.api.registry import EngineRegistry
from app.api.routers import health, stt, tts
from app.engines.base import BaseSTTEngine, BaseTTSEngine
from app.models.engine import Segment, STTChunk, STTResponse, TTSChunk, TTSResponse

# ============================================================================
# Generic Mock Engines (represents ANY engine implementation)
# ============================================================================


@pytest.fixture
def mock_stt_engine():
    """Generic mock STT engine"""
    engine = MagicMock(spec=BaseSTTEngine)
    engine.is_ready.return_value = True
    engine.engine_name = "test-stt"
    engine.supported_formats = ["wav", "mp3", "flac"]

    mock_response = STTResponse(
        text="Test transcription",
        language="en",
        segments=[Segment(id=0, text="Test transcription", start=0.0, end=1.0)],
        processing_time=0.1,
    )
    engine.transcribe = AsyncMock(return_value=mock_response)

    async def mock_stream(audio, **kwargs):
        yield STTChunk(text="Test", timestamp=0.0)
        yield STTChunk(text=" transcription", timestamp=0.5)
        yield mock_response

    engine.transcribe_stream = mock_stream
    return engine


@pytest.fixture
def mock_tts_engine():
    """Generic mock TTS engine"""
    engine = MagicMock(spec=BaseTTSEngine)
    engine.is_ready.return_value = True
    engine.engine_name = "test-tts"
    engine.supported_voices = ["default", "voice2"]

    mock_response = TTSResponse(
        audio_data=b"audio",
        sample_rate=16000,
        duration_seconds=1.0,
        format="wav",
    )
    engine.synthesize = AsyncMock(return_value=mock_response)

    async def mock_stream_generator(text, **kwargs):
        yield TTSChunk(audio_data=b"chunk1", sequence_number=0)
        yield TTSChunk(audio_data=b"chunk2", sequence_number=1)
        yield mock_response

    mock_stream = MagicMock()
    mock_stream.__aiter__.side_effect = mock_stream_generator
    # Make the mock callable and return the generator when called
    mock_stream.side_effect = mock_stream_generator

    engine.synthesize_stream = mock_stream
    return engine


# ============================================================================
# Registry Fixtures
# ============================================================================


@pytest.fixture
def registry_with_stt(mock_stt_engine):
    """Registry with one STT engine"""
    registry = EngineRegistry()
    registry._stt_engines = {"default": mock_stt_engine}
    return registry


@pytest.fixture
def registry_with_both(mock_stt_engine, mock_tts_engine):
    """Registry with STT and TTS engines"""
    registry = EngineRegistry()
    registry._stt_engines = {"default": mock_stt_engine}
    registry._tts_engines = {"default": mock_tts_engine}
    return registry


@pytest.fixture
def empty_registry():
    """Empty registry"""
    return EngineRegistry()


# ============================================================================
# App Fixtures
# ============================================================================


def create_app(registry):
    """Helper to create app with given registry"""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine_registry = registry
        yield

    app = FastAPI(lifespan=lifespan)
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(stt.router, prefix="/api/v1/stt", tags=["stt"])
    app.include_router(tts.router, prefix="/api/v1/tts", tags=["tts"])
    return app


@pytest.fixture
def app_with_stt(registry_with_stt):
    """App with STT engine"""
    return create_app(registry_with_stt)


@pytest.fixture
def app_with_both(registry_with_both):
    """App with both engines"""
    return create_app(registry_with_both)


@pytest.fixture
def app_empty(empty_registry):
    """App with empty registry"""
    return create_app(empty_registry)


@pytest.fixture
def app_no_registry():
    """App with NO registry"""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield  # No registry set

    app = FastAPI(lifespan=lifespan)
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(stt.router, prefix="/api/v1/stt", tags=["stt"])
    return app


# ============================================================================
# Client Fixtures
# ============================================================================


@pytest.fixture
def client(app_with_stt):
    """TestClient with STT"""
    with TestClient(app_with_stt, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def client_both(app_with_both):
    """TestClient with both engines"""
    with TestClient(app_with_both, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def client_empty(app_empty):
    """TestClient with empty registry"""
    with TestClient(app_empty, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def client_no_registry(app_no_registry):
    """TestClient with no registry"""
    with TestClient(app_no_registry, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
async def async_client(app_with_stt):
    """Async client"""
    async with AsyncClient(
        transport=ASGITransport(app=app_with_stt), base_url="http://test"
    ) as c:
        yield c


# ============================================================================
# Test Data
# ============================================================================


@pytest.fixture
def test_audio_bytes():
    """Fake audio data"""
    return b"RIFF" + b"\x00" * 100
