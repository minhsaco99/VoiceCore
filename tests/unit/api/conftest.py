"""Shared fixtures for API unit tests"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI

from app.api.registry import EngineRegistry
from app.engines.base import BaseSTTEngine, BaseTTSEngine
from app.models.engine import Segment, STTChunk, STTResponse


@pytest.fixture
def mock_stt_engine():
    """Mock STT engine with default behaviors"""
    engine = MagicMock(spec=BaseSTTEngine)
    engine.is_ready.return_value = True
    engine.engine_name = "mock-stt"
    engine.supported_formats = ["wav", "mp3"]

    # Mock transcribe response
    mock_response = STTResponse(
        text="Test transcription",
        language="en",
        segments=[
            Segment(
                id=0,
                text="Test transcription",
                start=0.0,
                end=1.0,
            )
        ],
        processing_time=0.5,
    )
    engine.transcribe = AsyncMock(return_value=mock_response)

    # Mock transcribe_stream generator
    async def mock_stream(audio, **kwargs):
        yield STTChunk(text="Test", timestamp=0.0)
        yield STTChunk(text=" transcription", timestamp=0.5)
        yield mock_response

    engine.transcribe_stream = mock_stream

    return engine


@pytest.fixture
def mock_tts_engine():
    """Mock TTS engine with default behaviors"""
    engine = MagicMock(spec=BaseTTSEngine)
    engine.is_ready.return_value = True
    engine.engine_name = "mock-tts"
    engine.supported_voices = ["voice1", "voice2"]
    engine.synthesize = AsyncMock(return_value=MagicMock(audio_data=b"audio"))
    return engine


@pytest.fixture
def mock_registry(mock_stt_engine):
    """Populated registry with mock engines"""
    registry = EngineRegistry()
    registry._stt_engines = {"whisper": mock_stt_engine}
    registry._tts_engines = {}
    return registry


@pytest.fixture
def mock_app_state(mock_registry):
    """Mock FastAPI app.state with registry"""
    mock_app = MagicMock(spec=FastAPI)
    mock_app.state.engine_registry = mock_registry
    return mock_app


@pytest.fixture
def test_audio_bytes():
    """Small test audio data"""
    return b"fake audio data for testing"
