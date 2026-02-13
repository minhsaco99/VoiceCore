"""
Unit tests for CosyVoice3 TTS Engine (HTTP Client)

Tests config, lifecycle, synthesize (batch), streaming, and properties
with a mocked HTTP client (no real CosyVoice3 server needed).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.engines.tts.cosyvoice3.config import CosyVoice3Config, VoiceConfig
from app.engines.tts.cosyvoice3.engine import CosyVoice3Engine
from app.exceptions import EngineNotReadyError, SynthesisError
from app.models.engine import TTSChunk, TTSResponse

# =============================================================================
# Fixtures
# =============================================================================


SAMPLE_RATE = 22050


def _pcm_bytes(duration_seconds: float = 1.0) -> bytes:
    """Generate dummy PCM int16 bytes for testing."""
    samples = int(SAMPLE_RATE * duration_seconds)
    audio = np.zeros(samples, dtype=np.int16)
    return audio.tobytes()


@pytest.fixture
def config():
    """Default CosyVoice3 config for testing."""
    return CosyVoice3Config(
        model_name="Fun-CosyVoice3-0.5B",
        service_url="http://localhost:50000",
        sample_rate=SAMPLE_RATE,
        default_voice="test_voice",
        voices={
            "test_voice": VoiceConfig(
                prompt_wav_path="/tmp/test_voice.wav",
                prompt_text="This is a test voice sample.",
            ),
            "another_voice": VoiceConfig(
                prompt_wav_path="/tmp/another_voice.wav",
                prompt_text="Another voice sample.",
            ),
        },
    )


@pytest.fixture
def config_no_voices():
    """Config without any voices configured."""
    return CosyVoice3Config(
        model_name="Fun-CosyVoice3-0.5B",
    )


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx.AsyncClient."""
    client = AsyncMock()
    # Mock the GET /docs health check
    mock_response = MagicMock()
    mock_response.status_code = 200
    client.get = AsyncMock(return_value=mock_response)
    return client


# =============================================================================
# Config Tests
# =============================================================================


class TestCosyVoice3Config:
    """Test CosyVoice3Config validation and defaults."""

    def test_config_defaults(self):
        """Config should have correct default values."""
        config = CosyVoice3Config(model_name="test-model")

        assert config.model_name == "test-model"
        assert config.service_url == "http://localhost:50000"
        assert config.sample_rate == 22050
        assert config.default_voice is None
        assert config.voices == {}
        assert config.speed == 1.0
        assert config.system_prompt == "You are a helpful assistant."
        assert config.connect_timeout == 10.0
        assert config.read_timeout is None

    def test_config_custom_values(self):
        """Config should accept custom values."""
        config = CosyVoice3Config(
            model_name="custom-model",
            service_url="http://tts-server:8080",
            sample_rate=16000,
            speed=1.5,
            default_voice="my_voice",
            voices={
                "my_voice": VoiceConfig(
                    prompt_wav_path="/data/voice.wav",
                    prompt_text="Hello world",
                )
            },
        )

        assert config.service_url == "http://tts-server:8080"
        assert config.sample_rate == 16000
        assert config.speed == 1.5
        assert config.default_voice == "my_voice"
        assert "my_voice" in config.voices

    def test_voice_config(self):
        """VoiceConfig should store prompt wav path and text."""
        voice = VoiceConfig(
            prompt_wav_path="/data/sample.wav",
            prompt_text="Sample transcript",
            description="A test voice",
        )

        assert voice.prompt_wav_path == "/data/sample.wav"
        assert voice.prompt_text == "Sample transcript"
        assert voice.description == "A test voice"


# =============================================================================
# Engine Lifecycle Tests
# =============================================================================


class TestCosyVoice3EngineLifecycle:
    """Test engine initialization and cleanup."""

    @pytest.mark.asyncio
    async def test_engine_initialization(self, config, mock_httpx_client):
        """Engine should create HTTP client on initialization."""
        engine = CosyVoice3Engine(config)

        # Mock httpx in sys.modules so the import in _initialize gets our mock
        with patch.dict("sys.modules", {"httpx": mock_httpx_client}):
            # The client itself is used as the module mock, which works if we setup structure right
            # But better: create a module mock
            mock_module = MagicMock()
            mock_module.AsyncClient.return_value = mock_httpx_client
            mock_module.Timeout = MagicMock()

            with patch.dict("sys.modules", {"httpx": mock_module}):
                await engine.initialize()

                assert engine.is_ready()
                mock_module.AsyncClient.assert_called_once()

    @pytest.mark.asyncio
    async def test_engine_cleanup(self, config, mock_httpx_client):
        """Engine should close HTTP client on cleanup."""
        engine = CosyVoice3Engine(config)
        engine._client = mock_httpx_client
        engine._initialized = True

        await engine.close()

        mock_httpx_client.aclose.assert_called_once()
        assert engine._client is None
        assert not engine.is_ready()

    @pytest.mark.asyncio
    async def test_engine_raises_on_missing_httpx(self, config):
        """Engine should raise EngineNotReadyError if httpx not installed."""
        engine = CosyVoice3Engine(config)

        with patch.dict("sys.modules", {"httpx": None}):
            with pytest.raises(EngineNotReadyError) as exc_info:
                await engine.initialize()

            assert "httpx" in str(exc_info.value).lower()


# =============================================================================
# Voice Resolution Tests
# =============================================================================


class TestVoiceResolution:
    """Test voice resolution logic."""

    def test_resolve_voice_by_name(self, config):
        """Should resolve voice from config.voices map."""
        engine = CosyVoice3Engine(config)

        wav_path, prompt_text, ref_bytes = engine._resolve_voice(
            "test_voice", None, None
        )

        assert wav_path == "/tmp/test_voice.wav"
        assert prompt_text == "This is a test voice sample."
        assert ref_bytes is None

    def test_resolve_default_voice(self, config):
        """Should use default_voice when voice is None."""
        engine = CosyVoice3Engine(config)

        wav_path, prompt_text, _ = engine._resolve_voice(None, None, None)

        assert wav_path == "/tmp/test_voice.wav"
        assert prompt_text == "This is a test voice sample."

    def test_resolve_reference_audio_priority(self, config):
        """Reference audio should take priority over voice name."""
        engine = CosyVoice3Engine(config)
        ref_audio = b"fake audio bytes"

        wav_path, prompt_text, ref_bytes = engine._resolve_voice(
            "test_voice", ref_audio, "custom prompt text"
        )

        assert wav_path is None
        assert prompt_text == "custom prompt text"
        assert ref_bytes == ref_audio

    def test_resolve_voice_no_default(self, config_no_voices):
        """Should raise SynthesisError if no voice and no default."""
        engine = CosyVoice3Engine(config_no_voices)

        with pytest.raises(SynthesisError) as exc_info:
            engine._resolve_voice(None, None, None)

        assert "No voice specified" in str(exc_info.value)

    def test_resolve_unknown_voice(self, config):
        """Should raise SynthesisError for unknown voice name."""
        engine = CosyVoice3Engine(config)

        with pytest.raises(SynthesisError) as exc_info:
            engine._resolve_voice("nonexistent", None, None)

        assert "not found" in str(exc_info.value)


# =============================================================================
# Prompt Text Tests
# =============================================================================


class TestPromptText:
    """Test prompt text preparation."""

    def test_auto_prepend_system_prompt(self, config):
        """Should auto-prepend system_prompt + <|endofprompt|>."""
        engine = CosyVoice3Engine(config)

        result = engine._prepare_prompt_text("Hello world")

        assert "<|endofprompt|>" in result
        assert result.startswith("You are a helpful assistant.")
        assert result.endswith("Hello world")

    def test_preserve_existing_endofprompt(self, config):
        """Should not modify text that already has <|endofprompt|>."""
        engine = CosyVoice3Engine(config)
        text = "Custom prompt<|endofprompt|>Hello"

        result = engine._prepare_prompt_text(text)

        assert result == text

    def test_empty_prompt_text(self, config):
        """Should return empty string for empty prompt."""
        engine = CosyVoice3Engine(config)
        assert engine._prepare_prompt_text("") == ""
        assert engine._prepare_prompt_text(None) == ""


# =============================================================================
# Synthesize Tests
# =============================================================================


class TestCosyVoice3Synthesize:
    """Test batch synthesis."""

    @pytest.mark.asyncio
    async def test_synthesize_returns_audio(self, config, mock_httpx_client):
        """Synthesize should return TTSResponse with audio data."""
        engine = CosyVoice3Engine(config)
        engine._client = mock_httpx_client
        engine._initialized = True

        pcm_data = _pcm_bytes(1.0)

        # Mock the streaming response
        mock_stream_response = AsyncMock()
        mock_stream_response.raise_for_status = MagicMock()
        # aiter_bytes returns an async iterator, not a coroutine
        mock_stream_response.aiter_bytes = MagicMock(
            return_value=_async_iter([pcm_data])
        )
        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_client.stream = MagicMock(return_value=mock_stream_ctx)

        with patch("pathlib.Path.open", _mock_open()):
            result = await engine.synthesize("Hello world")

        assert isinstance(result, TTSResponse)
        assert result.audio_data is not None
        assert len(result.audio_data) > 0
        assert result.sample_rate == SAMPLE_RATE
        assert result.duration_seconds > 0
        assert result.format == "wav"
        assert result.performance_metrics is not None

    @pytest.mark.asyncio
    async def test_synthesize_empty_response(self, config, mock_httpx_client):
        """Should raise SynthesisError on empty audio response."""
        engine = CosyVoice3Engine(config)
        engine._client = mock_httpx_client
        engine._initialized = True

        # Mock empty response
        mock_stream_response = AsyncMock()
        mock_stream_response.raise_for_status = MagicMock()
        mock_stream_response.aiter_bytes = MagicMock(return_value=_async_iter([]))
        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_client.stream = MagicMock(return_value=mock_stream_ctx)

        with patch("pathlib.Path.open", _mock_open()):
            with pytest.raises(SynthesisError) as exc_info:
                await engine.synthesize("Test")

            assert "empty" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_synthesize_client_not_ready(self, config):
        """Should raise EngineNotReadyError if client is None."""
        engine = CosyVoice3Engine(config)
        engine._initialized = True
        engine._client = None

        with pytest.raises(EngineNotReadyError):
            await engine.synthesize("Test")


# =============================================================================
# Streaming Tests
# =============================================================================


class TestCosyVoice3Streaming:
    """Test streaming synthesis."""

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self, config, mock_httpx_client):
        """Streaming should yield TTSChunks then TTSResponse."""
        engine = CosyVoice3Engine(config)
        engine._client = mock_httpx_client
        engine._initialized = True

        chunk1 = _pcm_bytes(0.25)
        chunk2 = _pcm_bytes(0.25)

        mock_stream_response = AsyncMock()
        mock_stream_response.raise_for_status = MagicMock()
        mock_stream_response.aiter_bytes = MagicMock(
            return_value=_async_iter([chunk1, chunk2])
        )
        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_client.stream = MagicMock(return_value=mock_stream_ctx)

        results = []
        with patch("pathlib.Path.open", _mock_open()):
            async for item in engine.synthesize_stream("Hello world"):
                results.append(item)

        # Should have 2 chunks + 1 final response
        assert len(results) == 3

        # First two are TTSChunk
        for i, chunk in enumerate(results[:2]):
            assert isinstance(chunk, TTSChunk)
            assert chunk.audio_data is not None
            assert chunk.sequence_number == i

        # Last is TTSResponse
        final = results[-1]
        assert isinstance(final, TTSResponse)
        assert final.performance_metrics is not None
        assert final.performance_metrics.total_chunks == 2

    @pytest.mark.asyncio
    async def test_stream_metrics_include_ttfb(self, config, mock_httpx_client):
        """Streaming metrics should include time to first byte."""
        engine = CosyVoice3Engine(config)
        engine._client = mock_httpx_client
        engine._initialized = True

        mock_stream_response = AsyncMock()
        mock_stream_response.raise_for_status = MagicMock()
        mock_stream_response.aiter_bytes = MagicMock(
            return_value=_async_iter([_pcm_bytes(0.5)])
        )
        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_client.stream = MagicMock(return_value=mock_stream_ctx)

        final_response = None
        with patch("pathlib.Path.open", _mock_open()):
            async for item in engine.synthesize_stream("Test"):
                if isinstance(item, TTSResponse):
                    final_response = item

        assert final_response is not None
        assert final_response.performance_metrics.time_to_first_byte_ms is not None

    @pytest.mark.asyncio
    async def test_stream_client_not_ready(self, config):
        """Stream should raise EngineNotReadyError if client is None."""
        engine = CosyVoice3Engine(config)
        engine._initialized = True
        engine._client = None

        with pytest.raises(EngineNotReadyError):
            async for _ in engine.synthesize_stream("Test"):
                pass

    @pytest.mark.asyncio
    async def test_stream_empty_result(self, config, mock_httpx_client):
        """Stream should handle empty response gracefully."""
        engine = CosyVoice3Engine(config)
        engine._client = mock_httpx_client
        engine._initialized = True

        mock_stream_response = AsyncMock()
        mock_stream_response.raise_for_status = MagicMock()
        mock_stream_response.aiter_bytes = MagicMock(return_value=_async_iter([]))
        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_client.stream = MagicMock(return_value=mock_stream_ctx)

        results = []
        with patch("pathlib.Path.open", _mock_open()):
            async for item in engine.synthesize_stream("Test"):
                results.append(item)

        # Only final response, no chunks
        assert len(results) == 1
        response = results[0]
        assert isinstance(response, TTSResponse)
        assert len(response.audio_data) == 0
        assert response.duration_seconds == 0.0

    @pytest.mark.asyncio
    async def test_stream_server_error(self, config, mock_httpx_client):
        """Stream should wrap server errors in SynthesisError."""
        engine = CosyVoice3Engine(config)
        engine._client = mock_httpx_client
        engine._initialized = True

        # Mock HTTP error
        mock_stream_response = AsyncMock()
        mock_stream_response.raise_for_status = MagicMock(
            side_effect=Exception("HTTP 500")
        )
        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_httpx_client.stream = MagicMock(return_value=mock_stream_ctx)

        with (
            patch("pathlib.Path.open", _mock_open()),
            pytest.raises(SynthesisError) as exc_info,
        ):
            async for _ in engine.synthesize_stream("Test"):
                pass

        assert "failed" in str(exc_info.value).lower()


# =============================================================================
# Property Tests
# =============================================================================


class TestCosyVoice3Properties:
    """Test engine properties."""

    def test_supported_voices(self, config):
        """Should return configured voice names."""
        engine = CosyVoice3Engine(config)
        voices = engine.supported_voices
        assert "test_voice" in voices
        assert "another_voice" in voices

    def test_supported_voices_default(self, config_no_voices):
        """Should return ['default'] when no voices configured."""
        engine = CosyVoice3Engine(config_no_voices)
        assert engine.supported_voices == ["default"]

    def test_engine_name(self, config):
        """Should return cosyvoice3 as engine name."""
        engine = CosyVoice3Engine(config)
        assert engine.engine_name == "cosyvoice3"


# =============================================================================
# Helpers
# =============================================================================


async def _async_iter(items):
    """Create an async iterator from a list."""
    for item in items:
        yield item


def _mock_open():
    """Create a mock for builtins.open that returns a file-like object."""
    mock = MagicMock()
    mock.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock.return_value.__exit__ = MagicMock(return_value=False)
    return mock
