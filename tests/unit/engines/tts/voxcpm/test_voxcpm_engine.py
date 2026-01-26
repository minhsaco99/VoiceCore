"""
Unit tests for VoxCPM TTS Engine

Tests config, lifecycle, synthesize, and streaming with mocked VoxCPM model.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.engines.tts.voxcpm.config import VoxCPMConfig
from app.engines.tts.voxcpm.engine import VoxCPMEngine
from app.exceptions import EngineNotReadyError, SynthesisError
from app.models.engine import TTSChunk, TTSResponse

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Default VoxCPM config for testing."""
    return VoxCPMConfig(
        model_name="openbmb/VoxCPM-0.5B",
        device="cpu",
    )


@pytest.fixture
def mock_voxcpm_model():
    """Create a mock VoxCPM model."""
    mock_model = MagicMock()

    # Mock tts_model.sample_rate
    mock_model.tts_model = MagicMock()
    mock_model.tts_model.sample_rate = 16000

    # Mock generate method - returns numpy array
    mock_model.generate.return_value = np.zeros(16000, dtype=np.float32)  # 1 second

    # Mock generate_streaming - yields chunks (use MagicMock to track calls)
    def mock_streaming(*args, **kwargs):
        yield np.zeros(4000, dtype=np.float32)  # 0.25 second chunk
        yield np.zeros(4000, dtype=np.float32)  # 0.25 second chunk

    # Wrap in MagicMock to track call arguments
    mock_model.generate_streaming = MagicMock(side_effect=mock_streaming)

    return mock_model


# =============================================================================
# Config Tests
# =============================================================================


class TestVoxCPMConfig:
    """Test VoxCPMConfig validation and defaults."""

    def test_config_defaults(self):
        """Config should have correct default values."""
        config = VoxCPMConfig(model_name="test-model")

        assert config.model_name == "test-model"
        assert config.device == "cpu"
        assert config.cfg_value == 2.0
        assert config.inference_timesteps == 10
        assert config.normalize is False
        assert config.denoise is False
        assert config.retry_badcase is True
        assert config.retry_badcase_max_times == 3
        assert config.retry_badcase_ratio_threshold == 6.0

    def test_config_custom_values(self):
        """Config should accept custom values."""
        config = VoxCPMConfig(
            model_name="openbmb/VoxCPM1.5",
            device="cuda",
            cfg_value=3.0,
            inference_timesteps=20,
            denoise=True,
        )

        assert config.model_name == "openbmb/VoxCPM1.5"
        assert config.device == "cuda"
        assert config.cfg_value == 3.0
        assert config.inference_timesteps == 20
        assert config.denoise is True


# =============================================================================
# Engine Lifecycle Tests
# =============================================================================


class TestVoxCPMEngineLifecycle:
    """Test engine initialization and cleanup."""

    @pytest.mark.asyncio
    async def test_engine_initialization(self, config, mock_voxcpm_model):
        """Engine should load model on initialization."""
        engine = VoxCPMEngine(config)

        with patch.dict("sys.modules", {"voxcpm": MagicMock()}):
            import sys

            sys.modules["voxcpm"].VoxCPM = MagicMock()
            sys.modules[
                "voxcpm"
            ].VoxCPM.from_pretrained.return_value = mock_voxcpm_model

            await engine.initialize()

            assert engine.is_ready()
            sys.modules["voxcpm"].VoxCPM.from_pretrained.assert_called_once_with(
                config.model_name,
                load_denoiser=config.denoise,
            )

    @pytest.mark.asyncio
    async def test_engine_cleanup(self, config, mock_voxcpm_model):
        """Engine should release model on cleanup."""
        engine = VoxCPMEngine(config)
        engine._model = mock_voxcpm_model
        engine._initialized = True

        assert engine._model is not None

        await engine.close()
        assert engine._model is None
        assert not engine.is_ready()

    @pytest.mark.asyncio
    async def test_engine_raises_on_missing_package(self, config):
        """Engine should raise EngineNotReadyError if voxcpm not installed."""
        engine = VoxCPMEngine(config)

        # Mock the import to raise ImportError
        with patch.dict("sys.modules", {"voxcpm": None}):
            with pytest.raises(EngineNotReadyError) as exc_info:
                await engine.initialize()

            assert "voxcpm" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_initialization_generic_error(self, config):
        """Engine should wrap generic initialization errors."""
        engine = VoxCPMEngine(config)

        # Mock ImportError to simulate generic exception during load
        with patch.dict("sys.modules", {"voxcpm": MagicMock()}):
            import sys

            sys.modules["voxcpm"].VoxCPM = MagicMock()
            sys.modules["voxcpm"].VoxCPM.from_pretrained.side_effect = Exception(
                "Unknown error"
            )

            with pytest.raises(EngineNotReadyError) as exc_info:
                await engine.initialize()

            assert "Failed to load VoxCPM model" in str(exc_info.value)


# =============================================================================
# Synthesize Tests
# =============================================================================


class TestVoxCPMSynthesize:
    """Test batch synthesis."""

    @pytest.mark.asyncio
    async def test_synthesize_returns_audio(self, config, mock_voxcpm_model):
        """Synthesize should return TTSResponse with audio data."""
        engine = VoxCPMEngine(config)
        engine._model = mock_voxcpm_model
        engine._initialized = True

        result = await engine.synthesize("Hello world")

        assert isinstance(result, TTSResponse)
        assert result.audio_data is not None
        assert len(result.audio_data) > 0
        assert result.sample_rate == 16000
        assert result.duration_seconds > 0
        assert result.format == "wav"
        assert result.performance_metrics is not None

    @pytest.mark.asyncio
    async def test_synthesize_passes_config_params(self, config, mock_voxcpm_model):
        """Synthesize should pass config params to model."""
        engine = VoxCPMEngine(config)
        engine._model = mock_voxcpm_model
        engine._initialized = True

        await engine.synthesize("Test")

        mock_voxcpm_model.generate.assert_called_once()
        call_kwargs = mock_voxcpm_model.generate.call_args.kwargs

        assert call_kwargs["cfg_value"] == config.cfg_value
        assert call_kwargs["inference_timesteps"] == config.inference_timesteps
        assert call_kwargs["normalize"] == config.normalize
        assert call_kwargs["denoise"] == config.denoise

    @pytest.mark.asyncio
    async def test_synthesize_with_voice_cloning(self, config, mock_voxcpm_model):
        """Synthesize should accept voice cloning parameters."""
        engine = VoxCPMEngine(config)
        engine._model = mock_voxcpm_model
        engine._initialized = True

        # Use bytes for reference audio
        reference_audio = b"fake audio bytes"

        await engine.synthesize(
            "Hello",
            reference_audio=reference_audio,
            reference_text="Reference text",
        )

        call_kwargs = mock_voxcpm_model.generate.call_args.kwargs
        # Engine saves bytes to temp file and passes path
        assert call_kwargs["prompt_wav_path"] is not None
        assert call_kwargs["prompt_text"] == "Reference text"

    @pytest.mark.asyncio
    async def test_synthesize_wraps_model_errors(self, config, mock_voxcpm_model):
        """Model errors should be wrapped in SynthesisError."""
        engine = VoxCPMEngine(config)
        engine._model = mock_voxcpm_model
        engine._initialized = True

        mock_voxcpm_model.generate.side_effect = RuntimeError("Model failed")

        with pytest.raises(SynthesisError) as exc_info:
            await engine.synthesize("Test")

        assert "Model failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_synthesize_reraises_typed_errors(self, config, mock_voxcpm_model):
        """Synthesize should re-raise SynthesisError as-is without wrapping."""
        engine = VoxCPMEngine(config)
        engine._model = mock_voxcpm_model
        engine._initialized = True

        mock_voxcpm_model.generate.side_effect = SynthesisError("Already typed error")

        with pytest.raises(SynthesisError) as exc_info:
            await engine.synthesize("Test")

        assert "Already typed error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_synthesize_model_not_loaded(self, config):
        """Synthesize should raise EngineNotReadyError if model is None."""
        engine = VoxCPMEngine(config)
        # initialized=True but model is None simulates a state where
        # check passes but model is actually missing (should be impossible in normal flow but good for coverage)
        # OR simply _model is None (which implies not ready)
        engine._initialized = True
        engine._model = None

        with pytest.raises(EngineNotReadyError) as exc_info:
            await engine.synthesize("Test")

        assert "not loaded" in str(exc_info.value)


# =============================================================================
# Streaming Tests
# =============================================================================


class TestVoxCPMStreaming:
    """Test streaming synthesis."""

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self, config, mock_voxcpm_model):
        """Streaming should yield TTSChunks then TTSResponse."""
        engine = VoxCPMEngine(config)
        engine._model = mock_voxcpm_model
        engine._initialized = True

        results = []
        async for item in engine.synthesize_stream("Hello world"):
            results.append(item)

        # Should have chunks followed by final response
        assert len(results) >= 2

        # All but last should be TTSChunk
        for chunk in results[:-1]:
            assert isinstance(chunk, TTSChunk)
            assert chunk.audio_data is not None
            assert chunk.sequence_number >= 0

        # Last should be TTSResponse
        final = results[-1]
        assert isinstance(final, TTSResponse)
        assert final.performance_metrics is not None
        assert final.performance_metrics.total_chunks == len(results) - 1

    @pytest.mark.asyncio
    async def test_stream_metrics_include_ttfb(self, config, mock_voxcpm_model):
        """Streaming metrics should include time to first byte."""
        engine = VoxCPMEngine(config)
        engine._model = mock_voxcpm_model
        engine._initialized = True

        final_response = None
        async for item in engine.synthesize_stream("Test"):
            if isinstance(item, TTSResponse):
                final_response = item

        assert final_response is not None
        assert final_response.performance_metrics.time_to_first_byte_ms is not None

    @pytest.mark.asyncio
    async def test_stream_model_not_loaded(self, config):
        """Stream should raise EngineNotReadyError if model is None."""
        engine = VoxCPMEngine(config)
        engine._initialized = True
        engine._model = None

        with pytest.raises(EngineNotReadyError) as exc_info:
            async for _ in engine.synthesize_stream("Test"):
                pass

        assert "not loaded" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_stream_with_reference_audio(self, config, mock_voxcpm_model):
        """Stream should accept reference_audio bytes."""
        engine = VoxCPMEngine(config)
        engine._model = mock_voxcpm_model
        engine._initialized = True

        reference_audio = b"fake audio bytes"
        results = []
        async for item in engine.synthesize_stream(
            "Test",
            reference_audio=reference_audio,
            reference_text="Reference",
        ):
            results.append(item)

        # Verify model was called with prompt_wav_path (temp file path)
        call_kwargs = mock_voxcpm_model.generate_streaming.call_args[1]
        assert call_kwargs["prompt_wav_path"] is not None
        assert call_kwargs["prompt_text"] == "Reference"

    @pytest.mark.asyncio
    async def test_stream_empty_result(self, config, mock_voxcpm_model):
        """Stream should handle empty generator gracefully."""
        engine = VoxCPMEngine(config)
        engine._model = mock_voxcpm_model
        engine._initialized = True

        # Mock empty streaming
        def empty_stream(*args, **kwargs):
            if False:
                yield  # Force generator type
            return

        mock_voxcpm_model.generate_streaming = empty_stream

        results = []
        async for item in engine.synthesize_stream("Test"):
            results.append(item)

        assert len(results) == 1
        response = results[0]
        assert isinstance(response, TTSResponse)
        assert len(response.audio_data) == 0
        assert response.duration_seconds == 0.0

    @pytest.mark.asyncio
    async def test_stream_generic_error(self, config, mock_voxcpm_model):
        """Stream should wrap generic errors in SynthesisError."""
        engine = VoxCPMEngine(config)
        engine._model = mock_voxcpm_model
        engine._initialized = True

        # Mock error during streaming
        def error_stream(*args, **kwargs):
            if False:
                yield
            raise RuntimeError("Stream failed")

        mock_voxcpm_model.generate_streaming = error_stream

        with pytest.raises(SynthesisError) as exc_info:
            async for _ in engine.synthesize_stream("Test"):
                pass

        assert "Stream failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_stream_reraises_typed_error(self, config, mock_voxcpm_model):
        """Stream should re-raise EngineNotReadyError/SynthesisError as-is."""
        engine = VoxCPMEngine(config)
        engine._model = mock_voxcpm_model
        engine._initialized = True

        # Mock typed error
        def typed_error_stream(*args, **kwargs):
            if False:
                yield
            raise EngineNotReadyError("Already typed")

        mock_voxcpm_model.generate_streaming = typed_error_stream

        # Should match exact type, not wrapped
        with pytest.raises(EngineNotReadyError):
            async for _ in engine.synthesize_stream("Test"):
                pass


# =============================================================================
# Property Tests
# =============================================================================


class TestVoxCPMProperties:
    """Test engine properties."""

    def test_supported_voices(self, config):
        """Should return default voice list."""
        engine = VoxCPMEngine(config)
        assert engine.supported_voices == ["default"]

    def test_engine_name(self, config):
        """Should return voxcpm as engine name."""
        engine = VoxCPMEngine(config)
        assert engine.engine_name == "voxcpm"
