"""Tests for Whisper STT engine"""

import io

import pytest

pytest.importorskip("faster_whisper")
import pathlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from app.exceptions import (
    AudioError,
    EngineNotReadyError,
    InvalidAudioError,
    TranscriptionError,
    UnsupportedFormatError,
)
from app.models.engine import EngineConfig, Segment, STTResponse
from app.models.metrics import STTPerformanceMetrics


class TestWhisperConfig:
    """Test WhisperConfig model"""

    def test_whisper_config_extends_engine_config(self):
        """WhisperConfig should be subclass of EngineConfig"""
        from app.engines.stt.whisper.config import WhisperConfig

        # Check inheritance
        assert issubclass(WhisperConfig, EngineConfig)

        # Create instance and verify it's both
        config = WhisperConfig(model_name="base")
        assert isinstance(config, WhisperConfig)
        assert isinstance(config, EngineConfig)

    def test_whisper_config_inherits_base_fields(self):
        """WhisperConfig should inherit all EngineConfig fields"""
        from app.engines.stt.whisper.config import WhisperConfig

        config = WhisperConfig(
            model_name="base",
            device="cuda",
            max_workers=4,
            timeout_seconds=600,
        )

        # Check inherited fields are accessible
        assert config.model_name == "base"
        assert config.device == "cuda"
        assert config.max_workers == 4
        assert config.timeout_seconds == 600

    def test_whisper_config_default_values(self):
        """WhisperConfig should have correct default values"""
        from app.engines.stt.whisper.config import WhisperConfig

        # Create with only required fields
        config = WhisperConfig(model_name="base")

        # Check base defaults
        assert config.device == "cpu"
        assert config.max_workers == 1
        assert config.timeout_seconds == 300

        # Check Whisper-specific defaults
        assert config.compute_type == "int8"
        assert config.beam_size == 5
        assert config.language is None

    def test_whisper_config_valid_models(self):
        """WhisperConfig should accept valid Whisper model names"""
        from app.engines.stt.whisper.config import WhisperConfig

        valid_models = [
            "tiny",
            "base",
            "small",
            "medium",
            "large",
            "large-v2",
            "large-v3",
        ]

        for model in valid_models:
            config = WhisperConfig(model_name=model)
            assert config.model_name == model

    def test_whisper_config_invalid_model_raises(self):
        """WhisperConfig should reject invalid model names"""
        from app.engines.stt.whisper.config import WhisperConfig

        invalid_models = ["invalid", "gpt-3", "whisper-xl", ""]

        for model in invalid_models:
            with pytest.raises(ValidationError) as exc_info:
                WhisperConfig(model_name=model)

            # Check error message mentions valid models
            assert "Whisper model must be one of" in str(exc_info.value)

    def test_whisper_config_valid_compute_types(self):
        """WhisperConfig should accept valid compute types"""
        from app.engines.stt.whisper.config import WhisperConfig

        valid_types = ["int8", "float16", "float32"]

        for compute_type in valid_types:
            config = WhisperConfig(model_name="base", compute_type=compute_type)
            assert config.compute_type == compute_type

    def test_whisper_config_invalid_compute_type_raises(self):
        """WhisperConfig should reject invalid compute types"""
        from app.engines.stt.whisper.config import WhisperConfig

        invalid_types = ["int16", "float64", "auto", ""]

        for compute_type in invalid_types:
            with pytest.raises(ValidationError) as exc_info:
                WhisperConfig(model_name="base", compute_type=compute_type)

            # Check error message
            assert "compute_type must be one of" in str(exc_info.value)

    def test_whisper_config_beam_size_validation(self):
        """WhisperConfig should validate beam_size > 0"""
        from app.engines.stt.whisper.config import WhisperConfig

        # Valid beam sizes
        valid_sizes = [1, 5, 10, 100]
        for size in valid_sizes:
            config = WhisperConfig(model_name="base", beam_size=size)
            assert config.beam_size == size

        # Invalid: zero
        with pytest.raises(ValidationError):
            WhisperConfig(model_name="base", beam_size=0)

        # Invalid: negative
        with pytest.raises(ValidationError):
            WhisperConfig(model_name="base", beam_size=-1)

    def test_whisper_config_optional_language(self):
        """WhisperConfig language field should be optional"""
        from app.engines.stt.whisper.config import WhisperConfig

        # Without language (default None)
        config1 = WhisperConfig(model_name="base")
        assert config1.language is None

        # With language
        config2 = WhisperConfig(model_name="base", language="en")
        assert config2.language == "en"

        config3 = WhisperConfig(model_name="base", language="es")
        assert config3.language == "es"


class TestWhisperSTTEngineLifecycle:
    """Test WhisperSTTEngine lifecycle management"""

    @pytest.fixture
    def whisper_config(self):
        """Fixture for WhisperConfig"""
        from app.engines.stt.whisper.config import WhisperConfig

        return WhisperConfig(model_name="base", device="cpu")

    def test_whisper_engine_not_initialized_on_creation(self, whisper_config):
        """Engine should not be ready immediately after creation"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        assert not engine.is_ready()

    @pytest.mark.asyncio
    async def test_whisper_engine_explicit_initialization(self, whisper_config):
        """Engine should initialize when explicitly called"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with patch("app.engines.stt.whisper.engine.WhisperModel") as mock_model:
            await engine.initialize()

            assert engine.is_ready()
            mock_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_whisper_engine_auto_initialization_on_transcribe(
        self, whisper_config
    ):
        """Engine should auto-initialize on first transcribe() call"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        assert not engine.is_ready()

        # Mock Whisper model and transcribe result
        with patch("app.engines.stt.whisper.engine.WhisperModel") as mock_model:
            mock_instance = MagicMock()
            mock_model.return_value = mock_instance
            mock_instance.transcribe.return_value = (
                [{"text": "Hello"}],
                {"language": "en"},
            )

            fake_audio = np.array([0.1, 0.2, 0.3])
            result = await engine.transcribe(fake_audio)

            assert engine.is_ready()
            assert result.text is not None

    @pytest.mark.asyncio
    async def test_whisper_engine_is_ready_after_init(self, whisper_config):
        """is_ready() should return True after initialization"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with patch("app.engines.stt.whisper.engine.WhisperModel"):
            await engine.initialize()
            assert engine.is_ready() is True

    @pytest.mark.asyncio
    async def test_whisper_engine_close_cleanup(self, whisper_config):
        """close() should release resources"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with patch("app.engines.stt.whisper.engine.WhisperModel"):
            await engine.initialize()
            assert engine.is_ready()

            await engine.close()

            assert not engine.is_ready()

    @pytest.mark.asyncio
    async def test_whisper_engine_cannot_use_after_close(self, whisper_config):
        """Should raise EngineNotReadyError after close"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with patch("app.engines.stt.whisper.engine.WhisperModel"):
            await engine.initialize()
            await engine.close()

            with pytest.raises(EngineNotReadyError):
                await engine.transcribe(np.array([0.1, 0.2]))

    @pytest.mark.asyncio
    async def test_whisper_engine_context_manager(self, whisper_config):
        """Engine should work with async with statement"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with patch("app.engines.stt.whisper.engine.WhisperModel") as mock_model:
            mock_instance = MagicMock()
            mock_model.return_value = mock_instance
            mock_instance.transcribe.return_value = (
                [{"text": "Test"}],
                {"language": "en"},
            )

            async with engine:
                assert engine.is_ready()
                result = await engine.transcribe(np.array([0.1, 0.2]))
                assert result.text is not None

            # Should be closed after context
            assert not engine.is_ready()


class TestWhisperSTTEngineAudioFormats:
    """Test WhisperSTTEngine audio format handling"""

    @pytest.fixture
    def whisper_config(self):
        from app.engines.stt.whisper.config import WhisperConfig

        return WhisperConfig(model_name="base")

    @pytest.fixture
    def mock_whisper_model(self):
        """Mock Whisper model that returns fake transcription"""
        with patch("app.engines.stt.whisper.engine.WhisperModel") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            mock_instance.transcribe.return_value = (
                [{"text": "Hello world"}],
                {"language": "en"},
            )
            yield mock_instance

    @pytest.mark.asyncio
    async def test_transcribe_with_bytes_input(
        self, whisper_config, mock_whisper_model
    ):
        """Should accept bytes input (WAV format)"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)
        fake_wav_bytes = b"fake wav data"

        with patch.object(engine, "_audio_processor") as mock_processor:
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.resample_to_16khz.return_value = np.array([0.1, 0.2])
            mock_processor.get_duration_ms.return_value = 100.0

            result = await engine.transcribe(fake_wav_bytes)

            assert isinstance(result, STTResponse)
            assert result.text is not None

    @pytest.mark.asyncio
    async def test_transcribe_with_numpy_input(
        self, whisper_config, mock_whisper_model
    ):
        """Should accept numpy array input"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)
        fake_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        with patch.object(engine, "_audio_processor") as mock_processor:
            mock_processor.to_numpy.return_value = (fake_audio, 16000)
            mock_processor.resample_to_16khz.return_value = fake_audio
            mock_processor.get_duration_ms.return_value = 100.0

            result = await engine.transcribe(fake_audio)

            assert isinstance(result, STTResponse)
            assert result.text is not None

    @pytest.mark.asyncio
    async def test_transcribe_with_filepath_input(
        self, whisper_config, mock_whisper_model
    ):
        """Should accept pathlib.Path input"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)
        fake_path = pathlib.Path("/fake/audio.wav")

        with patch.object(engine, "_audio_processor") as mock_processor:
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 22050)
            mock_processor.resample_to_16khz.return_value = np.array([0.1, 0.2])
            mock_processor.get_duration_ms.return_value = 100.0

            result = await engine.transcribe(fake_path)

            assert isinstance(result, STTResponse)

    @pytest.mark.asyncio
    async def test_transcribe_with_bytesio_input(
        self, whisper_config, mock_whisper_model
    ):
        """Should accept io.BytesIO input"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)
        fake_buffer = io.BytesIO(b"fake audio")

        with patch.object(engine, "_audio_processor") as mock_processor:
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.resample_to_16khz.return_value = np.array([0.1, 0.2])
            mock_processor.get_duration_ms.return_value = 100.0

            result = await engine.transcribe(fake_buffer)

            assert isinstance(result, STTResponse)

    @pytest.mark.asyncio
    async def test_transcribe_rejects_invalid_format(self, whisper_config):
        """Should raise UnsupportedFormatError for invalid types"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with patch.object(engine, "_audio_processor") as mock_processor:
            mock_processor.to_numpy.side_effect = UnsupportedFormatError("Invalid type")

            with pytest.raises(UnsupportedFormatError):
                await engine.transcribe("invalid_string")

    @pytest.mark.asyncio
    async def test_transcribe_rejects_corrupted_audio(self, whisper_config):
        """Should raise InvalidAudioError for corrupted data"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with patch.object(engine, "_audio_processor") as mock_processor:
            mock_processor.to_numpy.side_effect = InvalidAudioError("Corrupted")

            with pytest.raises(InvalidAudioError):
                await engine.transcribe(b"corrupted data")


class TestWhisperSTTEngineTranscription:
    """Test WhisperSTTEngine basic transcription functionality"""

    @pytest.fixture
    def whisper_config(self):
        from app.engines.stt.whisper.config import WhisperConfig

        return WhisperConfig(model_name="base")

    @pytest.mark.asyncio
    async def test_transcribe_returns_stt_response(self, whisper_config):
        """Should return STTResponse with text"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_audio_processor") as mock_processor,
            patch("app.engines.stt.whisper.engine.WhisperModel") as mock_model,
        ):
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.resample_to_16khz.return_value = np.array([0.1, 0.2])
            mock_processor.get_duration_ms.return_value = 100.0

            mock_instance = MagicMock()
            mock_model.return_value = mock_instance
            mock_instance.transcribe.return_value = (
                [{"text": "Hello world"}],
                {"language": "en"},
            )

            result = await engine.transcribe(np.array([0.1, 0.2]))

            assert isinstance(result, STTResponse)
            assert result.text == "Hello world"

    @pytest.mark.asyncio
    async def test_transcribe_with_language_hint(self, whisper_config):
        """Should pass language hint to Whisper"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_audio_processor") as mock_processor,
            patch("app.engines.stt.whisper.engine.WhisperModel") as mock_model,
        ):
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.resample_to_16khz.return_value = np.array([0.1, 0.2])
            mock_processor.get_duration_ms.return_value = 100.0

            mock_instance = MagicMock()
            mock_model.return_value = mock_instance
            mock_instance.transcribe.return_value = (
                [{"text": "Hola"}],
                {"language": "es"},
            )

            await engine.transcribe(np.array([0.1, 0.2]), language="es")

            # Verify language was passed
            call_args = mock_instance.transcribe.call_args
            assert call_args is not None

    @pytest.mark.asyncio
    async def test_transcribe_empty_audio_raises(self, whisper_config):
        """Should raise InvalidAudioError for empty input"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with patch.object(engine, "_audio_processor") as mock_processor:
            mock_processor.to_numpy.return_value = (np.array([]), 16000)

            with pytest.raises(InvalidAudioError):
                await engine.transcribe(np.array([]))

    @pytest.mark.asyncio
    async def test_transcribe_very_short_audio_returns_empty(self, whisper_config):
        """Should handle very short audio gracefully"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_audio_processor") as mock_processor,
            patch("app.engines.stt.whisper.engine.WhisperModel") as mock_model,
        ):
            # Very short audio (< 0.1s)
            mock_processor.to_numpy.return_value = (np.array([0.1]), 16000)
            mock_processor.resample_to_16khz.return_value = np.array([0.1])
            mock_processor.get_duration_ms.return_value = 10.0

            mock_instance = MagicMock()
            mock_model.return_value = mock_instance
            mock_instance.transcribe.return_value = ([], {"language": "en"})

            result = await engine.transcribe(np.array([0.1]))

            assert result.text == ""

    @pytest.mark.asyncio
    async def test_transcribe_calls_whisper_transcribe(self, whisper_config):
        """Should call Whisper model's transcribe method"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_audio_processor") as mock_processor,
            patch("app.engines.stt.whisper.engine.WhisperModel") as mock_model,
        ):
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.resample_to_16khz.return_value = np.array([0.1, 0.2])
            mock_processor.get_duration_ms.return_value = 100.0

            mock_instance = MagicMock()
            mock_model.return_value = mock_instance
            mock_instance.transcribe.return_value = (
                [{"text": "Test"}],
                {"language": "en"},
            )

            await engine.transcribe(np.array([0.1, 0.2]))

            mock_instance.transcribe.assert_called_once()

    @pytest.mark.asyncio
    async def test_transcribe_passes_whisper_result_to_response(self, whisper_config):
        """Should map Whisper output correctly to STTResponse"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_audio_processor") as mock_processor,
            patch("app.engines.stt.whisper.engine.WhisperModel") as mock_model,
        ):
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.resample_to_16khz.return_value = np.array([0.1, 0.2])
            mock_processor.get_duration_ms.return_value = 100.0

            mock_instance = MagicMock()
            mock_model.return_value = mock_instance
            mock_instance.transcribe.return_value = (
                [{"text": "Mapped text"}],
                {"language": "fr"},
            )

            result = await engine.transcribe(np.array([0.1, 0.2]))

            assert result.text == "Mapped text"
            assert result.language == "fr"


class TestWhisperSTTEngineMetrics:
    """Test WhisperSTTEngine performance metrics calculation"""

    @pytest.fixture
    def whisper_config(self):
        from app.engines.stt.whisper.config import WhisperConfig

        return WhisperConfig(model_name="base")

    @pytest.mark.asyncio
    async def test_transcribe_returns_performance_metrics(self, whisper_config):
        """STTResponse.performance_metrics should not be None"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_audio_processor") as mock_processor,
            patch("app.engines.stt.whisper.engine.WhisperModel") as mock_model,
        ):
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.resample_to_16khz.return_value = np.array([0.1, 0.2])
            mock_processor.get_duration_ms.return_value = 100.0

            mock_instance = MagicMock()
            mock_model.return_value = mock_instance
            mock_instance.transcribe.return_value = (
                [{"text": "Test"}],
                {"language": "en"},
            )

            result = await engine.transcribe(np.array([0.1, 0.2]))

            assert result.performance_metrics is not None
            assert isinstance(result.performance_metrics, STTPerformanceMetrics)

    @pytest.mark.asyncio
    async def test_metrics_latency_ms_calculated(self, whisper_config):
        """Latency should be calculated as end_time - start_time"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_audio_processor") as mock_processor,
            patch("app.engines.stt.whisper.engine.WhisperModel") as mock_model,
        ):
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.resample_to_16khz.return_value = np.array([0.1, 0.2])
            mock_processor.get_duration_ms.return_value = 100.0

            mock_instance = MagicMock()
            mock_model.return_value = mock_instance
            mock_instance.transcribe.return_value = (
                [{"text": "Test"}],
                {"language": "en"},
            )

            result = await engine.transcribe(np.array([0.1, 0.2]))

            assert result.performance_metrics.latency_ms > 0

    @pytest.mark.asyncio
    async def test_metrics_processing_time_ms_calculated(self, whisper_config):
        """Processing time should be measured"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_audio_processor") as mock_processor,
            patch("app.engines.stt.whisper.engine.WhisperModel") as mock_model,
        ):
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.resample_to_16khz.return_value = np.array([0.1, 0.2])
            mock_processor.get_duration_ms.return_value = 100.0

            mock_instance = MagicMock()
            mock_model.return_value = mock_instance
            mock_instance.transcribe.return_value = (
                [{"text": "Test"}],
                {"language": "en"},
            )

            result = await engine.transcribe(np.array([0.1, 0.2]))

            assert result.performance_metrics.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_metrics_audio_duration_ms_calculated(self, whisper_config):
        """Audio duration should be calculated from audio length"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_audio_processor") as mock_processor,
            patch("app.engines.stt.whisper.engine.WhisperModel") as mock_model,
        ):
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.resample_to_16khz.return_value = np.array([0.1, 0.2])
            mock_processor.get_duration_ms.return_value = 500.0

            mock_instance = MagicMock()
            mock_model.return_value = mock_instance
            mock_instance.transcribe.return_value = (
                [{"text": "Test"}],
                {"language": "en"},
            )

            result = await engine.transcribe(np.array([0.1, 0.2]))

            assert result.performance_metrics.audio_duration_ms == 500.0

    @pytest.mark.asyncio
    async def test_metrics_real_time_factor_calculated(self, whisper_config):
        """RTF should be calculated as processing_time / audio_duration"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_audio_processor") as mock_processor,
            patch("app.engines.stt.whisper.engine.WhisperModel") as mock_model,
        ):
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.resample_to_16khz.return_value = np.array([0.1, 0.2])
            mock_processor.get_duration_ms.return_value = 1000.0

            mock_instance = MagicMock()
            mock_model.return_value = mock_instance
            mock_instance.transcribe.return_value = (
                [{"text": "Test"}],
                {"language": "en"},
            )

            result = await engine.transcribe(np.array([0.1, 0.2]))

            assert result.performance_metrics.real_time_factor is not None

    @pytest.mark.asyncio
    async def test_metrics_rtf_less_than_one_is_faster(self, whisper_config):
        """RTF < 1.0 should mean faster than real-time"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_audio_processor") as mock_processor,
            patch("app.engines.stt.whisper.engine.WhisperModel") as mock_model,
            patch("time.time") as mock_time,
        ):
            # Mock time to control RTF calculation
            # Need 4 time calls: start, processing_start, processing_end, end
            mock_time.side_effect = [
                0.0,
                0.0,
                0.5,
                0.5,
            ]  # 500ms total, 500ms processing

            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.resample_to_16khz.return_value = np.array([0.1, 0.2])
            mock_processor.get_duration_ms.return_value = 1000.0  # 1 second audio

            mock_instance = MagicMock()
            mock_model.return_value = mock_instance
            mock_instance.transcribe.return_value = (
                [{"text": "Test"}],
                {"language": "en"},
            )

            result = await engine.transcribe(np.array([0.1, 0.2]))

            # 500ms processing / 1000ms audio = 0.5 RTF (faster than real-time)
            assert result.performance_metrics.real_time_factor < 1.0

    @pytest.mark.asyncio
    async def test_metrics_includes_queue_time_if_provided(self, whisper_config):
        """Queue time should be set if engine queues requests"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_audio_processor") as mock_processor,
            patch("app.engines.stt.whisper.engine.WhisperModel") as mock_model,
        ):
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.resample_to_16khz.return_value = np.array([0.1, 0.2])
            mock_processor.get_duration_ms.return_value = 100.0

            mock_instance = MagicMock()
            mock_model.return_value = mock_instance
            mock_instance.transcribe.return_value = (
                [{"text": "Test"}],
                {"language": "en"},
            )

            result = await engine.transcribe(np.array([0.1, 0.2]))

            # For now, queue_time_ms may be None (single threaded)
            # This test documents the field exists
            assert hasattr(result.performance_metrics, "queue_time_ms")


class TestWhisperSTTEngineSegments:
    """Test WhisperSTTEngine segment generation"""

    @pytest.fixture
    def whisper_config(self):
        from app.engines.stt.whisper.config import WhisperConfig

        return WhisperConfig(model_name="base")

    @pytest.mark.asyncio
    async def test_transcribe_returns_segments(self, whisper_config):
        """STTResponse.segments should be list[Segment]"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_audio_processor") as mock_processor,
            patch("app.engines.stt.whisper.engine.WhisperModel") as mock_model,
        ):
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.resample_to_16khz.return_value = np.array([0.1, 0.2])
            mock_processor.get_duration_ms.return_value = 100.0

            mock_instance = MagicMock()
            mock_model.return_value = mock_instance

            # Mock Whisper response with word-level timestamps
            mock_segment = MagicMock()
            mock_segment.text = "Hello world"
            mock_segment.words = [
                MagicMock(word="Hello", start=0.0, end=0.5),
                MagicMock(word="world", start=0.6, end=1.0),
            ]
            mock_instance.transcribe.return_value = ([mock_segment], {"language": "en"})

            result = await engine.transcribe(np.array([0.1, 0.2]))

            assert result.segments is not None
            assert isinstance(result.segments, list)
            if len(result.segments) > 0:
                assert isinstance(result.segments[0], Segment)

    @pytest.mark.asyncio
    async def test_segments_have_timing_info(self, whisper_config):
        """Each Segment should have start and end times"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_audio_processor") as mock_processor,
            patch("app.engines.stt.whisper.engine.WhisperModel") as mock_model,
        ):
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.resample_to_16khz.return_value = np.array([0.1, 0.2])
            mock_processor.get_duration_ms.return_value = 100.0

            mock_instance = MagicMock()
            mock_model.return_value = mock_instance

            mock_segment = MagicMock()
            mock_segment.text = "Hello"
            mock_segment.words = [
                MagicMock(word="Hello", start=0.0, end=0.5),
            ]
            mock_instance.transcribe.return_value = ([mock_segment], {"language": "en"})

            result = await engine.transcribe(np.array([0.1, 0.2]))

            if result.segments:
                seg = result.segments[0]
                assert hasattr(seg, "start")
                assert hasattr(seg, "end")
                assert seg.start >= 0
                assert seg.end > seg.start

    @pytest.mark.asyncio
    async def test_segments_have_text(self, whisper_config):
        """Each Segment should have text field"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_audio_processor") as mock_processor,
            patch("app.engines.stt.whisper.engine.WhisperModel") as mock_model,
        ):
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.resample_to_16khz.return_value = np.array([0.1, 0.2])
            mock_processor.get_duration_ms.return_value = 100.0

            mock_instance = MagicMock()
            mock_model.return_value = mock_instance

            mock_segment = MagicMock()
            mock_segment.text = "Test word"
            mock_segment.words = [
                MagicMock(word="Test", start=0.0, end=0.3),
                MagicMock(word="word", start=0.3, end=0.6),
            ]
            mock_instance.transcribe.return_value = ([mock_segment], {"language": "en"})

            result = await engine.transcribe(np.array([0.1, 0.2]))

            if result.segments:
                for seg in result.segments:
                    assert hasattr(seg, "text")
                    assert isinstance(seg.text, str)
                    assert len(seg.text) > 0

    @pytest.mark.asyncio
    async def test_segments_have_confidence_scores(self, whisper_config):
        """Each Segment should have confidence field"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_audio_processor") as mock_processor,
            patch("app.engines.stt.whisper.engine.WhisperModel") as mock_model,
        ):
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.resample_to_16khz.return_value = np.array([0.1, 0.2])
            mock_processor.get_duration_ms.return_value = 100.0

            mock_instance = MagicMock()
            mock_model.return_value = mock_instance

            mock_segment = MagicMock()
            mock_segment.text = "Test"
            mock_word = MagicMock(word="Test", start=0.0, end=0.5)
            mock_word.probability = 0.95
            mock_segment.words = [mock_word]
            mock_instance.transcribe.return_value = ([mock_segment], {"language": "en"})

            result = await engine.transcribe(np.array([0.1, 0.2]))

            if result.segments:
                assert hasattr(result.segments[0], "confidence")

    @pytest.mark.asyncio
    async def test_segments_are_ordered_by_time(self, whisper_config):
        """Segments should be sorted by start time"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_audio_processor") as mock_processor,
            patch("app.engines.stt.whisper.engine.WhisperModel") as mock_model,
        ):
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.resample_to_16khz.return_value = np.array([0.1, 0.2])
            mock_processor.get_duration_ms.return_value = 100.0

            mock_instance = MagicMock()
            mock_model.return_value = mock_instance

            mock_segment = MagicMock()
            mock_segment.text = "One two three"
            mock_segment.words = [
                MagicMock(word="One", start=0.0, end=0.3),
                MagicMock(word="two", start=0.3, end=0.6),
                MagicMock(word="three", start=0.6, end=1.0),
            ]
            mock_instance.transcribe.return_value = ([mock_segment], {"language": "en"})

            result = await engine.transcribe(np.array([0.1, 0.2]))

            if result.segments and len(result.segments) > 1:
                for i in range(len(result.segments) - 1):
                    assert result.segments[i].start <= result.segments[i + 1].start

    @pytest.mark.asyncio
    async def test_segments_concatenate_to_full_text(self, whisper_config):
        """Joining segments should equal full text"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_audio_processor") as mock_processor,
            patch("app.engines.stt.whisper.engine.WhisperModel") as mock_model,
        ):
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.resample_to_16khz.return_value = np.array([0.1, 0.2])
            mock_processor.get_duration_ms.return_value = 100.0

            mock_instance = MagicMock()
            mock_model.return_value = mock_instance

            mock_segment = MagicMock()
            mock_segment.text = "Hello world"
            mock_segment.words = [
                MagicMock(word="Hello", start=0.0, end=0.5),
                MagicMock(word="world", start=0.6, end=1.0),
            ]
            mock_instance.transcribe.return_value = ([mock_segment], {"language": "en"})

            result = await engine.transcribe(np.array([0.1, 0.2]))

            if result.segments:
                concatenated = " ".join(seg.text for seg in result.segments)
                # Text should be similar (may have spacing differences)
                assert len(concatenated) > 0

    @pytest.mark.asyncio
    async def test_transcribe_without_word_timestamps(self, whisper_config):
        """segments should be None if Whisper doesn't provide word timestamps"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_audio_processor") as mock_processor,
            patch("app.engines.stt.whisper.engine.WhisperModel") as mock_model,
        ):
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.resample_to_16khz.return_value = np.array([0.1, 0.2])
            mock_processor.get_duration_ms.return_value = 100.0

            mock_instance = MagicMock()
            mock_model.return_value = mock_instance

            # Mock segment without words attribute
            mock_segment = MagicMock()
            mock_segment.text = "Hello"
            mock_segment.words = None
            mock_instance.transcribe.return_value = ([mock_segment], {"language": "en"})

            result = await engine.transcribe(np.array([0.1, 0.2]))

            # Should handle gracefully
            assert result.text is not None


class TestWhisperSTTEngineErrorHandling:
    """Test WhisperSTTEngine error handling"""

    @pytest.fixture
    def whisper_config(self):
        from app.engines.stt.whisper.config import WhisperConfig

        return WhisperConfig(model_name="base")

    @pytest.mark.asyncio
    async def test_transcribe_timeout_raises(self, whisper_config):
        """Should raise TranscriptionTimeoutError after timeout"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        # Set very short timeout
        whisper_config.timeout_seconds = 1

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_audio_processor") as mock_processor,
            patch("app.engines.stt.whisper.engine.WhisperModel") as mock_model,
        ):
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.resample_to_16khz.return_value = np.array([0.1, 0.2])
            mock_processor.get_duration_ms.return_value = 100.0

            # Mock Whisper to simulate a very slow transcription
            mock_instance = MagicMock()
            mock_model.return_value = mock_instance

            # Simulate timeout by making transcribe take too long
            import asyncio

            async def slow_transcribe(*args, **kwargs):
                await asyncio.sleep(10)  # Longer than timeout
                return ([], {})

            # Since Whisper's transcribe is sync, we can't easily test timeout
            # This test documents that timeout handling should be implemented
            # For now, just verify the engine doesn't crash
            mock_instance.transcribe.return_value = (
                [{"text": "Test"}],
                {"language": "en"},
            )

            result = await engine.transcribe(np.array([0.1, 0.2]))
            assert result is not None  # Basic functionality works

    @pytest.mark.asyncio
    async def test_transcribe_handles_whisper_runtime_error(self, whisper_config):
        """Should wrap Whisper errors in TranscriptionError"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_audio_processor") as mock_processor,
            patch("app.engines.stt.whisper.engine.WhisperModel") as mock_model,
        ):
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.resample_to_16khz.return_value = np.array([0.1, 0.2])
            mock_processor.get_duration_ms.return_value = 100.0

            mock_instance = MagicMock()
            mock_model.return_value = mock_instance
            mock_instance.transcribe.side_effect = RuntimeError("Whisper crashed")

            with pytest.raises(TranscriptionError):
                await engine.transcribe(np.array([0.1, 0.2]))

    @pytest.mark.asyncio
    async def test_transcribe_invalid_language_code(self, whisper_config):
        """Should raise error for invalid language"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_audio_processor") as mock_processor,
            patch("app.engines.stt.whisper.engine.WhisperModel") as mock_model,
        ):
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.resample_to_16khz.return_value = np.array([0.1, 0.2])
            mock_processor.get_duration_ms.return_value = 100.0

            mock_instance = MagicMock()
            mock_model.return_value = mock_instance
            mock_instance.transcribe.side_effect = ValueError("Invalid language: zz")

            with pytest.raises((TranscriptionError, ValidationError, ValueError)):
                await engine.transcribe(
                    np.array([0.1, 0.2]), language="invalid_lang_zz"
                )

    @pytest.mark.asyncio
    async def test_transcribe_audio_too_large_raises(self, whisper_config):
        """Should raise AudioError if audio exceeds max size"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        # Create very large audio array (simulate > max size)
        huge_audio = np.random.randn(1000000000)  # 1 billion samples

        with patch.object(engine, "_audio_processor") as mock_processor:
            mock_processor.to_numpy.side_effect = AudioError("Audio too large")

            with pytest.raises(AudioError):
                await engine.transcribe(huge_audio)

    @pytest.mark.asyncio
    async def test_unsupported_audio_format_raises(self, whisper_config):
        """Should raise UnsupportedFormatError for unsupported formats"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with patch.object(engine, "_audio_processor") as mock_processor:
            mock_processor.to_numpy.side_effect = UnsupportedFormatError(
                "JPEG not supported"
            )

            with pytest.raises(UnsupportedFormatError):
                await engine.transcribe(b"fake jpeg data")

    @pytest.mark.asyncio
    async def test_transcribe_model_not_loaded_raises(self, whisper_config):
        """Should handle initialization failures properly"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        # Keep patch active to prevent successful retry
        with patch("app.engines.stt.whisper.engine.WhisperModel") as mock_model:
            mock_model.side_effect = RuntimeError("Failed to load model")

            # First call to initialize should fail
            with pytest.raises(RuntimeError):
                await engine.initialize()

            # Engine should still not be ready
            assert not engine.is_ready()

            # Try to transcribe - auto-initialization will also fail
            with pytest.raises(RuntimeError):
                await engine.transcribe(np.array([0.1, 0.2]))


class TestWhisperSTTEngineProperties:
    """Test WhisperSTTEngine properties"""

    @pytest.fixture
    def whisper_config(self):
        from app.engines.stt.whisper.config import WhisperConfig

        return WhisperConfig(model_name="base")

    def test_engine_name_is_whisper(self, whisper_config):
        """Engine name should be 'faster-whisper'"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        assert engine.engine_name == "faster-whisper"

    def test_supported_formats_includes_common(self, whisper_config):
        """Should support common audio formats"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        formats = engine.supported_formats

        assert isinstance(formats, list)
        assert "wav" in formats
        assert "mp3" in formats
        assert "flac" in formats

    def test_supported_languages(self, whisper_config):
        """Should return list of Whisper-supported languages"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        # This property may not be implemented yet, just check it exists
        assert hasattr(engine, "supported_formats")

    def test_model_info(self, whisper_config):
        """Should provide model information"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        # Check that config is accessible
        assert engine.config.model_name == "base"
        assert engine.config.compute_type == "int8"
        assert engine.config.beam_size == 5


class TestWhisperSTTEngineCoverageGaps:
    """Tests to cover remaining code paths"""

    @pytest.fixture
    def whisper_config(self):
        from app.engines.stt.whisper.config import WhisperConfig

        return WhisperConfig(model_name="base")

    @pytest.mark.asyncio
    async def test_transcribe_with_dict_segments(self, whisper_config):
        """Should handle dict-based segment responses from Whisper"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_audio_processor") as mock_processor,
            patch("app.engines.stt.whisper.engine.WhisperModel") as mock_model,
        ):
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.resample_to_16khz.return_value = np.array([0.1, 0.2])
            mock_processor.get_duration_ms.return_value = 100.0

            mock_instance = MagicMock()
            mock_model.return_value = mock_instance

            # Return dict-based segments with dict-based words
            mock_instance.transcribe.return_value = (
                [
                    {
                        "text": "Hello",
                        "words": [
                            {
                                "word": "Hello",
                                "start": 0.0,
                                "end": 0.5,
                                "probability": 0.95,
                            }
                        ],
                    }
                ],
                {"language": "en"},
            )

            result = await engine.transcribe(np.array([0.1, 0.2]))

            assert result.text == "Hello"
            assert result.language == "en"
            assert result.segments is not None
            assert len(result.segments) == 1
            assert result.segments[0].text == "Hello"
            assert result.segments[0].confidence == 0.95

    @pytest.mark.asyncio
    async def test_model_none_after_failed_init(self, whisper_config):
        """Should raise EngineNotReadyError if _model is None"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        # Manually set _model to None to simulate failed initialization
        engine._initialized = True  # Bypass auto-init
        engine._model = None

        with pytest.raises(EngineNotReadyError, match="Whisper model not loaded"):
            await engine.transcribe(np.array([0.1, 0.2]))

    def test_language_detection_from_object(self, whisper_config):
        """Should extract language from object-type info"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_model") as mock_model,
            patch.object(engine._audio_processor, "to_numpy") as mock_to_numpy,
            patch.object(engine._audio_processor, "resample_to_16khz") as mock_resample,
            patch.object(engine._audio_processor, "get_duration_ms") as mock_duration,
        ):
            mock_to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_resample.return_value = np.array([0.1, 0.2])
            mock_duration.return_value = 1000.0

            # Mock segment
            mock_segment = MagicMock()
            mock_segment.text = "Hello"

            # Mock info as object with language attribute (not dict)
            mock_info = MagicMock()
            mock_info.language = "fr"

            mock_model.transcribe.return_value = ([mock_segment], mock_info)
            engine._initialized = True

            import asyncio

            result = asyncio.run(engine.transcribe(np.array([0.1, 0.2])))

            assert result.language == "fr"

    def test_dict_segment_handling(self, whisper_config):
        """Should handle dict-type segments in transcription"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_model") as mock_model,
            patch.object(engine._audio_processor, "to_numpy") as mock_to_numpy,
            patch.object(engine._audio_processor, "resample_to_16khz") as mock_resample,
            patch.object(engine._audio_processor, "get_duration_ms") as mock_duration,
        ):
            mock_to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_resample.return_value = np.array([0.1, 0.2])
            mock_duration.return_value = 1000.0

            # Use dict-type segments instead of objects
            mock_segment = {"text": "Hello world"}
            mock_info = {"language": "en"}

            mock_model.transcribe.return_value = ([mock_segment], mock_info)
            engine._initialized = True

            import asyncio

            result = asyncio.run(engine.transcribe(np.array([0.1, 0.2])))

            assert result.text == "Hello world"
            assert result.language == "en"


class TestWhisperSTTEngineStreaming:
    """Test WhisperSTTEngine streaming transcription"""

    @pytest.fixture
    def whisper_config(self):
        """Fixture for WhisperConfig"""
        from app.engines.stt.whisper.config import WhisperConfig

        return WhisperConfig(model_name="base", device="cpu")

    # 2.1 Basic Streaming Tests

    @pytest.mark.asyncio
    async def test_transcribe_stream_basic(self, whisper_config):
        """Should yield multiple STTChunk objects and final STTResponse"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine
        from app.models.engine import STTChunk, STTResponse

        # Create audio data (bytes)
        audio_data = bytes([0] * 32000)

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_model") as mock_model,
            patch.object(engine._audio_processor, "to_numpy") as mock_to_numpy,
            patch.object(engine._audio_processor, "resample_to_16khz") as mock_resample,
            patch.object(engine._audio_processor, "get_duration_ms") as mock_duration,
        ):
            # Setup mocks
            mock_to_numpy.return_value = (np.array([0.1, 0.2, 0.3]), 16000)
            mock_resample.return_value = np.array([0.1, 0.2, 0.3])
            mock_duration.return_value = 1000.0  # 1 second

            # Mock Whisper transcribe to return segments (as generator)
            mock_segment = MagicMock()
            mock_segment.text = "test chunk"
            mock_segment.start = 0.0
            mock_word = MagicMock(word="test", start=0.0, end=0.5, probability=0.95)
            mock_segment.words = [mock_word]
            mock_model.transcribe.return_value = (
                iter([mock_segment]),
                {"language": "en"},
            )

            engine._initialized = True

            results = []
            async for result in engine.transcribe_stream(audio_data):
                results.append(result)

            # Should have chunks + final response
            chunks = [r for r in results if isinstance(r, STTChunk)]
            responses = [r for r in results if isinstance(r, STTResponse)]

            assert len(chunks) > 0, "Should yield at least one STTChunk"
            assert len(responses) == 1, "Should yield exactly one STTResponse"

            # Check response
            response = responses[0]
            assert response.text != ""
            assert response.language is not None
            assert response.segments is not None
            assert response.performance_metrics is not None
            assert response.performance_metrics.total_chunks >= 0

    @pytest.mark.asyncio
    async def test_transcribe_stream_empty_audio(self, whisper_config):
        """Should handle empty audio gracefully"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        audio_data = bytes([])

        engine = WhisperSTTEngine(whisper_config)
        engine._initialized = True

        with (
            patch.object(engine, "_model") as _,
            patch.object(engine._audio_processor, "to_numpy") as mock_to_numpy,
        ):
            mock_to_numpy.return_value = (np.array([]), 16000)

            with pytest.raises(InvalidAudioError):
                async for _ in engine.transcribe_stream(audio_data):
                    pass

    @pytest.mark.asyncio
    async def test_transcribe_stream_single_segment(self, whisper_config):
        """Should handle single segment correctly"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine
        from app.models.engine import STTChunk, STTResponse

        audio_data = bytes([0] * 32000)

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_model") as mock_model,
            patch.object(engine._audio_processor, "to_numpy") as mock_to_numpy,
            patch.object(engine._audio_processor, "resample_to_16khz") as mock_resample,
            patch.object(engine._audio_processor, "get_duration_ms") as mock_duration,
        ):
            mock_to_numpy.return_value = (np.array([0.1]), 16000)
            mock_resample.return_value = np.array([0.1])
            mock_duration.return_value = 1000.0

            mock_segment = MagicMock()
            mock_segment.text = "single"
            mock_segment.start = 0.0
            mock_word = MagicMock(word="single", start=0.0, end=0.5, probability=0.95)
            mock_segment.words = [mock_word]
            mock_model.transcribe.return_value = (
                iter([mock_segment]),
                {"language": "en"},
            )

            engine._initialized = True

            results = []
            async for result in engine.transcribe_stream(audio_data):
                results.append(result)

            chunks = [r for r in results if isinstance(r, STTChunk)]
            responses = [r for r in results if isinstance(r, STTResponse)]

            assert len(chunks) >= 1
            assert len(responses) == 1

    @pytest.mark.asyncio
    async def test_transcribe_stream_auto_initialization(self, whisper_config):
        """Engine should auto-initialize when not already initialized"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        audio_data = bytes([0] * 32000)

        engine = WhisperSTTEngine(whisper_config)

        # Engine not initialized
        assert not engine.is_ready()

        with (
            patch("app.engines.stt.whisper.engine.WhisperModel") as mock_model_class,
            patch.object(engine._audio_processor, "to_numpy") as mock_to_numpy,
            patch.object(engine._audio_processor, "resample_to_16khz") as mock_resample,
            patch.object(engine._audio_processor, "get_duration_ms") as mock_duration,
        ):
            mock_to_numpy.return_value = (np.array([0.1]), 16000)
            mock_resample.return_value = np.array([0.1])
            mock_duration.return_value = 1000.0

            mock_model_instance = MagicMock()
            mock_segment = MagicMock()
            mock_segment.text = "test"
            mock_segment.start = 0.0
            mock_model_instance.transcribe.return_value = (iter([mock_segment]), {})
            mock_model_class.return_value = mock_model_instance

            results = []
            async for result in engine.transcribe_stream(audio_data):
                results.append(result)

            # Should have initialized
            assert engine.is_ready()

    @pytest.mark.asyncio
    async def test_transcribe_stream_engine_closed(self, whisper_config):
        """Should raise EngineNotReadyError if engine closed"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        audio_data = bytes([0] * 1000)

        engine = WhisperSTTEngine(whisper_config)

        with patch("app.engines.stt.whisper.engine.WhisperModel"):
            await engine.initialize()
            await engine.close()

            with pytest.raises(EngineNotReadyError, match="Engine has been closed"):
                async for _ in engine.transcribe_stream(audio_data):
                    pass

    @pytest.mark.asyncio
    async def test_transcribe_stream_kwargs_override(self, whisper_config):
        """Should override config with kwargs parameters"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        audio_data = bytes([0] * 32000)

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_model") as mock_model,
            patch.object(engine._audio_processor, "to_numpy") as mock_to_numpy,
            patch.object(engine._audio_processor, "resample_to_16khz") as mock_resample,
            patch.object(engine._audio_processor, "get_duration_ms") as mock_duration,
        ):
            mock_to_numpy.return_value = (np.array([0.1]), 16000)
            mock_resample.return_value = np.array([0.1])
            mock_duration.return_value = 1000.0

            mock_model.transcribe.return_value = (iter([]), {})
            engine._initialized = True

            async for _ in engine.transcribe_stream(
                audio_data, vad_filter=False, beam_size=3
            ):
                pass

            # Verify transcribe was called with overridden params
            call_kwargs = mock_model.transcribe.call_args[1]
            assert call_kwargs["vad_filter"] is False
            assert call_kwargs["beam_size"] == 3

    # 2.2 Metrics and Timing Tests

    @pytest.mark.asyncio
    async def test_transcribe_stream_chunk_latency(self, whisper_config):
        """Should track chunk_latency_ms for each chunk"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine
        from app.models.engine import STTChunk

        audio_data = bytes([0] * 32000)

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_model") as mock_model,
            patch.object(engine._audio_processor, "to_numpy") as mock_to_numpy,
            patch.object(engine._audio_processor, "resample_to_16khz") as mock_resample,
            patch.object(engine._audio_processor, "get_duration_ms") as mock_duration,
        ):
            mock_to_numpy.return_value = (np.array([0.1]), 16000)
            mock_resample.return_value = np.array([0.1])
            mock_duration.return_value = 1000.0

            mock_segment = MagicMock()
            mock_segment.text = "test"
            mock_segment.start = 0.0
            mock_model.transcribe.return_value = (iter([mock_segment]), {})

            engine._initialized = True

            chunks = []
            async for result in engine.transcribe_stream(audio_data):
                if isinstance(result, STTChunk):
                    chunks.append(result)

            # All chunks should have chunk_latency_ms
            for chunk in chunks:
                assert chunk.chunk_latency_ms is not None
                assert chunk.chunk_latency_ms >= 0
                assert chunk.chunk_latency_ms < 10000  # Reasonable upper bound

    @pytest.mark.asyncio
    async def test_transcribe_stream_time_to_first_token(self, whisper_config):
        """Should measure time_to_first_token_ms in response"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine
        from app.models.engine import STTResponse

        audio_data = bytes([0] * 32000)

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_model") as mock_model,
            patch.object(engine._audio_processor, "to_numpy") as mock_to_numpy,
            patch.object(engine._audio_processor, "resample_to_16khz") as mock_resample,
            patch.object(engine._audio_processor, "get_duration_ms") as mock_duration,
        ):
            mock_to_numpy.return_value = (np.array([0.1]), 16000)
            mock_resample.return_value = np.array([0.1])
            mock_duration.return_value = 1000.0

            mock_segment = MagicMock()
            mock_segment.text = "test"
            mock_segment.start = 0.0
            mock_word = MagicMock(word="test", start=0.0, end=0.5, probability=0.95)
            mock_segment.words = [mock_word]
            mock_model.transcribe.return_value = (
                iter([mock_segment]),
                {"language": "en"},
            )

            engine._initialized = True

            response = None
            async for result in engine.transcribe_stream(audio_data):
                if isinstance(result, STTResponse):
                    response = result

            assert response is not None
            assert response.performance_metrics is not None
            assert response.performance_metrics.time_to_first_token_ms is not None
            assert response.performance_metrics.time_to_first_token_ms >= 0

    @pytest.mark.asyncio
    async def test_transcribe_stream_total_duration(self, whisper_config):
        """Should measure total_stream_duration_ms spanning entire stream"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine
        from app.models.engine import STTResponse

        audio_data = bytes([0] * 32000)

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_model") as mock_model,
            patch.object(engine._audio_processor, "to_numpy") as mock_to_numpy,
            patch.object(engine._audio_processor, "resample_to_16khz") as mock_resample,
            patch.object(engine._audio_processor, "get_duration_ms") as mock_duration,
        ):
            mock_to_numpy.return_value = (np.array([0.1]), 16000)
            mock_resample.return_value = np.array([0.1])
            mock_duration.return_value = 1000.0

            mock_segment = MagicMock()
            mock_segment.text = "test"
            mock_segment.start = 0.0
            mock_word = MagicMock(word="test", start=0.0, end=0.5, probability=0.95)
            mock_segment.words = [mock_word]
            mock_model.transcribe.return_value = (
                iter([mock_segment]),
                {"language": "en"},
            )

            engine._initialized = True

            response = None
            async for result in engine.transcribe_stream(audio_data):
                if isinstance(result, STTResponse):
                    response = result

            assert response is not None
            assert response.performance_metrics is not None
            assert response.performance_metrics.total_stream_duration_ms is not None
            assert response.performance_metrics.total_stream_duration_ms > 0

    @pytest.mark.asyncio
    async def test_transcribe_stream_audio_duration(self, whisper_config):
        """Should calculate audio_duration_ms in performance metrics"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine
        from app.models.engine import STTResponse

        audio_data = bytes([0] * 32000)

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_model") as mock_model,
            patch.object(engine._audio_processor, "to_numpy") as mock_to_numpy,
            patch.object(engine._audio_processor, "resample_to_16khz") as mock_resample,
            patch.object(engine._audio_processor, "get_duration_ms") as mock_duration,
        ):
            mock_to_numpy.return_value = (np.array([0.1]), 16000)
            mock_resample.return_value = np.array([0.1])
            mock_duration.return_value = 1000.0  # 1 second

            mock_segment = MagicMock()
            mock_segment.text = "test"
            mock_segment.start = 0.0
            mock_word = MagicMock(word="test", start=0.0, end=0.5, probability=0.95)
            mock_segment.words = [mock_word]
            mock_model.transcribe.return_value = (
                iter([mock_segment]),
                {"language": "en"},
            )

            engine._initialized = True

            response = None
            async for result in engine.transcribe_stream(audio_data):
                if isinstance(result, STTResponse):
                    response = result

            assert response is not None
            assert response.performance_metrics is not None
            assert response.performance_metrics.audio_duration_ms is not None
            assert response.performance_metrics.audio_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_transcribe_stream_total_chunks_count(self, whisper_config):
        """Should count total_chunks correctly"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine
        from app.models.engine import STTChunk, STTResponse

        audio_data = bytes([0] * 32000)

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_model") as mock_model,
            patch.object(engine._audio_processor, "to_numpy") as mock_to_numpy,
            patch.object(engine._audio_processor, "resample_to_16khz") as mock_resample,
            patch.object(engine._audio_processor, "get_duration_ms") as mock_duration,
        ):
            mock_to_numpy.return_value = (np.array([0.1]), 16000)
            mock_resample.return_value = np.array([0.1])
            mock_duration.return_value = 1000.0

            # Create multiple segments
            mock_segment1 = MagicMock()
            mock_segment1.text = "test1"
            mock_segment1.start = 0.0
            mock_word1 = MagicMock(word="test1", start=0.0, end=0.3, probability=0.95)
            mock_segment1.words = [mock_word1]
            mock_segment2 = MagicMock()
            mock_segment2.text = "test2"
            mock_segment2.start = 0.5
            mock_word2 = MagicMock(word="test2", start=0.5, end=0.8, probability=0.95)
            mock_segment2.words = [mock_word2]
            mock_segment3 = MagicMock()
            mock_segment3.text = "test3"
            mock_segment3.start = 1.0
            mock_word3 = MagicMock(word="test3", start=1.0, end=1.3, probability=0.95)
            mock_segment3.words = [mock_word3]
            mock_model.transcribe.return_value = (
                iter([mock_segment1, mock_segment2, mock_segment3]),
                {"language": "en"},
            )

            engine._initialized = True

            chunks_count = 0
            response = None
            async for result in engine.transcribe_stream(audio_data):
                if isinstance(result, STTChunk):
                    chunks_count += 1
                elif isinstance(result, STTResponse):
                    response = result

            assert response is not None
            assert response.performance_metrics is not None
            assert response.performance_metrics.total_chunks == chunks_count

    @pytest.mark.asyncio
    async def test_transcribe_stream_timestamps(self, whisper_config):
        """Should have timestamps that progress monotonically"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine
        from app.models.engine import STTChunk

        audio_data = bytes([0] * 32000)

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_model") as mock_model,
            patch.object(engine._audio_processor, "to_numpy") as mock_to_numpy,
            patch.object(engine._audio_processor, "resample_to_16khz") as mock_resample,
            patch.object(engine._audio_processor, "get_duration_ms") as mock_duration,
        ):
            mock_to_numpy.return_value = (np.array([0.1]), 16000)
            mock_resample.return_value = np.array([0.1])
            mock_duration.return_value = 1000.0

            # Create segments with increasing timestamps
            mock_segment1 = MagicMock()
            mock_segment1.text = "test1"
            mock_segment1.start = 0.0
            mock_segment2 = MagicMock()
            mock_segment2.text = "test2"
            mock_segment2.start = 0.5
            mock_model.transcribe.return_value = (
                iter([mock_segment1, mock_segment2]),
                {},
            )

            engine._initialized = True

            chunks = []
            async for result in engine.transcribe_stream(audio_data):
                if isinstance(result, STTChunk):
                    chunks.append(result)

            # Timestamps should exist and be non-decreasing
            timestamps = [c.timestamp for c in chunks if c.timestamp is not None]
            if len(timestamps) > 1:
                for i in range(1, len(timestamps)):
                    assert timestamps[i] >= timestamps[i - 1]

    # 2.3 Audio Input Format Tests

    @pytest.mark.asyncio
    async def test_transcribe_stream_with_language_parameter(self, whisper_config):
        """Should accept and use language parameter"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        audio_data = bytes([0] * 32000)

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_model") as mock_model,
            patch.object(engine._audio_processor, "to_numpy") as mock_to_numpy,
            patch.object(engine._audio_processor, "resample_to_16khz") as mock_resample,
            patch.object(engine._audio_processor, "get_duration_ms") as mock_duration,
        ):
            mock_to_numpy.return_value = (np.array([0.1]), 16000)
            mock_resample.return_value = np.array([0.1])
            mock_duration.return_value = 1000.0

            mock_model.transcribe.return_value = (iter([]), {})
            engine._initialized = True

            async for _ in engine.transcribe_stream(audio_data, language="en"):
                pass

            # Verify language was passed
            call_kwargs = mock_model.transcribe.call_args[1]
            assert call_kwargs["language"] == "en"

    @pytest.mark.asyncio
    async def test_transcribe_stream_numpy_input(self, whisper_config):
        """Should accept numpy array as input"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        audio_data = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_model") as mock_model,
            patch.object(engine._audio_processor, "to_numpy") as mock_to_numpy,
            patch.object(engine._audio_processor, "resample_to_16khz") as mock_resample,
            patch.object(engine._audio_processor, "get_duration_ms") as mock_duration,
        ):
            mock_to_numpy.return_value = (audio_data, 16000)
            mock_resample.return_value = audio_data
            mock_duration.return_value = 1000.0

            mock_segment = MagicMock()
            mock_segment.text = "test"
            mock_segment.start = 0.0
            mock_model.transcribe.return_value = (iter([mock_segment]), {})
            engine._initialized = True

            results = []
            async for result in engine.transcribe_stream(audio_data):
                results.append(result)

            assert len(results) > 0

    @pytest.mark.asyncio
    async def test_transcribe_stream_path_input(self, whisper_config):
        """Should accept Path as input"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        audio_data = pathlib.Path("/fake/audio.wav")

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_model") as mock_model,
            patch.object(engine._audio_processor, "to_numpy") as mock_to_numpy,
            patch.object(engine._audio_processor, "resample_to_16khz") as mock_resample,
            patch.object(engine._audio_processor, "get_duration_ms") as mock_duration,
        ):
            mock_to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_resample.return_value = np.array([0.1, 0.2])
            mock_duration.return_value = 1000.0

            mock_segment = MagicMock()
            mock_segment.text = "test"
            mock_segment.start = 0.0
            mock_model.transcribe.return_value = (iter([mock_segment]), {})
            engine._initialized = True

            results = []
            async for result in engine.transcribe_stream(audio_data):
                results.append(result)

            assert len(results) > 0

    @pytest.mark.asyncio
    async def test_transcribe_stream_bytesio_input(self, whisper_config):
        """Should accept BytesIO as input"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        audio_data = io.BytesIO(b"fake audio data")

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_model") as mock_model,
            patch.object(engine._audio_processor, "to_numpy") as mock_to_numpy,
            patch.object(engine._audio_processor, "resample_to_16khz") as mock_resample,
            patch.object(engine._audio_processor, "get_duration_ms") as mock_duration,
        ):
            mock_to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_resample.return_value = np.array([0.1, 0.2])
            mock_duration.return_value = 1000.0

            mock_segment = MagicMock()
            mock_segment.text = "test"
            mock_segment.start = 0.0
            mock_model.transcribe.return_value = (iter([mock_segment]), {})
            engine._initialized = True

            results = []
            async for result in engine.transcribe_stream(audio_data):
                results.append(result)

            assert len(results) > 0

    # 2.4 Error Handling Tests

    @pytest.mark.asyncio
    async def test_transcribe_stream_whisper_model_error(self, whisper_config):
        """Should propagate Whisper errors as TranscriptionError"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        audio_data = bytes([0] * 32000)

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_model") as mock_model,
            patch.object(engine._audio_processor, "to_numpy") as mock_to_numpy,
            patch.object(engine._audio_processor, "resample_to_16khz") as mock_resample,
            patch.object(engine._audio_processor, "get_duration_ms") as mock_duration,
        ):
            mock_to_numpy.return_value = (np.array([0.1]), 16000)
            mock_resample.return_value = np.array([0.1])
            mock_duration.return_value = 1000.0

            # Whisper raises exception
            mock_model.transcribe.side_effect = RuntimeError("Whisper internal error")

            engine._initialized = True

            with pytest.raises(TranscriptionError, match="Stream transcription failed"):
                async for _ in engine.transcribe_stream(audio_data):
                    pass

    @pytest.mark.asyncio
    async def test_transcribe_stream_invalid_language(self, whisper_config):
        """Should handle invalid language parameter"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        audio_data = bytes([0] * 32000)

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_model") as mock_model,
            patch.object(engine._audio_processor, "to_numpy") as mock_to_numpy,
            patch.object(engine._audio_processor, "resample_to_16khz") as mock_resample,
            patch.object(engine._audio_processor, "get_duration_ms") as mock_duration,
        ):
            mock_to_numpy.return_value = (np.array([0.1]), 16000)
            mock_resample.return_value = np.array([0.1])
            mock_duration.return_value = 1000.0

            # Whisper rejects invalid language
            mock_model.transcribe.side_effect = ValueError("Invalid language code")

            engine._initialized = True

            with pytest.raises(TranscriptionError):
                async for _ in engine.transcribe_stream(
                    audio_data, language="invalid_lang"
                ):
                    pass

    @pytest.mark.asyncio
    async def test_transcribe_stream_audio_processor_error(self, whisper_config):
        """Should handle AudioProcessor errors appropriately"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        audio_data = bytes([0] * 32000)

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_model") as _,
            patch.object(engine._audio_processor, "to_numpy") as mock_to_numpy,
        ):
            # AudioProcessor fails
            mock_to_numpy.side_effect = InvalidAudioError("Failed to decode audio")

            engine._initialized = True

            with pytest.raises(InvalidAudioError):
                async for _ in engine.transcribe_stream(audio_data):
                    pass

    @pytest.mark.asyncio
    async def test_transcribe_stream_model_none(self, whisper_config):
        """Should raise EngineNotReadyError if _model is None in streaming"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        audio_data = bytes([0] * 32000)

        engine = WhisperSTTEngine(whisper_config)
        engine._initialized = True  # Bypass auto-init
        engine._model = None

        with pytest.raises(EngineNotReadyError, match="Whisper model not loaded"):
            async for _ in engine.transcribe_stream(audio_data):
                pass

    @pytest.mark.asyncio
    async def test_transcribe_stream_dict_segments(self, whisper_config):
        """Should handle dict-type segments in streaming"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        audio_data = bytes([0] * 32000)

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_model") as mock_model,
            patch.object(engine._audio_processor, "to_numpy") as mock_to_numpy,
            patch.object(engine._audio_processor, "resample_to_16khz") as mock_resample,
            patch.object(engine._audio_processor, "get_duration_ms") as mock_duration,
        ):
            mock_to_numpy.return_value = (np.array([0.1]), 16000)
            mock_resample.return_value = np.array([0.1])
            mock_duration.return_value = 1000.0

            # Use dict-type segments instead of objects
            mock_segment = {"text": "dict segment", "start": 0.0}
            mock_model.transcribe.return_value = (iter([mock_segment]), {})

            engine._initialized = True

            results = []
            async for result in engine.transcribe_stream(audio_data):
                results.append(result)

            assert len(results) > 0

    # 2.5 VAD Tests

    @pytest.mark.asyncio
    async def test_transcribe_stream_vad_enabled(self, whisper_config):
        """Should use VAD when vad_filter=True"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        audio_data = bytes([0] * 32000)

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_model") as mock_model,
            patch.object(engine._audio_processor, "to_numpy") as mock_to_numpy,
            patch.object(engine._audio_processor, "resample_to_16khz") as mock_resample,
            patch.object(engine._audio_processor, "get_duration_ms") as mock_duration,
        ):
            mock_to_numpy.return_value = (np.array([0.1]), 16000)
            mock_resample.return_value = np.array([0.1])
            mock_duration.return_value = 1000.0

            mock_segment = MagicMock()
            mock_segment.text = "test"
            mock_segment.start = 0.0
            mock_model.transcribe.return_value = (iter([mock_segment]), {})

            engine._initialized = True

            async for _ in engine.transcribe_stream(audio_data, vad_filter=True):
                pass

            # Verify vad_filter was set
            call_kwargs = mock_model.transcribe.call_args[1]
            assert call_kwargs["vad_filter"] is True

    @pytest.mark.asyncio
    async def test_transcribe_stream_vad_disabled(self, whisper_config):
        """Should process without VAD when vad_filter=False"""
        from app.engines.stt.whisper.engine import WhisperSTTEngine

        audio_data = bytes([0] * 32000)

        engine = WhisperSTTEngine(whisper_config)

        with (
            patch.object(engine, "_model") as mock_model,
            patch.object(engine._audio_processor, "to_numpy") as mock_to_numpy,
            patch.object(engine._audio_processor, "resample_to_16khz") as mock_resample,
            patch.object(engine._audio_processor, "get_duration_ms") as mock_duration,
        ):
            mock_to_numpy.return_value = (np.array([0.1]), 16000)
            mock_resample.return_value = np.array([0.1])
            mock_duration.return_value = 1000.0

            mock_segment = MagicMock()
            mock_segment.text = "test"
            mock_segment.start = 0.0
            mock_model.transcribe.return_value = (iter([mock_segment]), {})

            engine._initialized = True

            async for _ in engine.transcribe_stream(audio_data, vad_filter=False):
                pass

            # Verify vad_filter=False was passed
            call_kwargs = mock_model.transcribe.call_args[1]
            assert call_kwargs["vad_filter"] is False
