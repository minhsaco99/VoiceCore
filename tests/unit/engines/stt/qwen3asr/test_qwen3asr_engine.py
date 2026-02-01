"""Tests for Qwen3-ASR STT engine"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from app.exceptions import (
    EngineNotReadyError,
    InvalidAudioError,
)
from app.models.engine import EngineConfig, STTResponse
from app.models.metrics import STTPerformanceMetrics


class TestQwen3ASRConfig:
    """Test Qwen3ASRConfig model"""

    def test_qwen3asr_config_extends_engine_config(self):
        """Qwen3ASRConfig should be subclass of EngineConfig"""
        from app.engines.stt.qwen3asr.config import Qwen3ASRConfig

        # Check inheritance
        assert issubclass(Qwen3ASRConfig, EngineConfig)

        # Create instance and verify it's both
        config = Qwen3ASRConfig(model_name="Qwen/Qwen3-ASR-1.7B")
        assert isinstance(config, Qwen3ASRConfig)
        assert isinstance(config, EngineConfig)

    def test_qwen3asr_config_inherits_base_fields(self):
        """Qwen3ASRConfig should inherit all EngineConfig fields"""
        from app.engines.stt.qwen3asr.config import Qwen3ASRConfig

        config = Qwen3ASRConfig(
            model_name="Qwen/Qwen3-ASR-1.7B",
            device="cuda",
            max_workers=4,
            timeout_seconds=600,
        )

        # Check inherited fields are accessible
        assert config.model_name == "Qwen/Qwen3-ASR-1.7B"
        assert config.device == "cuda"
        assert config.max_workers == 4
        assert config.timeout_seconds == 600

    def test_qwen3asr_config_default_values(self):
        """Qwen3ASRConfig should have correct default values"""
        from app.engines.stt.qwen3asr.config import Qwen3ASRConfig

        config = Qwen3ASRConfig(model_name="Qwen/Qwen3-ASR-1.7B")

        # Check base defaults
        assert config.device == "cpu"
        assert config.max_workers == 1
        assert config.timeout_seconds == 300

        # Check Qwen3-ASR specific defaults
        assert config.gpu_memory_utilization == 0.7
        assert config.max_inference_batch_size == 32
        assert config.max_new_tokens == 512
        assert config.forced_aligner is None
        assert config.language is None

    def test_qwen3asr_config_valid_models(self):
        """Qwen3ASRConfig should accept valid Qwen3-ASR model names"""
        from app.engines.stt.qwen3asr.config import Qwen3ASRConfig

        valid_models = [
            "Qwen/Qwen3-ASR-1.7B",
            "Qwen/Qwen3-ASR-0.6B",
            "/local/path/to/model",
        ]
        for model in valid_models:
            config = Qwen3ASRConfig(model_name=model)
            assert config.model_name == model

    def test_qwen3asr_config_accepts_local_path(self):
        """Should accept local paths for model loading"""
        from app.engines.stt.qwen3asr.config import Qwen3ASRConfig

        config = Qwen3ASRConfig(model_name="/opt/models/qwen3")
        assert config.model_name == "/opt/models/qwen3"

    def test_qwen3asr_config_invalid_model_raises(self):
        """Should raise ValidationError for invalid model names"""
        from app.engines.stt.qwen3asr.config import Qwen3ASRConfig

        with pytest.raises(ValidationError):
            Qwen3ASRConfig(model_name="Invalid/Model")

    def test_qwen3asr_config_gpu_memory_utilization_validation(self):
        """Should validate gpu_memory_utilization range"""
        from app.engines.stt.qwen3asr.config import Qwen3ASRConfig

        # Valid
        Qwen3ASRConfig(model_name="Qwen/Qwen3-ASR-1.7B", gpu_memory_utilization=0.1)
        Qwen3ASRConfig(model_name="Qwen/Qwen3-ASR-1.7B", gpu_memory_utilization=0.95)

        # Invalid
        with pytest.raises(ValidationError):
            Qwen3ASRConfig(
                model_name="Qwen/Qwen3-ASR-1.7B", gpu_memory_utilization=0.05
            )
        with pytest.raises(ValidationError):
            Qwen3ASRConfig(model_name="Qwen/Qwen3-ASR-1.7B", gpu_memory_utilization=1.0)

    def test_qwen3asr_config_max_new_tokens_validation(self):
        """Should validate max_new_tokens > 0"""
        from app.engines.stt.qwen3asr.config import Qwen3ASRConfig

        with pytest.raises(ValidationError):
            Qwen3ASRConfig(model_name="Qwen/Qwen3-ASR-1.7B", max_new_tokens=0)

        # Invalid: negative
        with pytest.raises(ValidationError):
            Qwen3ASRConfig(model_name="Qwen/Qwen3-ASR-1.7B", max_new_tokens=-1)

    def test_qwen3asr_config_optional_language(self):
        """Qwen3ASRConfig language field should be optional"""
        from app.engines.stt.qwen3asr.config import Qwen3ASRConfig

        # Without language (default None)
        config1 = Qwen3ASRConfig(model_name="Qwen/Qwen3-ASR-1.7B")
        assert config1.language is None

        # With language
        config2 = Qwen3ASRConfig(model_name="Qwen/Qwen3-ASR-1.7B", language="English")
        assert config2.language == "English"

        config3 = Qwen3ASRConfig(
            model_name="Qwen/Qwen3-ASR-1.7B", language="Vietnamese"
        )
        assert config3.language == "Vietnamese"


class TestQwen3ASREngineLifecycle:
    """Test Qwen3ASREngine lifecycle management"""

    @pytest.fixture
    def qwen3asr_config(self):
        """Fixture for Qwen3ASRConfig"""
        from app.engines.stt.qwen3asr.config import Qwen3ASRConfig

        return Qwen3ASRConfig(model_name="Qwen/Qwen3-ASR-1.7B", device="cuda")

    def test_qwen3asr_engine_not_initialized_on_creation(self, qwen3asr_config):
        """Engine should not be ready immediately after creation"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine

        engine = Qwen3ASREngine(qwen3asr_config)

        assert not engine.is_ready()

    @pytest.mark.asyncio
    async def test_qwen3asr_engine_explicit_initialization(self, qwen3asr_config):
        """Engine should initialize when explicitly called"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine

        engine = Qwen3ASREngine(qwen3asr_config)

        with patch("qwen_asr.Qwen3ASRModel") as mock_model:
            mock_model.LLM.return_value = MagicMock()
            await engine.initialize()

            assert engine.is_ready()
            mock_model.LLM.assert_called_once()

    @pytest.mark.asyncio
    async def test_qwen3asr_engine_initialization_calls_llm_method(
        self, qwen3asr_config
    ):
        """Engine should use LLM() constructor for vLLM backend"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine

        engine = Qwen3ASREngine(qwen3asr_config)

        with patch("qwen_asr.Qwen3ASRModel") as mock_model:
            mock_model.LLM.return_value = MagicMock()
            await engine.initialize()

            mock_model.LLM.assert_called_once_with(
                model="Qwen/Qwen3-ASR-1.7B",
                gpu_memory_utilization=0.7,
                max_inference_batch_size=32,
                max_new_tokens=512,
                forced_aligner=None,
            )

    @pytest.mark.asyncio
    async def test_qwen3asr_engine_auto_initialization_on_transcribe(
        self, qwen3asr_config
    ):
        """Engine should auto-initialize on first transcribe() call"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine

        engine = Qwen3ASREngine(qwen3asr_config)

        assert not engine.is_ready()

        with (
            patch("qwen_asr.Qwen3ASRModel") as mock_model,
            patch.object(engine, "_audio_processor") as mock_processor,
        ):
            mock_instance = MagicMock()
            mock_model.LLM.return_value = mock_instance

            # Mock transcribe result
            mock_result = MagicMock()
            mock_result.text = "Hello"
            mock_result.language = "English"
            mock_instance.transcribe.return_value = [mock_result]

            # Mock audio processor
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.get_duration_ms.return_value = 100.0

            result = await engine.transcribe(np.array([0.1, 0.2]))

            assert engine.is_ready()
            assert result.text is not None

    @pytest.mark.asyncio
    async def test_qwen3asr_engine_is_ready_after_init(self, qwen3asr_config):
        """is_ready() should return True after initialization"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine

        engine = Qwen3ASREngine(qwen3asr_config)

        with patch("qwen_asr.Qwen3ASRModel") as mock_model:
            mock_model.LLM.return_value = MagicMock()
            await engine.initialize()
            assert engine.is_ready() is True

    @pytest.mark.asyncio
    async def test_qwen3asr_engine_close_cleanup(self, qwen3asr_config):
        """close() should release resources"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine

        engine = Qwen3ASREngine(qwen3asr_config)

        with patch("qwen_asr.Qwen3ASRModel") as mock_model:
            mock_model.LLM.return_value = MagicMock()
            await engine.initialize()
            assert engine.is_ready()

            await engine.close()

            assert not engine.is_ready()

    @pytest.mark.asyncio
    async def test_qwen3asr_engine_cannot_use_after_close(self, qwen3asr_config):
        """Should raise EngineNotReadyError after close"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine

        engine = Qwen3ASREngine(qwen3asr_config)

        with patch("qwen_asr.Qwen3ASRModel") as mock_model:
            mock_model.LLM.return_value = MagicMock()
            await engine.initialize()
            await engine.close()

            with pytest.raises(EngineNotReadyError):
                await engine.transcribe(np.array([0.1, 0.2]))

    @pytest.mark.asyncio
    async def test_qwen3asr_engine_context_manager(self, qwen3asr_config):
        """Engine should work with async with statement"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine

        engine = Qwen3ASREngine(qwen3asr_config)

        with (
            patch("qwen_asr.Qwen3ASRModel") as mock_model,
            patch.object(engine, "_audio_processor") as mock_processor,
        ):
            mock_instance = MagicMock()
            mock_model.LLM.return_value = mock_instance

            mock_result = MagicMock()
            mock_result.text = "Test"
            mock_result.language = "English"
            mock_instance.transcribe.return_value = [mock_result]

            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.get_duration_ms.return_value = 100.0

            async with engine:
                assert engine.is_ready()
                result = await engine.transcribe(np.array([0.1, 0.2]))
                assert result.text is not None

            # Should be closed after context
            assert not engine.is_ready()

    @pytest.mark.asyncio
    async def test_qwen3asr_engine_init_fails_without_package(self, qwen3asr_config):
        """Should raise EngineNotReadyError if qwen-asr package not installed"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine

        engine = Qwen3ASREngine(qwen3asr_config)

        # Simulate qwen_asr package missing
        with patch.dict("sys.modules", {"qwen_asr": None}):
            with pytest.raises(EngineNotReadyError) as exc_info:
                await engine.initialize()

            assert "qwen-asr package not installed" in str(exc_info.value)


class TestQwen3ASREngineTranscription:
    """Test Qwen3ASREngine transcription functionality"""

    @pytest.fixture
    def qwen3asr_config(self):
        from app.engines.stt.qwen3asr.config import Qwen3ASRConfig

        return Qwen3ASRConfig(model_name="Qwen/Qwen3-ASR-1.7B")

    @pytest.fixture
    def mock_qwen3asr_model(self):
        """Mock Qwen3-ASR model that returns fake transcription"""
        with patch("qwen_asr.Qwen3ASRModel") as mock:
            mock_instance = MagicMock()
            mock.LLM.return_value = mock_instance

            mock_result = MagicMock()
            mock_result.text = "Hello world"
            mock_result.language = "English"
            mock_instance.transcribe.return_value = [mock_result]

            yield mock_instance

    @pytest.mark.asyncio
    async def test_transcribe_returns_stt_response(
        self, qwen3asr_config, mock_qwen3asr_model
    ):
        """Should return STTResponse with text"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine

        engine = Qwen3ASREngine(qwen3asr_config)

        with patch.object(engine, "_audio_processor") as mock_processor:
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.get_duration_ms.return_value = 100.0

            result = await engine.transcribe(np.array([0.1, 0.2]))

            assert isinstance(result, STTResponse)
            assert result.text == "Hello world"

    @pytest.mark.asyncio
    async def test_transcribe_with_bytes_input(
        self, qwen3asr_config, mock_qwen3asr_model
    ):
        """Should accept bytes input (WAV format)"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine

        engine = Qwen3ASREngine(qwen3asr_config)
        fake_wav_bytes = b"fake wav data"

        with patch.object(engine, "_audio_processor") as mock_processor:
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.get_duration_ms.return_value = 100.0

            result = await engine.transcribe(fake_wav_bytes)

            assert isinstance(result, STTResponse)
            assert result.text is not None

    @pytest.mark.asyncio
    async def test_transcribe_with_numpy_input(
        self, qwen3asr_config, mock_qwen3asr_model
    ):
        """Should accept numpy array input"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine

        engine = Qwen3ASREngine(qwen3asr_config)
        fake_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        with patch.object(engine, "_audio_processor") as mock_processor:
            mock_processor.to_numpy.return_value = (fake_audio, 16000)
            mock_processor.get_duration_ms.return_value = 100.0

            result = await engine.transcribe(fake_audio)

            assert isinstance(result, STTResponse)
            assert result.text is not None

    @pytest.mark.asyncio
    async def test_transcribe_with_language_hint(
        self, qwen3asr_config, mock_qwen3asr_model
    ):
        """Should pass language hint to Qwen3-ASR"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine

        engine = Qwen3ASREngine(qwen3asr_config)

        with patch.object(engine, "_audio_processor") as mock_processor:
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.get_duration_ms.return_value = 100.0

            await engine.transcribe(np.array([0.1, 0.2]), language="vi")

            # Verify transcribe was called with language
            call_args = mock_qwen3asr_model.transcribe.call_args
            assert call_args is not None
            assert call_args.kwargs.get("language") == "Vietnamese"

    @pytest.mark.asyncio
    async def test_transcribe_empty_audio_raises(
        self, qwen3asr_config, mock_qwen3asr_model
    ):
        """Should raise InvalidAudioError for empty input"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine

        engine = Qwen3ASREngine(qwen3asr_config)

        with patch.object(engine, "_audio_processor") as mock_processor:
            mock_processor.to_numpy.return_value = (np.array([]), 16000)

            with pytest.raises(InvalidAudioError):
                await engine.transcribe(np.array([]))

    @pytest.mark.asyncio
    async def test_transcribe_detects_language(
        self, qwen3asr_config, mock_qwen3asr_model
    ):
        """Should return detected language in response"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine

        engine = Qwen3ASREngine(qwen3asr_config)

        mock_result = MagicMock()
        mock_result.text = "Xin chào"
        mock_result.language = "Vietnamese"
        mock_qwen3asr_model.transcribe.return_value = [mock_result]

        with patch.object(engine, "_audio_processor") as mock_processor:
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.get_duration_ms.return_value = 100.0

            result = await engine.transcribe(np.array([0.1, 0.2]))

            assert result.language == "Vietnamese"

    @pytest.mark.asyncio
    async def test_transcribe_handles_empty_result(
        self, qwen3asr_config, mock_qwen3asr_model
    ):
        """Should handle empty transcription result gracefully"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine

        engine = Qwen3ASREngine(qwen3asr_config)

        # Mock empty result
        mock_qwen3asr_model.transcribe.return_value = []

        with patch.object(engine, "_audio_processor") as mock_processor:
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.get_duration_ms.return_value = 100.0

            result = await engine.transcribe(np.array([0.1, 0.2]))

            assert result.text == ""

    @pytest.mark.asyncio
    async def test_transcribe_returns_segments_with_timestamps(
        self, qwen3asr_config, mock_qwen3asr_model
    ):
        """Should extract and return segments when timestamps are available"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine
        from app.models.engine import Segment

        engine = Qwen3ASREngine(qwen3asr_config)

        mock_result = MagicMock()
        mock_result.text = "Hello world"
        mock_result.language = "English"

        # Mock qwen-asr ForcedAlignItem mocks
        item1 = MagicMock()
        item1.text = "Hello"
        item1.start_time = 0.0
        item1.end_time = 0.5

        item2 = MagicMock()
        item2.text = "world"
        item2.start_time = 0.5
        item2.end_time = 1.0

        mock_result.time_stamps = [item1, item2]
        mock_qwen3asr_model.transcribe.return_value = [mock_result]

        with patch.object(engine, "_audio_processor") as mock_processor:
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.get_duration_ms.return_value = 1000.0

            result = await engine.transcribe(np.array([0.1, 0.2]))

            assert result.segments is not None
            assert len(result.segments) == 2
            assert isinstance(result.segments[0], Segment)
            assert result.segments[0].text == "Hello"
            assert result.segments[0].start == 0.0
            assert result.segments[0].end == 0.5
            assert result.segments[1].text == "world"


class TestQwen3ASREngineMetrics:
    """Test Qwen3ASREngine performance metrics calculation"""

    @pytest.fixture
    def qwen3asr_config(self):
        from app.engines.stt.qwen3asr.config import Qwen3ASRConfig

        return Qwen3ASRConfig(model_name="Qwen/Qwen3-ASR-1.7B")

    @pytest.fixture
    def mock_qwen3asr_model(self):
        with patch("qwen_asr.Qwen3ASRModel") as mock:
            mock_instance = MagicMock()
            mock.LLM.return_value = mock_instance

            mock_result = MagicMock()
            mock_result.text = "Test"
            mock_result.language = "English"
            mock_instance.transcribe.return_value = [mock_result]

            yield mock_instance

    @pytest.mark.asyncio
    async def test_transcribe_returns_performance_metrics(
        self, qwen3asr_config, mock_qwen3asr_model
    ):
        """STTResponse.performance_metrics should not be None"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine

        engine = Qwen3ASREngine(qwen3asr_config)

        with patch.object(engine, "_audio_processor") as mock_processor:
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.get_duration_ms.return_value = 100.0

            result = await engine.transcribe(np.array([0.1, 0.2]))

            assert result.performance_metrics is not None
            assert isinstance(result.performance_metrics, STTPerformanceMetrics)

    @pytest.mark.asyncio
    async def test_metrics_latency_ms_calculated(
        self, qwen3asr_config, mock_qwen3asr_model
    ):
        """Latency should be calculated as end_time - start_time"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine

        engine = Qwen3ASREngine(qwen3asr_config)

        with patch.object(engine, "_audio_processor") as mock_processor:
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.get_duration_ms.return_value = 100.0

            result = await engine.transcribe(np.array([0.1, 0.2]))

            assert result.performance_metrics.latency_ms > 0

    @pytest.mark.asyncio
    async def test_metrics_processing_time_ms_calculated(
        self, qwen3asr_config, mock_qwen3asr_model
    ):
        """Processing time should be measured"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine

        engine = Qwen3ASREngine(qwen3asr_config)

        with patch.object(engine, "_audio_processor") as mock_processor:
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.get_duration_ms.return_value = 100.0

            result = await engine.transcribe(np.array([0.1, 0.2]))

            assert result.performance_metrics.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_metrics_audio_duration_ms_calculated(
        self, qwen3asr_config, mock_qwen3asr_model
    ):
        """Audio duration should be calculated from audio length"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine

        engine = Qwen3ASREngine(qwen3asr_config)

        with patch.object(engine, "_audio_processor") as mock_processor:
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.get_duration_ms.return_value = 500.0

            result = await engine.transcribe(np.array([0.1, 0.2]))

            assert result.performance_metrics.audio_duration_ms == 500.0

    @pytest.mark.asyncio
    async def test_metrics_real_time_factor_calculated(
        self, qwen3asr_config, mock_qwen3asr_model
    ):
        """RTF should be calculated as processing_time / audio_duration"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine

        engine = Qwen3ASREngine(qwen3asr_config)

        with patch.object(engine, "_audio_processor") as mock_processor:
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.get_duration_ms.return_value = 1000.0

            result = await engine.transcribe(np.array([0.1, 0.2]))

            assert result.performance_metrics.real_time_factor is not None


class TestQwen3ASREngineStreaming:
    """Test Qwen3ASREngine streaming transcription"""

    @pytest.fixture
    def qwen3asr_config(self):
        from app.engines.stt.qwen3asr.config import Qwen3ASRConfig

        return Qwen3ASRConfig(model_name="Qwen/Qwen3-ASR-1.7B")

    @pytest.fixture
    def mock_qwen3asr_model(self):
        with patch("qwen_asr.Qwen3ASRModel") as mock:
            mock_instance = MagicMock()
            mock.LLM.return_value = mock_instance

            mock_result = MagicMock()
            mock_result.text = "Streaming test"
            mock_result.language = "English"
            mock_instance.transcribe.return_value = [mock_result]

            yield mock_instance

    @pytest.mark.asyncio
    async def test_transcribe_stream_yields_chunks(
        self, qwen3asr_config, mock_qwen3asr_model
    ):
        """transcribe_stream should yield STTChunk and STTResponse"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine
        from app.models.engine import STTChunk

        engine = Qwen3ASREngine(qwen3asr_config)

        with patch.object(engine, "_audio_processor") as mock_processor:
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.get_duration_ms.return_value = 100.0

            chunks = []
            async for item in engine.transcribe_stream(np.array([0.1, 0.2])):
                chunks.append(item)

            # Should have at least one chunk and final response
            assert len(chunks) >= 2
            assert isinstance(chunks[0], STTChunk)
            assert isinstance(chunks[-1], STTResponse)

    @pytest.mark.asyncio
    async def test_transcribe_stream_final_response_has_text(
        self, qwen3asr_config, mock_qwen3asr_model
    ):
        """Final response should contain complete text"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine

        engine = Qwen3ASREngine(qwen3asr_config)

        with patch.object(engine, "_audio_processor") as mock_processor:
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.get_duration_ms.return_value = 100.0

            chunks = []
            async for item in engine.transcribe_stream(np.array([0.1, 0.2])):
                chunks.append(item)

            final_response = chunks[-1]
            assert final_response.text == "Streaming test"

    @pytest.mark.asyncio
    async def test_transcribe_stream_metrics_include_streaming_fields(
        self, qwen3asr_config, mock_qwen3asr_model
    ):
        """Streaming metrics should include time_to_first_token_ms"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine

        engine = Qwen3ASREngine(qwen3asr_config)

        with patch.object(engine, "_audio_processor") as mock_processor:
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.get_duration_ms.return_value = 100.0

            chunks = []
            async for item in engine.transcribe_stream(np.array([0.1, 0.2])):
                chunks.append(item)

            final_response = chunks[-1]
            metrics = final_response.performance_metrics

            assert metrics.time_to_first_token_ms is not None
            assert metrics.total_stream_duration_ms is not None
            assert metrics.total_chunks is not None

    @pytest.mark.asyncio
    async def test_transcribe_stream_returns_segments_with_timestamps(
        self, qwen3asr_config, mock_qwen3asr_model
    ):
        """Should extract and return segments in streaming mode"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine

        engine = Qwen3ASREngine(qwen3asr_config)

        mock_result = MagicMock()
        mock_result.text = "Hello world"
        mock_result.language = "English"

        # Mock qwen-asr ForcedAlignItem mocks
        item1 = MagicMock()
        item1.text = "Hello"
        item1.start_time = 0.0
        item1.end_time = 0.5

        item2 = MagicMock()
        item2.text = "world"
        item2.start_time = 0.5
        item2.end_time = 1.0

        mock_result.time_stamps = [item1, item2]
        mock_qwen3asr_model.transcribe.return_value = [mock_result]

        with patch.object(engine, "_audio_processor") as mock_processor:
            mock_processor.to_numpy.return_value = (np.array([0.1, 0.2]), 16000)
            mock_processor.get_duration_ms.return_value = 1000.0

            chunks = []
            async for item in engine.transcribe_stream(np.array([0.1, 0.2])):
                chunks.append(item)

            final_response = chunks[-1]
            assert final_response.segments is not None
            assert len(final_response.segments) == 2
            assert final_response.segments[0].text == "Hello"
            assert final_response.segments[0].start == 0.0
            assert final_response.segments[0].end == 0.5

    @pytest.mark.asyncio
    async def test_transcribe_stream_empty_audio_raises(
        self, qwen3asr_config, mock_qwen3asr_model
    ):
        """Should raise InvalidAudioError for empty input in streaming"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine

        engine = Qwen3ASREngine(qwen3asr_config)

        with patch.object(engine, "_audio_processor") as mock_processor:
            mock_processor.to_numpy.return_value = (np.array([]), 16000)

            with pytest.raises(InvalidAudioError):
                async for _ in engine.transcribe_stream(np.array([])):
                    pass


class TestQwen3ASREngineLanguageMapping:
    """Test Qwen3ASREngine language code mapping"""

    @pytest.fixture
    def qwen3asr_config(self):
        from app.engines.stt.qwen3asr.config import Qwen3ASRConfig

        return Qwen3ASRConfig(model_name="Qwen/Qwen3-ASR-1.7B")

    def test_language_mapping_short_codes(self, qwen3asr_config):
        """Should map short language codes to full names"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine

        engine = Qwen3ASREngine(qwen3asr_config)

        # Test common mappings
        assert engine._map_language("en") == "English"
        assert engine._map_language("zh") == "Chinese"
        assert engine._map_language("vi") == "Vietnamese"
        assert engine._map_language("ja") == "Japanese"
        assert engine._map_language("ko") == "Korean"
        assert engine._map_language("fr") == "French"
        assert engine._map_language("de") == "German"
        assert engine._map_language("es") == "Spanish"

    def test_language_mapping_case_insensitive(self, qwen3asr_config):
        """Language mapping should be case insensitive"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine

        engine = Qwen3ASREngine(qwen3asr_config)

        assert engine._map_language("EN") == "English"
        assert engine._map_language("Vi") == "Vietnamese"
        assert engine._map_language("ZH") == "Chinese"

    def test_language_mapping_full_names_passthrough(self, qwen3asr_config):
        """Full language names should pass through unchanged"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine

        engine = Qwen3ASREngine(qwen3asr_config)

        assert engine._map_language("English") == "English"
        assert engine._map_language("Vietnamese") == "Vietnamese"
        assert engine._map_language("Chinese") == "Chinese"

    def test_language_mapping_none_returns_none(self, qwen3asr_config):
        """None language should return None"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine

        engine = Qwen3ASREngine(qwen3asr_config)

        assert engine._map_language(None) is None


class TestQwen3ASREngineSupportedFormats:
    """Test Qwen3ASREngine supported audio formats"""

    @pytest.fixture
    def qwen3asr_config(self):
        from app.engines.stt.qwen3asr.config import Qwen3ASRConfig

        return Qwen3ASRConfig(model_name="Qwen/Qwen3-ASR-1.7B")

    def test_supported_formats_property(self, qwen3asr_config):
        """Should return list of supported audio formats"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine

        engine = Qwen3ASREngine(qwen3asr_config)

        formats = engine.supported_formats

        assert isinstance(formats, list)
        assert "wav" in formats
        assert "mp3" in formats
        assert "flac" in formats

    def test_engine_name_property(self, qwen3asr_config):
        """Should return correct engine name"""
        from app.engines.stt.qwen3asr.engine import Qwen3ASREngine

        engine = Qwen3ASREngine(qwen3asr_config)

        assert engine.engine_name == "qwen3-asr"
