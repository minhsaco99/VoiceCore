import pytest
from pydantic import ValidationError

from app.models.engine import EngineConfig, STTChunk, STTOutput, TTSChunk, TTSOutput
from app.models.metrics import (
    QualityMetrics,
    STTInvokePerformanceMetrics,
    STTStreamPerformanceMetrics,
    TTSInvokePerformanceMetrics,
    TTSStreamPerformanceMetrics,
)


class TestEngineConfig:
    """Test EngineConfig model"""

    def test_create_config_with_required_fields(self):
        """Should create config with model_name"""
        config = EngineConfig(model_name="whisper-base")

        assert config.model_name == "whisper-base"
        assert config.device == "cpu"  # Default
        assert config.max_workers == 1  # Default
        assert config.timeout_seconds == 300  # Default
        assert config.engine_params == {}  # Default

    def test_create_config_with_all_fields(self):
        """Should create config with all fields"""
        config = EngineConfig(
            model_name="whisper-large",
            device="cuda",
            max_workers=4,
            timeout_seconds=600,
            engine_params={"language": "en", "temperature": 0.0},
        )

        assert config.model_name == "whisper-large"
        assert config.device == "cuda"
        assert config.max_workers == 4
        assert config.timeout_seconds == 600
        assert config.engine_params == {"language": "en", "temperature": 0.0}

    def test_config_validates_device(self):
        """Device should only accept cpu, cuda, or mps"""
        EngineConfig(model_name="test", device="cpu")
        EngineConfig(model_name="test", device="cuda")
        EngineConfig(model_name="test", device="mps")

        with pytest.raises(ValidationError):
            EngineConfig(model_name="test", device="gpu")  # Invalid


class TestSTTOutput:
    """Test STTOutput model for invoke mode"""

    def test_create_stt_output(self):
        """Should create STTOutput with STT-specific invoke metrics"""
        perf = STTInvokePerformanceMetrics(
            latency_ms=100.0,
            processing_time_ms=95.0,
            real_time_factor=0.5,  # STT-specific
        )
        qual = QualityMetrics(
            confidence_score=0.95,
            word_error_rate=0.03,  # STT-specific
        )

        output = STTOutput(
            text="Hello world",
            quality_metrics=qual,
            performance_metrics=perf,
        )

        assert output.text == "Hello world"
        assert output.language is None
        assert output.segments is None
        assert output.quality_metrics == qual
        assert output.performance_metrics == perf
        assert output.performance_metrics.real_time_factor == 0.5

    def test_stt_output_with_language_and_segments(self):
        """Should create STTOutput with optional fields"""
        perf = STTInvokePerformanceMetrics(latency_ms=100.0, processing_time_ms=95.0)
        qual = QualityMetrics(confidence_score=0.95)
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
            {"start": 1.0, "end": 2.0, "text": "world"},
        ]

        output = STTOutput(
            text="Hello world",
            language="en",
            segments=segments,
            quality_metrics=qual,
            performance_metrics=perf,
        )

        assert output.language == "en"
        assert output.segments == segments

    def test_stt_output_requires_text(self):
        """Should require text field"""
        perf = STTInvokePerformanceMetrics(latency_ms=100.0, processing_time_ms=95.0)
        qual = QualityMetrics()

        with pytest.raises(ValidationError):
            STTOutput(
                quality_metrics=qual,
                performance_metrics=perf,
            )


class TestTTSOutput:
    """Test TTSOutput model for invoke mode"""

    def test_create_tts_output(self):
        """Should create TTSOutput with TTS-specific invoke metrics"""
        perf = TTSInvokePerformanceMetrics(
            latency_ms=200.0,
            processing_time_ms=180.0,
            characters_per_second=50.0,  # TTS-specific
        )
        qual = QualityMetrics(
            mean_opinion_score=4.2,  # TTS-specific
        )

        output = TTSOutput(
            audio_data=b"fake audio bytes",
            sample_rate=22050,
            duration_seconds=2.5,
            quality_metrics=qual,
            performance_metrics=perf,
        )

        assert output.audio_data == b"fake audio bytes"
        assert output.sample_rate == 22050
        assert output.format == "wav"  # Default
        assert output.duration_seconds == 2.5
        assert output.quality_metrics == qual
        assert output.performance_metrics == perf
        assert output.performance_metrics.characters_per_second == 50.0

    def test_tts_output_with_custom_format(self):
        """Should create TTSOutput with custom format"""
        perf = TTSInvokePerformanceMetrics(latency_ms=200.0, processing_time_ms=180.0)
        qual = QualityMetrics()

        output = TTSOutput(
            audio_data=b"fake mp3 bytes",
            sample_rate=44100,
            format="mp3",
            duration_seconds=3.0,
            quality_metrics=qual,
            performance_metrics=perf,
        )

        assert output.format == "mp3"


class TestSTTChunk:
    """Test STTChunk model for streaming"""

    def test_create_stt_chunk_minimal(self):
        """Should create STTChunk with STT-specific stream metrics"""
        stream_metrics = STTStreamPerformanceMetrics(
            time_to_first_token_ms=280.0,  # STT TTFT
            chunk_latency_ms=45.0,
        )

        chunk = STTChunk(
            text="Hello",
            performance_metrics=stream_metrics,
        )

        assert chunk.text == "Hello"
        assert chunk.is_final is False  # Default
        assert chunk.timestamp is None
        assert chunk.confidence is None
        assert chunk.performance_metrics.time_to_first_token_ms == 280.0

    def test_create_stt_chunk_final(self):
        """Should create final STTChunk with STT-specific streaming metrics"""
        stream_metrics = STTStreamPerformanceMetrics(
            time_to_first_token_ms=270.0,
            chunk_latency_ms=50.0,
            final_latency_ms=700.0,  # STT-specific
            tokens_per_second=15.0,  # STT-specific
        )

        chunk = STTChunk(
            text="Hello world",
            is_final=True,
            timestamp=2.5,
            confidence=0.95,
            performance_metrics=stream_metrics,
        )

        assert chunk.text == "Hello world"
        assert chunk.is_final is True
        assert chunk.timestamp == 2.5
        assert chunk.confidence == 0.95
        assert chunk.performance_metrics.final_latency_ms == 700.0
        assert chunk.performance_metrics.tokens_per_second == 15.0


class TestTTSChunk:
    """Test TTSChunk model for streaming"""

    def test_create_tts_chunk(self):
        """Should create TTSChunk with TTS-specific stream metrics"""
        stream_metrics = TTSStreamPerformanceMetrics(
            time_to_first_token_ms=150.0,
            chunk_latency_ms=30.0,
            time_to_first_byte_ms=120.0,  # TTS TTFB
            leading_silence_ms=100.0,  # TTS-specific
        )

        chunk = TTSChunk(
            audio_data=b"audio chunk",
            performance_metrics=stream_metrics,
        )

        assert chunk.audio_data == b"audio chunk"
        assert chunk.is_final is False
        assert chunk.performance_metrics.time_to_first_byte_ms == 120.0
        assert chunk.performance_metrics.leading_silence_ms == 100.0

    def test_create_tts_chunk_final(self):
        """Should create final TTSChunk with TTS-specific metrics"""
        stream_metrics = TTSStreamPerformanceMetrics(
            time_to_first_token_ms=150.0,
            chunk_latency_ms=35.0,
            total_duration_ms=3000.0,
        )

        chunk = TTSChunk(
            audio_data=b"last chunk",
            is_final=True,
            performance_metrics=stream_metrics,
        )

        assert chunk.audio_data == b"last chunk"
        assert chunk.is_final is True
        assert chunk.performance_metrics.total_duration_ms == 3000.0
