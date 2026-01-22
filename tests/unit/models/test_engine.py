import pytest
from pydantic import ValidationError

from app.models.engine import (
    EngineConfig,
    Segment,
    STTChunk,
    STTResponse,
    TTSChunk,
    TTSResponse,
)
from app.models.metrics import (
    STTPerformanceMetrics,
    TTSPerformanceMetrics,
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

    def test_create_config_with_all_fields(self):
        """Should create config with all fields"""
        config = EngineConfig(
            model_name="whisper-large",
            device="cuda",
            max_workers=4,
            timeout_seconds=600,
        )

        assert config.model_name == "whisper-large"
        assert config.device == "cuda"
        assert config.max_workers == 4
        assert config.timeout_seconds == 600

    def test_config_validates_device(self):
        """Device should only accept cpu, cuda, or mps"""
        EngineConfig(model_name="test", device="cpu")
        EngineConfig(model_name="test", device="cuda")
        EngineConfig(model_name="test", device="mps")

        with pytest.raises(ValidationError):
            EngineConfig(model_name="test", device="gpu")  # Invalid


class TestSegment:
    """Test Segment model"""

    def test_create_segment_with_required_fields(self):
        """Should create Segment with required fields"""
        segment = Segment(
            start=0.0,
            end=1.5,
            text="Hello",
        )

        assert segment.start == 0.0
        assert segment.end == 1.5
        assert segment.text == "Hello"
        assert segment.confidence is None

    def test_create_segment_with_confidence(self):
        """Should create Segment with confidence score"""
        segment = Segment(
            start=1.5,
            end=2.3,
            text="world",
            confidence=0.95,
        )

        assert segment.start == 1.5
        assert segment.end == 2.3
        assert segment.text == "world"
        assert segment.confidence == 0.95

    def test_segment_requires_all_fields(self):
        """Should require start, end, and text"""
        with pytest.raises(ValidationError):
            Segment(start=0.0, end=1.0)  # Missing text

        with pytest.raises(ValidationError):
            Segment(start=0.0, text="hello")  # Missing end

        with pytest.raises(ValidationError):
            Segment(end=1.0, text="hello")  # Missing start


class TestSTTResponse:
    """Test STTResponse model for invoke mode"""

    def test_create_stt_response_minimal(self):
        """Should create STTResponse with only required text"""
        response = STTResponse(text="Hello world")

        assert response.text == "Hello world"
        assert response.language is None
        assert response.segments is None
        assert response.performance_metrics is None

    def test_create_stt_response_with_metrics(self):
        """Should create STTResponse with metrics"""
        perf = STTPerformanceMetrics(
            latency_ms=100.0,
            processing_time_ms=95.0,
            real_time_factor=0.5,
        )

        response = STTResponse(
            text="Hello world",
            performance_metrics=perf,
        )

        assert response.text == "Hello world"
        assert response.performance_metrics == perf
        assert response.performance_metrics.real_time_factor == 0.5

    def test_stt_response_with_segments(self):
        """Should create STTResponse with word-level segments"""
        segments = [
            Segment(start=0.0, end=1.0, text="Hello", confidence=0.98),
            Segment(start=1.0, end=2.0, text="world", confidence=0.96),
        ]

        response = STTResponse(
            text="Hello world",
            language="en",
            segments=segments,
        )

        assert response.language == "en"
        assert len(response.segments) == 2
        assert response.segments[0].text == "Hello"
        assert response.segments[0].confidence == 0.98


class TestTTSResponse:
    """Test TTSResponse model for invoke mode"""

    def test_create_tts_response_minimal(self):
        """Should create TTSResponse with required fields"""
        response = TTSResponse(
            audio_data=b"fake audio bytes",
            sample_rate=22050,
            duration_seconds=2.5,
        )

        assert response.audio_data == b"fake audio bytes"
        assert response.sample_rate == 22050
        assert response.format == "wav"  # Default
        assert response.duration_seconds == 2.5
        assert response.performance_metrics is None

    def test_create_tts_response_with_metrics(self):
        """Should create TTSResponse with metrics"""
        perf = TTSPerformanceMetrics(
            latency_ms=200.0,
            processing_time_ms=180.0,
            characters_per_second=50.0,
        )

        response = TTSResponse(
            audio_data=b"fake audio bytes",
            sample_rate=22050,
            duration_seconds=2.5,
            performance_metrics=perf,
        )

        assert response.performance_metrics == perf
        assert response.performance_metrics.characters_per_second == 50.0


class TestSTTChunk:
    """Test STTChunk model for streaming (lightweight)"""

    def test_create_stt_chunk_minimal(self):
        """Should create STTChunk with only text"""
        chunk = STTChunk(text="Hello")

        assert chunk.text == "Hello"
        assert chunk.timestamp is None
        assert chunk.confidence is None
        assert chunk.chunk_latency_ms is None

    def test_create_stt_chunk_with_latency(self):
        """Should create STTChunk with per-chunk latency"""
        chunk = STTChunk(
            text="Hello",
            chunk_latency_ms=45.0,
        )

        assert chunk.text == "Hello"
        assert chunk.chunk_latency_ms == 45.0

    def test_create_stt_chunk_with_all_fields(self):
        """Should create STTChunk with all fields"""
        chunk = STTChunk(
            text="Hello world",
            timestamp=2.5,
            confidence=0.95,
            chunk_latency_ms=50.0,
        )

        assert chunk.text == "Hello world"
        assert chunk.timestamp == 2.5
        assert chunk.confidence == 0.95
        assert chunk.chunk_latency_ms == 50.0


class TestTTSChunk:
    """Test TTSChunk model for streaming (lightweight)"""

    def test_create_tts_chunk_minimal(self):
        """Should create TTSChunk with required fields"""
        chunk = TTSChunk(
            audio_data=b"audio chunk",
            sequence_number=0,
        )

        assert chunk.audio_data == b"audio chunk"
        assert chunk.sequence_number == 0
        assert chunk.chunk_latency_ms is None

    def test_create_tts_chunk_with_latency(self):
        """Should create TTSChunk with per-chunk latency"""
        chunk = TTSChunk(
            audio_data=b"audio chunk",
            sequence_number=1,
            chunk_latency_ms=30.0,
        )

        assert chunk.chunk_latency_ms == 30.0

    def test_create_tts_chunk_with_all_fields(self):
        """Should create TTSChunk with all fields"""
        chunk = TTSChunk(
            audio_data=b"last chunk",
            sequence_number=10,
            chunk_latency_ms=35.0,
        )

        assert chunk.audio_data == b"last chunk"
        assert chunk.sequence_number == 10
        assert chunk.chunk_latency_ms == 35.0


class TestSTTPerformanceMetricsStreaming:
    """Test STTPerformanceMetrics with streaming-specific fields"""

    def test_create_stt_metrics_with_streaming_fields(self):
        """Should create STTPerformanceMetrics with streaming fields"""
        metrics = STTPerformanceMetrics(
            latency_ms=500.0,
            processing_time_ms=450.0,
            audio_duration_ms=3000.0,
            real_time_factor=0.15,
            time_to_first_token_ms=250.0,  # Streaming field
            total_stream_duration_ms=500.0,  # Streaming field
            total_chunks=5,  # Streaming field
        )

        assert metrics.latency_ms == 500.0
        assert metrics.processing_time_ms == 450.0
        assert metrics.audio_duration_ms == 3000.0
        assert metrics.real_time_factor == 0.15
        assert metrics.time_to_first_token_ms == 250.0
        assert metrics.total_stream_duration_ms == 500.0
        assert metrics.total_chunks == 5

    def test_create_stt_metrics_invoke_mode(self):
        """Should create STTPerformanceMetrics for invoke mode (streaming fields None)"""
        metrics = STTPerformanceMetrics(
            latency_ms=500.0,
            processing_time_ms=450.0,
            audio_duration_ms=3000.0,
        )

        assert metrics.latency_ms == 500.0
        assert metrics.processing_time_ms == 450.0
        # Streaming fields should be None
        assert metrics.time_to_first_token_ms is None
        assert metrics.total_stream_duration_ms is None
        assert metrics.total_chunks is None

    def test_stt_response_for_streaming(self):
        """Should work with STTResponse for streaming mode"""
        response = STTResponse(
            text="Hello world",
            language="en",
            segments=[
                Segment(start=0.0, end=0.5, text="Hello", confidence=0.95),
                Segment(start=0.5, end=1.0, text="world", confidence=0.92),
            ],
            performance_metrics=STTPerformanceMetrics(
                latency_ms=500.0,
                processing_time_ms=450.0,
                audio_duration_ms=1000.0,
                time_to_first_token_ms=250.0,
                total_stream_duration_ms=500.0,
                total_chunks=3,
            ),
        )

        assert response.text == "Hello world"
        assert response.language == "en"
        assert len(response.segments) == 2
        assert response.performance_metrics.time_to_first_token_ms == 250.0
        assert response.performance_metrics.total_chunks == 3


class TestTTSPerformanceMetricsStreaming:
    """Test TTSPerformanceMetrics with streaming-specific fields"""

    def test_create_tts_metrics_with_streaming_fields(self):
        """Should create TTSPerformanceMetrics with streaming fields"""
        metrics = TTSPerformanceMetrics(
            latency_ms=400.0,
            processing_time_ms=380.0,
            characters_per_second=50.0,
            time_to_first_byte_ms=120.0,  # Streaming field
            total_stream_duration_ms=400.0,  # Streaming field
            total_chunks=8,  # Streaming field
        )

        assert metrics.latency_ms == 400.0
        assert metrics.processing_time_ms == 380.0
        assert metrics.characters_per_second == 50.0
        assert metrics.time_to_first_byte_ms == 120.0
        assert metrics.total_stream_duration_ms == 400.0
        assert metrics.total_chunks == 8

    def test_create_tts_metrics_invoke_mode(self):
        """Should create TTSPerformanceMetrics for invoke mode (streaming fields None)"""
        metrics = TTSPerformanceMetrics(
            latency_ms=400.0,
            processing_time_ms=380.0,
        )

        assert metrics.latency_ms == 400.0
        assert metrics.processing_time_ms == 380.0
        # Streaming fields should be None
        assert metrics.time_to_first_byte_ms is None
        assert metrics.total_stream_duration_ms is None
        assert metrics.total_chunks is None
