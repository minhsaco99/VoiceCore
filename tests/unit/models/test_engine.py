import pytest
from pydantic import ValidationError

from app.models.engine import (
    EngineConfig,
    Segment,
    STTChunk,
    STTRequest,
    STTResponse,
    STTStreamSummary,
    TTSChunk,
    TTSRequest,
    TTSResponse,
    TTSStreamSummary,
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


class TestSTTRequest:
    """Test STTRequest model (unified for REST and WebSocket)"""

    def test_create_stt_request_for_rest(self):
        """Should create STTRequest for REST with audio_data"""
        request = STTRequest(audio_data=b"fake audio")

        assert request.audio_data == b"fake audio"
        assert request.language is None
        assert request.format is None
        assert request.sample_rate is None
        assert request.stream_response is False

    def test_create_stt_request_for_websocket_config(self):
        """Should create STTRequest for WebSocket config (no audio_data)"""
        request = STTRequest(
            language="en",
            format="wav",
            sample_rate=16000,
        )

        assert request.audio_data is None  # WebSocket config
        assert request.language == "en"
        assert request.format == "wav"
        assert request.sample_rate == 16000

    def test_create_stt_request_with_stream_response(self):
        """Should create STTRequest with streaming response preference"""
        request = STTRequest(
            audio_data=b"fake audio",
            stream_response=True,
        )

        assert request.audio_data == b"fake audio"
        assert request.stream_response is True

    def test_stt_request_all_fields_optional_except_pattern(self):
        """Should allow creating empty STTRequest (for WebSocket config)"""
        request = STTRequest()
        assert request.audio_data is None
        assert request.engine_params == {}

    def test_stt_request_with_engine_params(self):
        """Should allow passing engine-specific parameters"""
        params = {"temperature": 0.7, "beam_size": 5}
        request = STTRequest(engine_params=params)
        assert request.engine_params == params


class TestTTSRequest:
    """Test TTSRequest model (REST only)"""

    def test_create_tts_request_minimal(self):
        """Should create TTSRequest with only text"""
        request = TTSRequest(text="Hello world")

        assert request.text == "Hello world"
        assert request.voice is None
        assert request.speed == 1.0  # Default
        assert request.stream_response is False

    def test_create_tts_request_with_all_fields(self):
        """Should create TTSRequest with all fields"""
        request = TTSRequest(
            text="Hello world",
            voice="en-US-JennyNeural",
            speed=1.2,
            stream_response=True,
        )

        assert request.text == "Hello world"
        assert request.voice == "en-US-JennyNeural"
        assert request.speed == 1.2
        assert request.stream_response is True

    def test_tts_request_requires_text(self):
        """Should require text field"""
        with pytest.raises(ValidationError):
            TTSRequest(voice="en-US-JennyNeural")

    def test_tts_request_validates_speed(self):
        """Speed must be > 0 and <= 3.0"""
        TTSRequest(text="test", speed=0.5)
        TTSRequest(text="test", speed=3.0)

        with pytest.raises(ValidationError):
            TTSRequest(text="test", speed=0)  # Too low

        with pytest.raises(ValidationError):
            TTSRequest(text="test", speed=3.5)  # Too high

    def test_tts_request_with_engine_params(self):
        """Should allow passing engine-specific parameters"""
        params = {"api_key": "test_key", "model_version": "v1"}
        request = TTSRequest(text="hello", engine_params=params)
        assert request.engine_params == params
        assert request.text == "hello"


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
        assert chunk.is_final is False  # Default
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

    def test_create_stt_chunk_final(self):
        """Should create final STTChunk"""
        chunk = STTChunk(
            text="Hello world",
            is_final=True,
            timestamp=2.5,
            confidence=0.95,
            chunk_latency_ms=50.0,
        )

        assert chunk.text == "Hello world"
        assert chunk.is_final is True
        assert chunk.timestamp == 2.5
        assert chunk.confidence == 0.95


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
        assert chunk.is_final is False
        assert chunk.chunk_latency_ms is None

    def test_create_tts_chunk_with_latency(self):
        """Should create TTSChunk with per-chunk latency"""
        chunk = TTSChunk(
            audio_data=b"audio chunk",
            sequence_number=1,
            chunk_latency_ms=30.0,
        )

        assert chunk.chunk_latency_ms == 30.0

    def test_create_tts_chunk_final(self):
        """Should create final TTSChunk"""
        chunk = TTSChunk(
            audio_data=b"last chunk",
            sequence_number=10,
            is_final=True,
            chunk_latency_ms=35.0,
        )

        assert chunk.audio_data == b"last chunk"
        assert chunk.is_final is True
        assert chunk.sequence_number == 10


class TestSTTStreamSummary:
    """Test STTStreamSummary model (sent at end of stream)"""

    def test_create_stt_stream_summary(self):
        """Should create STTStreamSummary with aggregate metrics"""
        summary = STTStreamSummary(
            total_text="Hello world this is a test",
            total_chunks=5,
            audio_duration_seconds=3.5,
            time_to_first_token_ms=280.0,
            total_duration_ms=4000.0,
        )

        assert summary.total_text == "Hello world this is a test"
        assert summary.total_chunks == 5
        assert summary.audio_duration_seconds == 3.5
        assert summary.time_to_first_token_ms == 280.0
        assert summary.total_duration_ms == 4000.0

    def test_create_stt_stream_summary_minimal(self):
        """Should create STTStreamSummary with only required fields"""
        summary = STTStreamSummary(
            total_text="Hello",
            total_chunks=1,
        )

        assert summary.total_text == "Hello"
        assert summary.total_chunks == 1
        assert summary.time_to_first_token_ms is None


class TestTTSStreamSummary:
    """Test TTSStreamSummary model (sent at end of stream)"""

    def test_create_tts_stream_summary(self):
        """Should create TTSStreamSummary with aggregate metrics"""
        summary = TTSStreamSummary(
            total_bytes=48000,
            total_chunks=10,
            audio_duration_seconds=3.0,
            time_to_first_byte_ms=120.0,
            total_duration_ms=3500.0,
        )

        assert summary.total_bytes == 48000
        assert summary.total_chunks == 10
        assert summary.audio_duration_seconds == 3.0
        assert summary.time_to_first_byte_ms == 120.0
        assert summary.total_duration_ms == 3500.0

    def test_create_tts_stream_summary_minimal(self):
        """Should create TTSStreamSummary with only required fields"""
        summary = TTSStreamSummary(
            total_bytes=4800,
            total_chunks=1,
        )

        assert summary.total_bytes == 4800
        assert summary.total_chunks == 1
        assert summary.time_to_first_byte_ms is None
