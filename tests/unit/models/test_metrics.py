import pytest
from pydantic import ValidationError

from app.models.metrics import (
    QualityMetrics,
    STTInvokePerformanceMetrics,
    STTStreamPerformanceMetrics,
    TTSInvokePerformanceMetrics,
    TTSStreamPerformanceMetrics,
)


class TestSTTInvokePerformanceMetrics:
    """Test STTInvokePerformanceMetrics for STT batch processing"""

    def test_create_stt_invoke_metrics_with_required_fields(self):
        """Should create STTInvokePerformanceMetrics with required fields"""
        metrics = STTInvokePerformanceMetrics(
            latency_ms=150.5,
            processing_time_ms=145.0,
        )

        assert metrics.latency_ms == 150.5
        assert metrics.processing_time_ms == 145.0
        assert metrics.queue_time_ms == 0.0  # Default

    def test_stt_invoke_metrics_with_all_fields(self):
        """Should create STTInvokePerformanceMetrics with all STT-specific fields"""
        metrics = STTInvokePerformanceMetrics(
            latency_ms=200.0,
            processing_time_ms=180.0,
            queue_time_ms=20.0,
            real_time_factor=0.5,  # 2x faster than real-time (STT-specific)
            audio_duration_ms=360.0,  # STT-specific
            throughput_seconds_per_second=2.0,  # STT-specific
        )

        assert metrics.latency_ms == 200.0
        assert metrics.processing_time_ms == 180.0
        assert metrics.queue_time_ms == 20.0
        assert metrics.real_time_factor == 0.5
        assert metrics.audio_duration_ms == 360.0
        assert metrics.throughput_seconds_per_second == 2.0

    def test_stt_invoke_metrics_validates_types(self):
        """Should validate field types"""
        with pytest.raises(ValidationError):
            STTInvokePerformanceMetrics(
                latency_ms="invalid",
                processing_time_ms=100.0,
            )

    def test_stt_invoke_metrics_requires_latency(self):
        """Should require latency_ms field"""
        with pytest.raises(ValidationError):
            STTInvokePerformanceMetrics(processing_time_ms=100.0)

    def test_stt_invoke_metrics_requires_processing_time(self):
        """Should require processing_time_ms field"""
        with pytest.raises(ValidationError):
            STTInvokePerformanceMetrics(latency_ms=100.0)


class TestSTTStreamPerformanceMetrics:
    """Test STTStreamPerformanceMetrics for STT streaming"""

    def test_create_stt_stream_metrics_with_required_fields(self):
        """Should create STTStreamPerformanceMetrics with required fields"""
        metrics = STTStreamPerformanceMetrics(
            time_to_first_token_ms=280.0,  # STT TTFT ~270-300ms
            chunk_latency_ms=50.0,
        )

        assert metrics.time_to_first_token_ms == 280.0
        assert metrics.chunk_latency_ms == 50.0

    def test_stt_stream_metrics_with_all_fields(self):
        """Should create STTStreamPerformanceMetrics with all STT-specific fields"""
        metrics = STTStreamPerformanceMetrics(
            time_to_first_token_ms=270.0,
            chunk_latency_ms=45.0,
            final_latency_ms=700.0,  # STT-specific: time to final output
            tokens_per_second=15.5,  # STT-specific
            partial_update_latency_ms=100.0,  # STT-specific: between partial updates
            total_duration_ms=5000.0,
        )

        assert metrics.time_to_first_token_ms == 270.0
        assert metrics.chunk_latency_ms == 45.0
        assert metrics.final_latency_ms == 700.0
        assert metrics.tokens_per_second == 15.5
        assert metrics.partial_update_latency_ms == 100.0
        assert metrics.total_duration_ms == 5000.0

    def test_stt_stream_metrics_validates_types(self):
        """Should validate field types"""
        with pytest.raises(ValidationError):
            STTStreamPerformanceMetrics(
                time_to_first_token_ms="invalid",
                chunk_latency_ms=50.0,
            )

    def test_stt_stream_metrics_requires_ttft(self):
        """Should require time_to_first_token_ms field"""
        with pytest.raises(ValidationError):
            STTStreamPerformanceMetrics(chunk_latency_ms=50.0)

    def test_stt_stream_metrics_requires_chunk_latency(self):
        """Should require chunk_latency_ms field"""
        with pytest.raises(ValidationError):
            STTStreamPerformanceMetrics(time_to_first_token_ms=280.0)


class TestTTSInvokePerformanceMetrics:
    """Test TTSInvokePerformanceMetrics for TTS batch processing"""

    def test_create_tts_invoke_metrics_with_required_fields(self):
        """Should create TTSInvokePerformanceMetrics with required fields"""
        metrics = TTSInvokePerformanceMetrics(
            latency_ms=200.0,
            processing_time_ms=180.0,
        )

        assert metrics.latency_ms == 200.0
        assert metrics.processing_time_ms == 180.0
        assert metrics.queue_time_ms == 0.0  # Default

    def test_tts_invoke_metrics_with_all_fields(self):
        """Should create TTSInvokePerformanceMetrics with all TTS-specific fields"""
        metrics = TTSInvokePerformanceMetrics(
            latency_ms=200.0,
            processing_time_ms=180.0,
            queue_time_ms=15.0,
            characters_per_second=50.0,  # TTS-specific
            audio_generation_ratio=1.2,  # TTS-specific: audio_duration / processing_time
        )

        assert metrics.latency_ms == 200.0
        assert metrics.processing_time_ms == 180.0
        assert metrics.queue_time_ms == 15.0
        assert metrics.characters_per_second == 50.0
        assert metrics.audio_generation_ratio == 1.2

    def test_tts_invoke_metrics_validates_types(self):
        """Should validate field types"""
        with pytest.raises(ValidationError):
            TTSInvokePerformanceMetrics(
                latency_ms="invalid",
                processing_time_ms=100.0,
            )

    def test_tts_invoke_metrics_requires_latency(self):
        """Should require latency_ms field"""
        with pytest.raises(ValidationError):
            TTSInvokePerformanceMetrics(processing_time_ms=100.0)

    def test_tts_invoke_metrics_requires_processing_time(self):
        """Should require processing_time_ms field"""
        with pytest.raises(ValidationError):
            TTSInvokePerformanceMetrics(latency_ms=100.0)


class TestTTSStreamPerformanceMetrics:
    """Test TTSStreamPerformanceMetrics for TTS streaming"""

    def test_create_tts_stream_metrics_with_required_fields(self):
        """Should create TTSStreamPerformanceMetrics with required fields"""
        metrics = TTSStreamPerformanceMetrics(
            time_to_first_token_ms=150.0,
            chunk_latency_ms=30.0,
        )

        assert metrics.time_to_first_token_ms == 150.0
        assert metrics.chunk_latency_ms == 30.0

    def test_tts_stream_metrics_with_all_fields(self):
        """Should create TTSStreamPerformanceMetrics with all TTS-specific fields"""
        metrics = TTSStreamPerformanceMetrics(
            time_to_first_token_ms=150.0,
            chunk_latency_ms=30.0,
            time_to_first_byte_ms=120.0,  # TTS TTFB ~100-200ms
            leading_silence_ms=100.0,  # TTS-specific: silence before speech starts
            chunk_generation_rate_ms=25.0,  # TTS-specific: time between chunks
            total_duration_ms=3000.0,
        )

        assert metrics.time_to_first_token_ms == 150.0
        assert metrics.chunk_latency_ms == 30.0
        assert metrics.time_to_first_byte_ms == 120.0
        assert metrics.leading_silence_ms == 100.0
        assert metrics.chunk_generation_rate_ms == 25.0
        assert metrics.total_duration_ms == 3000.0

    def test_tts_stream_metrics_validates_types(self):
        """Should validate field types"""
        with pytest.raises(ValidationError):
            TTSStreamPerformanceMetrics(
                time_to_first_token_ms="invalid",
                chunk_latency_ms=30.0,
            )

    def test_tts_stream_metrics_requires_ttft(self):
        """Should require time_to_first_token_ms field"""
        with pytest.raises(ValidationError):
            TTSStreamPerformanceMetrics(chunk_latency_ms=30.0)

    def test_tts_stream_metrics_requires_chunk_latency(self):
        """Should require chunk_latency_ms field"""
        with pytest.raises(ValidationError):
            TTSStreamPerformanceMetrics(time_to_first_token_ms=150.0)


class TestQualityMetrics:
    """Test QualityMetrics model (shared by both STT and TTS)"""

    def test_create_quality_metrics_all_optional(self):
        """Should create QualityMetrics with no fields (all optional)"""
        metrics = QualityMetrics()

        assert metrics.confidence_score is None
        assert metrics.word_error_rate is None
        assert metrics.signal_to_noise_ratio is None
        assert metrics.mean_opinion_score is None

    def test_quality_metrics_with_confidence(self):
        """Should create QualityMetrics with confidence score"""
        metrics = QualityMetrics(confidence_score=0.95)

        assert metrics.confidence_score == 0.95
        assert metrics.word_error_rate is None

    def test_quality_metrics_validates_confidence_range(self):
        """Confidence score should be between 0 and 1"""
        # Valid
        QualityMetrics(confidence_score=0.0)
        QualityMetrics(confidence_score=1.0)
        QualityMetrics(confidence_score=0.5)

        # Invalid - too low
        with pytest.raises(ValidationError):
            QualityMetrics(confidence_score=-0.1)

        # Invalid - too high
        with pytest.raises(ValidationError):
            QualityMetrics(confidence_score=1.1)

    def test_quality_metrics_with_all_fields(self):
        """Should create QualityMetrics with all fields"""
        metrics = QualityMetrics(
            confidence_score=0.92,
            word_error_rate=0.05,  # STT-specific
            signal_to_noise_ratio=25.3,
            mean_opinion_score=4.5,  # TTS-specific
        )

        assert metrics.confidence_score == 0.92
        assert metrics.word_error_rate == 0.05
        assert metrics.signal_to_noise_ratio == 25.3
        assert metrics.mean_opinion_score == 4.5

    def test_quality_metrics_validates_mos_range(self):
        """Mean Opinion Score should be between 1 and 5"""
        QualityMetrics(mean_opinion_score=1.0)
        QualityMetrics(mean_opinion_score=5.0)
        QualityMetrics(mean_opinion_score=3.5)

        with pytest.raises(ValidationError):
            QualityMetrics(mean_opinion_score=0.5)  # Too low

        with pytest.raises(ValidationError):
            QualityMetrics(mean_opinion_score=5.5)  # Too high
