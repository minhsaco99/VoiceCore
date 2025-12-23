import pytest
from pydantic import ValidationError

from app.models.metrics import (
    STTPerformanceMetrics,
    TTSPerformanceMetrics,
)


class TestSTTPerformanceMetrics:
    """Test STTPerformanceMetrics for STT processing"""

    def test_create_stt_metrics_with_required_fields(self):
        """Should create STTPerformanceMetrics with required fields"""
        metrics = STTPerformanceMetrics(
            latency_ms=150.5,
            processing_time_ms=145.0,
        )

        assert metrics.latency_ms == 150.5
        assert metrics.processing_time_ms == 145.0
        assert metrics.queue_time_ms is None  # Optional

    def test_stt_metrics_with_all_fields(self):
        """Should create STTPerformanceMetrics with all fields"""
        metrics = STTPerformanceMetrics(
            latency_ms=200.0,
            processing_time_ms=180.0,
            queue_time_ms=20.0,
            audio_duration_ms=360.0,
            real_time_factor=0.5,  # 2x faster than real-time
        )

        assert metrics.latency_ms == 200.0
        assert metrics.processing_time_ms == 180.0
        assert metrics.queue_time_ms == 20.0
        assert metrics.audio_duration_ms == 360.0
        assert metrics.real_time_factor == 0.5

    def test_stt_metrics_validates_types(self):
        """Should validate field types"""
        with pytest.raises(ValidationError):
            STTPerformanceMetrics(
                latency_ms="invalid",
                processing_time_ms=100.0,
            )

    def test_stt_metrics_requires_latency(self):
        """Should require latency_ms field"""
        with pytest.raises(ValidationError):
            STTPerformanceMetrics(processing_time_ms=100.0)

    def test_stt_metrics_requires_processing_time(self):
        """Should require processing_time_ms field"""
        with pytest.raises(ValidationError):
            STTPerformanceMetrics(latency_ms=100.0)


class TestTTSPerformanceMetrics:
    """Test TTSPerformanceMetrics for TTS processing"""

    def test_create_tts_metrics_with_required_fields(self):
        """Should create TTSPerformanceMetrics with required fields"""
        metrics = TTSPerformanceMetrics(
            latency_ms=200.0,
            processing_time_ms=180.0,
        )

        assert metrics.latency_ms == 200.0
        assert metrics.processing_time_ms == 180.0
        assert metrics.queue_time_ms is None  # Optional

    def test_tts_metrics_with_all_fields(self):
        """Should create TTSPerformanceMetrics with all fields"""
        metrics = TTSPerformanceMetrics(
            latency_ms=200.0,
            processing_time_ms=180.0,
            queue_time_ms=15.0,
            characters_per_second=50.0,
            real_time_factor=1.2,
        )

        assert metrics.latency_ms == 200.0
        assert metrics.processing_time_ms == 180.0
        assert metrics.queue_time_ms == 15.0
        assert metrics.characters_per_second == 50.0
        assert metrics.real_time_factor == 1.2

    def test_tts_metrics_validates_types(self):
        """Should validate field types"""
        with pytest.raises(ValidationError):
            TTSPerformanceMetrics(
                latency_ms="invalid",
                processing_time_ms=100.0,
            )

    def test_tts_metrics_requires_latency(self):
        """Should require latency_ms field"""
        with pytest.raises(ValidationError):
            TTSPerformanceMetrics(processing_time_ms=100.0)

    def test_tts_metrics_requires_processing_time(self):
        """Should require processing_time_ms field"""
        with pytest.raises(ValidationError):
            TTSPerformanceMetrics(latency_ms=100.0)
