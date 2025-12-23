"""
Performance metrics for STT and TTS engines

Design:
- STTPerformanceMetrics: Metrics for STT invoke mode
- TTSPerformanceMetrics: Metrics for TTS invoke mode
- Streaming metrics are in STTStreamSummary/TTSStreamSummary (engine.py)

Industry benchmarks (2024):
- STT TTFT: ~270-300ms (Gladia, Deepgram)
- TTS TTFB: ~100-200ms (Elevenlabs ~100ms, Deepgram ~150ms)
- RTF < 1.0 = faster than real-time
"""

from pydantic import BaseModel, Field


# =============================================================================
# STT Performance Metrics
# =============================================================================


class STTPerformanceMetrics(BaseModel):
    """
    Performance metrics for STT processing

    Used in STTResponse for invoke/batch mode.
    Streaming metrics are in STTStreamSummary.
    """

    # Core timing (required)
    latency_ms: float = Field(..., description="Total end-to-end latency")
    processing_time_ms: float = Field(..., description="Actual model processing time")

    # Optional timing
    queue_time_ms: float | None = Field(None, description="Time spent in queue")

    # STT-specific metrics
    audio_duration_ms: float | None = Field(
        None, description="Input audio duration in ms"
    )
    real_time_factor: float | None = Field(
        None,
        description="RTF = processing_time / audio_duration (<1.0 = faster than real-time)",
    )


# =============================================================================
# TTS Performance Metrics
# =============================================================================


class TTSPerformanceMetrics(BaseModel):
    """
    Performance metrics for TTS processing

    Used in TTSResponse for invoke/batch mode.
    Streaming metrics are in TTSStreamSummary.
    """

    # Core timing (required)
    latency_ms: float = Field(..., description="Total end-to-end latency")
    processing_time_ms: float = Field(..., description="Actual model processing time")

    # Optional timing
    queue_time_ms: float | None = Field(None, description="Time spent in queue")

    # TTS-specific metrics
    characters_per_second: float | None = Field(
        None, description="Text→audio synthesis speed"
    )
    real_time_factor: float | None = Field(
        None,
        description="RTF = processing_time / audio_duration (<1.0 = faster than real-time)",
    )
