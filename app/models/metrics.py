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
    Performance metrics for STT processing (invoke and streaming modes)

    Used in STTResponse for both invoke and streaming modes.
    Streaming-specific fields are None in invoke mode.
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

    # Streaming-specific metrics (None in invoke mode)
    time_to_first_token_ms: float | None = Field(
        None, description="TTFT - Time to first token (streaming only)"
    )
    total_stream_duration_ms: float | None = Field(
        None, description="Total stream duration end-to-end (streaming only)"
    )
    total_chunks: int | None = Field(
        None, ge=0, description="Number of chunks yielded (streaming only)"
    )


# =============================================================================
# TTS Performance Metrics
# =============================================================================


class TTSPerformanceMetrics(BaseModel):
    """
    Performance metrics for TTS processing (invoke and streaming modes)

    Used in TTSResponse for both invoke and streaming modes.
    Streaming-specific fields are None in invoke mode.
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

    # Streaming-specific metrics (None in invoke mode)
    time_to_first_byte_ms: float | None = Field(
        None, description="TTFB - Time to first audio byte (streaming only)"
    )
    total_stream_duration_ms: float | None = Field(
        None, description="Total stream duration end-to-end (streaming only)"
    )
    total_chunks: int | None = Field(
        None, ge=0, description="Number of chunks yielded (streaming only)"
    )
