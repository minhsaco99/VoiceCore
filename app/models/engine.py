"""
Engine configuration and input/output data models for STT/TTS

Design:
- STTRequest: Unified for REST (with audio_data) and WebSocket config (audio_data=None)
- TTSRequest: REST only with stream_response option
- Lightweight chunks for streaming (minimal per-chunk metrics)
- Full response models with complete metrics for invoke mode
- Stream summaries for aggregate metrics at end of stream
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.metrics import (
    STTPerformanceMetrics,
    TTSPerformanceMetrics,
)


# =============================================================================
# Configuration
# =============================================================================


class EngineConfig(BaseModel):
    """
    Configuration for engine initialization

    Supports both common fields and engine-specific params
    """

    # Required
    model_name: str = Field(..., description="Model name/path to use")

    # Common optional fields
    device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu", description="Device to run on"
    )
    max_workers: int = Field(default=1, ge=1, description="Max parallel workers")
    timeout_seconds: int = Field(default=300, ge=1, description="Processing timeout")

    # Engine-specific parameters (flexible dict)
    engine_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Engine-specific parameters (e.g., temperature, beam_size, api_keys)",
    )


# =============================================================================
# Request Models
# =============================================================================


class STTRequest(BaseModel):
    """
    Input for STT processing (REST or WebSocket)

    Usage:
    - REST POST: audio_data is required, stream_response controls output format
    - WebSocket config: audio_data=None, subsequent messages are raw audio bytes
    """

    # Audio - required for REST, None for WebSocket config message
    audio_data: bytes | None = Field(
        None, description="Audio bytes (REST) or None (WebSocket config)"
    )

    # Config options
    language: str | None = Field(None, description="Language hint (e.g., 'en', 'vi')")
    format: str | None = Field(None, description="Audio format hint (wav, mp3, webm)")
    sample_rate: int | None = Field(
        None, description="Sample rate in Hz (important for streaming)"
    )

    # Response preference (REST only)
    stream_response: bool = Field(
        default=False, description="Return StreamingResponse chunks vs full response"
    )

    # Engine-specific parameters (flexible dict)
    engine_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Engine-specific parameters (e.g., temperature, beam_size, api_keys)",
    )


class TTSRequest(BaseModel):
    """
    Input for TTS processing (REST only)

    stream_response controls whether to return full audio or StreamingResponse
    """

    text: str = Field(..., description="Text to synthesize")
    voice: str | None = Field(None, description="Voice name/ID to use")
    speed: float = Field(
        default=1.0, gt=0, le=3.0, description="Speech speed multiplier"
    )

    # Response preference
    stream_response: bool = Field(
        default=False, description="Return StreamingResponse chunks vs full response"
    )

    # Engine-specific parameters (flexible dict)
    engine_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Engine-specific parameters (e.g., temperature, beam_size, api_keys)",
    )


# =============================================================================
# Streaming Chunk Models (Lightweight - minimal per-chunk overhead)
# =============================================================================


class STTChunk(BaseModel):
    """
    Streaming chunk from STT

    Lightweight model for real-time streaming - heavy metrics in STTStreamSummary
    """

    text: str = Field(..., description="Partial or final transcription text")
    is_final: bool = Field(default=False, description="Is this the final chunk?")
    timestamp: float | None = Field(
        None, description="Timestamp position in audio (seconds)"
    )
    confidence: float | None = Field(
        None, ge=0.0, le=1.0, description="Confidence score for this chunk"
    )

    # Per-chunk timing only
    chunk_latency_ms: float | None = Field(
        None, description="Processing latency for this specific chunk"
    )


class TTSChunk(BaseModel):
    """
    Streaming chunk from TTS

    Lightweight model for real-time audio streaming
    """

    audio_data: bytes = Field(..., description="Audio chunk bytes")
    is_final: bool = Field(default=False, description="Is this the final chunk?")
    sequence_number: int = Field(..., ge=0, description="Chunk sequence for ordering")

    # Per-chunk timing only
    chunk_latency_ms: float | None = Field(
        None, description="Generation latency for this specific chunk"
    )


# =============================================================================
# Full Response Models (Invoke mode - complete output with all metrics)
# =============================================================================


class Segment(BaseModel):
    """
    Word/phrase segment with timing information

    Used in STTResponse for word-level timestamps
    """

    start: float = Field(..., ge=0, description="Start time in seconds")
    end: float = Field(..., ge=0, description="End time in seconds")
    text: str = Field(..., description="The word or phrase")
    confidence: float | None = Field(
        None, ge=0.0, le=1.0, description="Confidence score for this segment"
    )


class STTResponse(BaseModel):
    """
    Full response from STT invoke mode (REST non-streaming)

    Contains complete transcription with all metrics
    """

    text: str = Field(..., description="Complete transcribed text")
    language: str | None = Field(None, description="Detected or specified language")
    segments: list[Segment] | None = Field(
        None, description="Word-level timestamps (if available)"
    )

    # Metrics (optional - not all engines provide all metrics)
    performance_metrics: STTPerformanceMetrics | None = Field(
        None, description="Performance metrics"
    )


class TTSResponse(BaseModel):
    """
    Full response from TTS invoke mode (REST non-streaming)

    Contains complete audio with all metrics
    """

    audio_data: bytes = Field(..., description="Complete generated audio")
    sample_rate: int = Field(..., description="Audio sample rate in Hz")
    duration_seconds: float = Field(..., ge=0, description="Audio duration in seconds")
    format: str = Field(default="wav", description="Audio format (wav, mp3, etc.)")

    # Metrics (optional)
    performance_metrics: TTSPerformanceMetrics | None = Field(
        None, description="Performance metrics"
    )


# =============================================================================
# Stream Summary Models (Sent at end of streaming session)
# =============================================================================


class STTStreamSummary(BaseModel):
    """
    Summary sent at end of STT streaming session

    Aggregates metrics that only make sense after stream completes
    """

    total_text: str = Field(..., description="Complete accumulated transcription")
    total_chunks: int = Field(..., ge=0, description="Number of chunks sent")
    audio_duration_seconds: float | None = Field(
        None, ge=0, description="Total audio duration processed"
    )

    # Aggregate timing metrics
    time_to_first_token_ms: float | None = Field(
        None, description="TTFT - Time to first token"
    )
    total_duration_ms: float | None = Field(None, description="Total stream duration")


class TTSStreamSummary(BaseModel):
    """
    Summary sent at end of TTS streaming session

    Aggregates metrics that only make sense after stream completes
    """

    total_bytes: int = Field(..., ge=0, description="Total audio bytes sent")
    total_chunks: int = Field(..., ge=0, description="Number of chunks sent")
    audio_duration_seconds: float | None = Field(
        None, ge=0, description="Total audio duration generated"
    )

    # Aggregate timing metrics
    time_to_first_byte_ms: float | None = Field(
        None, description="TTFB - Time to first audio byte"
    )
    total_duration_ms: float | None = Field(None, description="Total stream duration")
