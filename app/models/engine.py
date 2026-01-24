"""
Engine configuration and input/output data models for STT/TTS

Design:
- Lightweight chunks for streaming (minimal per-chunk metrics)
- Full response models (STTResponse/TTSResponse) used for both invoke and streaming modes
- STTPerformanceMetrics/TTSPerformanceMetrics: Extended with streaming-specific fields
"""

import base64
from typing import Literal

from pydantic import BaseModel, Field, field_serializer

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


# =============================================================================
# Streaming Chunk Models (Lightweight - minimal per-chunk overhead)
# =============================================================================


class STTChunk(BaseModel):
    """
    Streaming chunk from STT

    Lightweight model for real-time streaming - represents partial transcription results.
    Stream ends when STTResponse is received instead of STTChunk.
    """

    text: str = Field(..., description="Partial transcription text")
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

    Lightweight model for real-time audio streaming - represents partial audio generation.
    Stream ends when TTSResponse is received instead of TTSChunk.
    """

    audio_data: bytes = Field(..., description="Audio chunk bytes")
    sequence_number: int = Field(..., ge=0, description="Chunk sequence for ordering")

    # Per-chunk timing only
    chunk_latency_ms: float | None = Field(
        None, description="Generation latency for this specific chunk"
    )

    @field_serializer("audio_data")
    def serialize_audio(self, audio_data: bytes, _info) -> str:
        return base64.b64encode(audio_data).decode("utf-8")


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

    @field_serializer("audio_data")
    def serialize_audio(self, audio_data: bytes, _info) -> str:
        return base64.b64encode(audio_data).decode("utf-8")
