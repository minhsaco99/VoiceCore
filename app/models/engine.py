"""
Engine configuration and output data models

Key design:
- STTOutput uses STTInvokePerformanceMetrics (STT batch processing)
- STTChunk uses STTStreamPerformanceMetrics (STT streaming)
- TTSOutput uses TTSInvokePerformanceMetrics (TTS batch processing)
- TTSChunk uses TTSStreamPerformanceMetrics (TTS streaming)
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.metrics import (
    QualityMetrics,
    STTInvokePerformanceMetrics,
    STTStreamPerformanceMetrics,
    TTSInvokePerformanceMetrics,
    TTSStreamPerformanceMetrics,
)


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


class STTOutput(BaseModel):
    """Output from STT invoke mode (batch processing)"""

    text: str = Field(..., description="Transcribed text")
    language: str | None = Field(None, description="Detected/specified language")
    segments: list[dict] | None = Field(None, description="Word-level timestamps")

    quality_metrics: QualityMetrics
    performance_metrics: STTInvokePerformanceMetrics  # STT-specific invoke metrics!


class TTSOutput(BaseModel):
    """Output from TTS invoke mode (batch processing)"""

    audio_data: bytes = Field(..., description="Generated audio")
    sample_rate: int = Field(..., description="Audio sample rate (Hz)")
    duration_seconds: float = Field(..., description="Audio duration")
    format: str = Field(default="wav", description="Audio format")

    quality_metrics: QualityMetrics
    performance_metrics: TTSInvokePerformanceMetrics  # TTS-specific invoke metrics!


class STTChunk(BaseModel):
    """Streaming chunk from STT"""

    text: str = Field(..., description="Partial or final transcription")
    is_final: bool = Field(default=False, description="Is this the final chunk?")
    timestamp: float | None = Field(None, description="Timestamp in audio")
    confidence: float | None = Field(
        None, ge=0.0, le=1.0, description="Confidence score"
    )

    performance_metrics: STTStreamPerformanceMetrics  # STT-specific streaming metrics!


class TTSChunk(BaseModel):
    """Streaming chunk from TTS"""

    audio_data: bytes = Field(..., description="Audio chunk")
    is_final: bool = Field(default=False, description="Is this the final chunk?")

    performance_metrics: TTSStreamPerformanceMetrics  # TTS-specific streaming metrics!
