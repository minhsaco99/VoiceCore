"""
Performance and quality metrics for STT and TTS engines

Key design:
- Separate metrics for STT vs TTS (4 classes total)
- STTInvokePerformanceMetrics: STT batch processing
- STTStreamPerformanceMetrics: STT streaming
- TTSInvokePerformanceMetrics: TTS batch processing
- TTSStreamPerformanceMetrics: TTS streaming
- QualityMetrics: Shared quality metrics (confidence, WER, MOS, etc.)

Industry benchmarks (2024):
- STT TTFT: ~270-300ms (Gladia, Deepgram)
- TTS TTFB: ~100-200ms (Elevenlabs ~100ms, Deepgram ~150ms)
- RTF < 1.0 = faster than real-time
"""

from pydantic import BaseModel, Field

# ============================================================================
# STT Performance Metrics
# ============================================================================


class STTInvokePerformanceMetrics(BaseModel):
    """
    Performance metrics for STT invoke/batch mode processing

    Used for: Full audio file transcription (STT invoke endpoint)

    STT-specific metrics:
    - real_time_factor: RTF = audio_duration / processing_time (<1.0 = faster than real-time)
    - audio_duration_ms: Duration of input audio
    - throughput_seconds_per_second: Seconds of audio processed per second of compute time
    """

    # Core timing (required)
    latency_ms: float = Field(..., description="Total end-to-end latency")
    processing_time_ms: float = Field(..., description="Actual processing time")

    # Common optional
    queue_time_ms: float = Field(default=0.0, description="Time spent in queue")

    # STT-specific metrics
    real_time_factor: float | None = Field(
        None,
        description="RTF = audio_duration / processing_time (<1.0 = faster than real-time)",
    )
    audio_duration_ms: float | None = Field(
        None, description="Input audio duration in ms"
    )
    throughput_seconds_per_second: float | None = Field(
        None, description="Seconds of audio processed per second of compute time"
    )


class STTStreamPerformanceMetrics(BaseModel):
    """
    Performance metrics for STT streaming mode

    Used for: Real-time streaming transcription (WebSocket STT)

    Industry benchmarks (2024):
    - TTFT: ~270-300ms (Gladia, Deepgram)
    - Final latency: ~700ms (Gladia)

    STT-specific metrics:
    - final_latency_ms: Time to final/complete transcription
    - tokens_per_second: Token generation speed
    - partial_update_latency_ms: Time between partial updates
    """

    # Critical streaming metrics (required)
    time_to_first_token_ms: float = Field(
        ..., description="TTFT - Time to first token (critical for UX!)"
    )
    chunk_latency_ms: float = Field(..., description="Latency for this specific chunk")

    # STT-specific streaming metrics
    final_latency_ms: float | None = Field(
        None, description="Time to final/complete output"
    )
    tokens_per_second: float | None = Field(None, description="Token generation speed")
    partial_update_latency_ms: float | None = Field(
        None, description="Time between partial updates"
    )

    # Total tracking
    total_duration_ms: float | None = Field(
        None, description="Total duration since stream started"
    )


# ============================================================================
# TTS Performance Metrics
# ============================================================================


class TTSInvokePerformanceMetrics(BaseModel):
    """
    Performance metrics for TTS invoke/batch mode processing

    Used for: Full text synthesis (TTS invoke endpoint)

    TTS-specific metrics:
    - characters_per_second: Text→audio synthesis speed
    - audio_generation_ratio: audio_duration / processing_time
    """

    # Core timing (required)
    latency_ms: float = Field(..., description="Total end-to-end latency")
    processing_time_ms: float = Field(..., description="Actual processing time")

    # Common optional
    queue_time_ms: float = Field(default=0.0, description="Time spent in queue")

    # TTS-specific metrics
    characters_per_second: float | None = Field(
        None, description="Text→audio synthesis speed"
    )
    audio_generation_ratio: float | None = Field(
        None,
        description="audio_duration / processing_time (how much audio generated per processing second)",
    )


class TTSStreamPerformanceMetrics(BaseModel):
    """
    Performance metrics for TTS streaming mode

    Used for: Real-time text-to-speech streaming (WebSocket TTS)

    Industry benchmarks (2024):
    - TTFB: ~100-200ms (Elevenlabs ~100ms, Deepgram ~150ms)
    - Leading silence: ~100-200ms

    TTS-specific metrics:
    - time_to_first_byte_ms: TTFB - When first audio data arrives
    - leading_silence_ms: Silence before speech starts
    - chunk_generation_rate_ms: Time between audio chunks
    """

    # Critical streaming metrics (required)
    time_to_first_token_ms: float = Field(..., description="TTFT - Time to first token")
    chunk_latency_ms: float = Field(..., description="Latency for this specific chunk")

    # TTS-specific streaming metrics
    time_to_first_byte_ms: float | None = Field(
        None, description="TTFB - When first audio data arrives"
    )
    leading_silence_ms: float | None = Field(
        None, description="Silence before speech starts"
    )
    chunk_generation_rate_ms: float | None = Field(
        None, description="Time between audio chunks"
    )

    # Total tracking
    total_duration_ms: float | None = Field(
        None, description="Total duration since stream started"
    )


# ============================================================================
# Quality Metrics (Shared by STT and TTS)
# ============================================================================


class QualityMetrics(BaseModel):
    """
    Quality metrics for both STT and TTS

    All fields optional since not all engines provide all metrics

    STT quality:
    - confidence_score: Model confidence (0-1)
    - word_error_rate: WER - 5% = 1 error per 20 words

    TTS quality:
    - mean_opinion_score: MOS - Audio quality rating (1-5 scale)

    Audio quality (both):
    - signal_to_noise_ratio: Audio quality metric (dB)
    """

    # STT quality
    confidence_score: float | None = Field(
        None, ge=0.0, le=1.0, description="Model confidence (0-1)"
    )
    word_error_rate: float | None = Field(
        None, description="WER for STT - 5% = 1 error per 20 words"
    )

    # Audio quality (both STT and TTS)
    signal_to_noise_ratio: float | None = Field(
        None, description="Audio quality metric (dB)"
    )

    # TTS quality
    mean_opinion_score: float | None = Field(
        None, ge=1.0, le=5.0, description="MOS - Audio quality rating (1-5 scale)"
    )
