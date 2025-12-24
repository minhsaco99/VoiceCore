"""Configuration for Whisper STT engine"""

from pydantic import Field, field_validator

from app.models.engine import EngineConfig


class WhisperConfig(EngineConfig):
    """
    Configuration for Whisper STT engine

    Extends EngineConfig with Whisper-specific parameters:
    - Inherits: model_name, device, max_workers, timeout_seconds
    - Adds: compute_type, beam_size, language, temperature, vad_filter, etc.

    Additional parameters can be passed via engine_params in requests.
    """

    # Override model_name validator for Whisper-specific models
    @field_validator("model_name")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate Whisper model name"""
        valid = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        if v not in valid:
            raise ValueError(f"Whisper model must be one of {valid}")
        return v

    # Whisper-specific fields (commonly used defaults)
    compute_type: str = Field(
        default="int8",
        description="Compute type: int8, float16, float32",
    )
    beam_size: int = Field(default=5, gt=0, description="Beam search size")
    language: str | None = Field(
        None, description="Default language hint (e.g., 'en', 'es')"
    )
    temperature: float | list[float] = Field(
        default=0.0,
        description="Temperature for sampling (0.0 = greedy, or list of fallback temps)",
    )
    vad_filter: bool = Field(
        default=False,
        description="Enable Voice Activity Detection filter",
    )
    condition_on_previous_text: bool = Field(
        default=True,
        description="Condition on previous text for better context",
    )
    compression_ratio_threshold: float | None = Field(
        default=2.4,
        description="If compression ratio > threshold, treat as failed",
    )
    log_prob_threshold: float | None = Field(
        default=-1.0,
        description="If avg log prob < threshold, treat as failed",
    )
    no_speech_threshold: float | None = Field(
        default=0.6,
        description="If no_speech prob > threshold, skip segment",
    )

    @field_validator("compute_type")
    @classmethod
    def validate_compute_type(cls, v: str) -> str:
        """Validate compute type"""
        valid = ["int8", "float16", "float32"]
        if v not in valid:
            raise ValueError(f"compute_type must be one of {valid}")
        return v
