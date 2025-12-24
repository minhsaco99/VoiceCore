"""Configuration for Whisper STT engine"""
from pydantic import Field, field_validator

from app.models.engine import EngineConfig


class WhisperConfig(EngineConfig):
    """
    Configuration for Whisper STT engine

    Extends EngineConfig with Whisper-specific parameters:
    - Inherits: model_name, device, max_workers, timeout_seconds
    - Adds: compute_type, beam_size, language
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

    # Whisper-specific fields
    compute_type: str = Field(
        default="int8",
        description="Compute type: int8, float16, float32",
    )
    beam_size: int = Field(default=5, gt=0, description="Beam search size")
    language: str | None = Field(
        None, description="Default language hint (e.g., 'en', 'es')"
    )

    @field_validator("compute_type")
    @classmethod
    def validate_compute_type(cls, v: str) -> str:
        """Validate compute type"""
        valid = ["int8", "float16", "float32"]
        if v not in valid:
            raise ValueError(f"compute_type must be one of {valid}")
        return v
