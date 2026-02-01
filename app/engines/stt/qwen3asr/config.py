"""Configuration for Qwen3-ASR STT Engine"""

from pydantic import Field, field_validator

from app.models.engine import EngineConfig


class Qwen3ASRConfig(EngineConfig):
    """
    Configuration for Qwen3-ASR STT engine (vLLM backend)

    Extends EngineConfig with Qwen3-ASR specific parameters.

    Attributes:
        model_name: HuggingFace model ID (Qwen/Qwen3-ASR-1.7B or Qwen/Qwen3-ASR-0.6B)
        dtype: Torch dtype for model loading (bfloat16 recommended)
        gpu_memory_utilization: Fraction of GPU memory to use for vLLM
        max_inference_batch_size: Maximum batch size for inference
        max_new_tokens: Maximum tokens to generate (increase for long audio)
    """

    @field_validator("model_name")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate Qwen3-ASR model name"""
        valid = [
            "Qwen/Qwen3-ASR-1.7B",
            "Qwen/Qwen3-ASR-0.6B",
        ]
        # Allow local paths
        if v in valid or v.startswith("/") or v.startswith("./"):
            return v
        raise ValueError(f"Qwen3-ASR model must be one of {valid} or a local path")

    # vLLM backend specific fields
    gpu_memory_utilization: float = Field(
        default=0.7,
        ge=0.1,
        le=0.95,
        description="Fraction of GPU memory to use for vLLM",
    )
    max_inference_batch_size: int = Field(
        default=32,
        gt=0,
        description="Maximum batch size for inference. -1 for unlimited.",
    )
    max_new_tokens: int = Field(
        default=512,
        gt=0,
        description="Maximum tokens to generate. Increase for long audio.",
    )
    forced_aligner: str | None = Field(
        default=None, description="Path or name of ForcedAligner model for timestamps."
    )
    language: str | None = Field(
        default=None,
        description="Default language for transcription (e.g., 'en', 'vi').",
    )
