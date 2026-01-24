"""
VoxCPM TTS Engine Configuration

Configuration class for VoxCPM - a tokenizer-free TTS system with voice cloning support.
"""

from pydantic import Field

from app.models.engine import EngineConfig


class VoxCPMConfig(EngineConfig):
    """
    Configuration for VoxCPM TTS Engine

    Attributes:
        model_name: HuggingFace model ID (e.g., "openbmb/VoxCPM-0.5B", "openbmb/VoxCPM1.5")
        device: Device to run on ("cpu", "cuda", "mps")
        cfg_value: LM guidance value for LocDiT (higher = better adherence to prompt)
        inference_timesteps: LocDiT inference timesteps (higher = better quality, slower)
        normalize: Enable external text normalization tool
        denoise: Enable external denoiser (restricts output to 16kHz)
        retry_badcase: Enable retrying mode for bad cases (unstoppable generation)
        retry_badcase_max_times: Maximum retry attempts for bad cases
        retry_badcase_ratio_threshold: Length restriction threshold for bad case detection
    """

    # VoxCPM-specific settings
    cfg_value: float = Field(
        default=2.0,
        ge=0.0,
        description="LM guidance on LocDiT, higher for better adherence to prompt",
    )
    inference_timesteps: int = Field(
        default=10,
        ge=1,
        le=50,
        description="LocDiT inference timesteps, higher for better quality",
    )
    normalize: bool = Field(
        default=False,
        description="Enable external text normalization tool",
    )
    denoise: bool = Field(
        default=False,
        description="Enable external denoiser (restricts to 16kHz)",
    )
    retry_badcase: bool = Field(
        default=True,
        description="Enable retrying mode for bad cases",
    )
    retry_badcase_max_times: int = Field(
        default=3,
        ge=1,
        description="Maximum retry attempts for bad cases",
    )
    retry_badcase_ratio_threshold: float = Field(
        default=6.0,
        ge=1.0,
        description="Length restriction threshold for bad case detection",
    )
