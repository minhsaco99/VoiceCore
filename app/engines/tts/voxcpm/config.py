from pydantic import Field

from app.models.engine import EngineConfig


class VoxCPMConfig(EngineConfig):
    """Configuration for VoxCPM TTS Engine"""

    prompt_wav_path: str | None = Field(
        default=None, description="Path to a prompt speech for voice cloning"
    )
    prompt_text: str | None = Field(
        default=None, description="Reference text for the prompt speech"
    )
    cfg_value: float = Field(default=2.0, description="LM guidance on LocDiT strength")
    inference_timesteps: int = Field(
        default=10, description="LocDiT inference timesteps"
    )
    normalize: bool = Field(default=False, description="Enable external TN tool")
    denoise: bool = Field(default=False, description="Enable external Denoise tool")
    retry_badcase: bool = Field(
        default=True, description="Enable retrying for bad cases"
    )
    retry_badcase_max_times: int = Field(default=3, description="Maximum retry times")
