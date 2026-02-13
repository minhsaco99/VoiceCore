"""
CosyVoice3 TTS Engine Configuration

Configuration for CosyVoice3 HTTP client engine.
Connects to an external CosyVoice3 FastAPI server for TTS synthesis.
"""

from pydantic import BaseModel, Field

from app.models.engine import EngineConfig


class VoiceConfig(BaseModel):
    """Configuration for a single voice (zero-shot voice cloning)."""

    prompt_wav_path: str = Field(
        ..., description="Path to the prompt WAV file for voice cloning"
    )
    prompt_text: str = Field(..., description="Transcript of the prompt WAV file")
    description: str = Field(
        default="", description="Human-readable description of the voice"
    )


class CosyVoice3Config(EngineConfig):
    """
    Configuration for CosyVoice3 TTS Engine (HTTP Client)

    Attributes:
        model_name: Model identifier (for display/logging only)
        service_url: URL of the CosyVoice3 FastAPI server
        sample_rate: Output audio sample rate (CosyVoice3 default: 22050)
        voices: Mapping of voice names to VoiceConfig (prompt_wav_path + prompt_text)
        default_voice: Default voice name when none specified in request
        speed: Default speech speed multiplier
        system_prompt: System prompt prefix for CosyVoice3 prompt_text
        connect_timeout: HTTP connection timeout in seconds
        read_timeout: HTTP read timeout in seconds (None = no timeout for streaming)
    """

    service_url: str = Field(
        default="http://localhost:50000",
        description="URL of the CosyVoice3 FastAPI server",
    )
    sample_rate: int = Field(
        default=22050,
        ge=8000,
        le=48000,
        description="Output audio sample rate in Hz",
    )
    voices: dict[str, VoiceConfig] = Field(
        default_factory=dict,
        description="Voice name to VoiceConfig mapping for zero-shot cloning",
    )
    default_voice: str | None = Field(
        default=None,
        description="Default voice name when none specified",
    )
    speed: float = Field(
        default=1.0,
        gt=0.0,
        le=3.0,
        description="Default speech speed multiplier",
    )
    system_prompt: str = Field(
        default="You are a helpful assistant.",
        description="System prompt prefix for CosyVoice3",
    )
    connect_timeout: float = Field(
        default=10.0,
        ge=1.0,
        description="HTTP connection timeout in seconds",
    )
    read_timeout: float | None = Field(
        default=None,
        description="HTTP read timeout (None = no timeout for long synthesis)",
    )
