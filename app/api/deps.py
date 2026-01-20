from fastapi import Depends, HTTPException, Query, Request, UploadFile

from app.api.config import Settings
from app.api.registry import EngineRegistry
from app.engines.base import BaseSTTEngine, BaseTTSEngine


def get_settings() -> Settings:
    return Settings()


def get_engine_registry(request: Request) -> EngineRegistry:
    """Get engine registry from app.state"""
    registry = getattr(request.app.state, "engine_registry", None)
    if registry is None:
        raise HTTPException(503, "Engine registry not available")
    return registry


def get_stt_engine(
    engine: str = Query(..., description="STT engine name (e.g., 'whisper')"),
    registry: EngineRegistry = Depends(get_engine_registry),
) -> BaseSTTEngine:
    """
    Get STT engine by name from registry

    Raises:
        HTTPException 404: Engine not found
        HTTPException 503: Engine not ready
    """
    try:
        return registry.get_stt(engine)
    except Exception as e:
        raise HTTPException(404, str(e)) from e


def get_tts_engine(
    engine: str = Query(..., description="TTS engine name"),
    registry: EngineRegistry = Depends(get_engine_registry),
) -> BaseTTSEngine:
    """Get TTS engine by name from registry"""
    try:
        return registry.get_tts(engine)
    except Exception as e:
        raise HTTPException(404, str(e)) from e


async def validate_audio_upload(
    audio: UploadFile,
    settings: Settings = Depends(get_settings),
) -> bytes:
    """Validate and read uploaded audio file"""
    max_size = settings.max_audio_size_mb * 1024 * 1024
    audio_bytes = await audio.read(max_size + 1)

    if len(audio_bytes) > max_size:
        raise HTTPException(
            413, f"Audio too large (max {settings.max_audio_size_mb}MB)"
        )
    if len(audio_bytes) == 0:
        raise HTTPException(400, "Audio file is empty")

    return audio_bytes
