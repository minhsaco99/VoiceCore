from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.config import Settings
from app.api.deps import get_engine_registry, get_settings
from app.api.registry import EngineRegistry

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadinessResponse(BaseModel):
    ready: bool
    stt_engines: dict[str, bool]
    tts_engines: dict[str, bool]


class EngineInfo(BaseModel):
    name: str
    type: str
    ready: bool
    engine_name: str
    supported_formats: list[str] | None = None
    supported_voices: list[str] | None = None


@router.get("/health", response_model=HealthResponse)
async def health_check(settings: Settings = Depends(get_settings)):
    """Basic health check"""
    return HealthResponse(status="healthy", version=settings.version)


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check(registry: EngineRegistry = Depends(get_engine_registry)):
    """Check if engines are ready"""
    stt_status = {
        name: engine.is_ready() for name, engine in registry.list_stt_engines().items()
    }
    tts_status = {
        name: engine.is_ready() for name, engine in registry.list_tts_engines().items()
    }

    return ReadinessResponse(
        ready=all(stt_status.values()) and all(tts_status.values()),
        stt_engines=stt_status,
        tts_engines=tts_status,
    )


@router.get("/engines")
async def list_engines(registry: EngineRegistry = Depends(get_engine_registry)):
    """List available engines (discovery endpoint)"""
    engines = []

    # List STT engines
    for name, engine in registry.list_stt_engines().items():
        engines.append(
            EngineInfo(
                name=name,
                type="stt",
                ready=engine.is_ready(),
                engine_name=engine.engine_name,
                supported_formats=engine.supported_formats,
            )
        )

    # List TTS engines
    for name, engine in registry.list_tts_engines().items():
        engines.append(
            EngineInfo(
                name=name,
                type="tts",
                ready=engine.is_ready(),
                engine_name=engine.engine_name,
                supported_voices=engine.supported_voices,
            )
        )

    return {"engines": engines}


@router.get("/metrics")
async def get_metrics():
    """Performance metrics (placeholder)"""
    return {"message": "Metrics endpoint placeholder"}
