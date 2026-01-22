from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_tts_engine
from app.engines.base import BaseTTSEngine

router = APIRouter()


@router.post("/synthesize")
async def synthesize_text(
    text: str = Query(..., description="Text to synthesize"),
    voice: str | None = Query(None, description="Voice name/ID to use"),
    speed: float = Query(1.0, gt=0, le=3.0, description="Speech speed multiplier"),
    engine_params: str | None = Query(None, description="JSON engine parameters"),
    tts_engine: BaseTTSEngine = Depends(get_tts_engine),
):
    """
    Synthesize text to speech (invoke mode)

    Query params:
    - engine: TTS engine name (required)
    - text: Text to synthesize (required)
    - voice: Optional voice name/ID
    - speed: Speech speed multiplier (0 < speed <= 3.0)
    - engine_params: Optional JSON engine parameters

    Returns complete audio with metrics.
    """
    raise HTTPException(501, "TTS not implemented yet")


@router.post("/synthesize/stream")
async def synthesize_text_stream(
    text: str = Query(..., description="Text to synthesize"),
    voice: str | None = Query(None, description="Voice name/ID to use"),
    speed: float = Query(1.0, gt=0, le=3.0, description="Speech speed multiplier"),
    engine_params: str | None = Query(None, description="JSON engine parameters"),
    tts_engine: BaseTTSEngine = Depends(get_tts_engine),
):
    """
    Synthesize text to speech with streaming

    Query params:
    - engine: TTS engine name (required)
    - text: Text to synthesize (required)
    - voice: Optional voice name/ID
    - speed: Speech speed multiplier (0 < speed <= 3.0)
    - engine_params: Optional JSON engine parameters

    Returns progressive audio chunks followed by final response.
    """
    raise HTTPException(501, "TTS streaming not implemented yet")
