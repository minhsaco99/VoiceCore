import json

from fastapi import APIRouter, Depends, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from app.api.deps import get_tts_engine
from app.engines.base import BaseTTSEngine
from app.models.engine import TTSChunk, TTSResponse

router = APIRouter()


@router.post("/synthesize", response_model=TTSResponse)
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

    Returns complete audio (base64 encoded) with metrics.
    """
    kwargs = {}
    if engine_params:
        try:
            kwargs = json.loads(engine_params)
        except json.JSONDecodeError as e:
            raise HTTPException(400, "Invalid engine_params JSON") from e

    result = await tts_engine.synthesize(text, voice=voice, speed=speed, **kwargs)

    return result


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
    Event types: "chunk" (TTSChunk with base64 audio), "complete" (full response)
    """
    kwargs = {}
    if engine_params:
        try:
            kwargs = json.loads(engine_params)
        except json.JSONDecodeError as e:
            raise HTTPException(400, "Invalid engine_params JSON") from e

    async def event_generator():
        async for result in tts_engine.synthesize_stream(
            text, voice=voice, speed=speed, **kwargs
        ):
            if isinstance(result, TTSChunk):
                yield {"event": "chunk", "data": result.model_dump_json()}
            elif isinstance(result, TTSResponse):
                yield {"event": "complete", "data": result.model_dump_json()}

    return EventSourceResponse(event_generator())
