import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sse_starlette.sse import EventSourceResponse

from app.api.deps import get_tts_engine
from app.engines.base import BaseTTSEngine
from app.models.engine import TTSChunk, TTSResponse

router = APIRouter()


@router.post("/synthesize", response_model=TTSResponse)
async def synthesize_text(
    text: str = Form(..., description="Text to synthesize"),
    reference_audio: UploadFile | None = File(
        None, description="Reference audio for voice cloning"
    ),
    reference_text: str | None = Form(
        None, description="Transcript of reference audio"
    ),
    engine_params: str | None = Form(None, description="JSON engine parameters"),
    voice: str | None = Query(None, description="Voice name/ID to use"),
    speed: float = Query(1.0, gt=0, le=3.0, description="Speech speed multiplier"),
    tts_engine: BaseTTSEngine = Depends(get_tts_engine),
):
    """
    Synthesize text to speech (invoke mode)

    Query params:
    - engine: TTS engine name (required)
    - voice: Optional voice name/ID
    - speed: Speech speed multiplier (0 < speed <= 3.0)

    Form params:
    - text: Text to synthesize (required)
    - reference_audio: Optional reference audio file for voice cloning
    - reference_text: Optional transcript of reference audio
    - engine_params: Optional JSON engine parameters

    Returns complete audio (base64 encoded) with metrics.
    """
    kwargs = {}
    if engine_params:
        try:
            kwargs = json.loads(engine_params)
        except json.JSONDecodeError as e:
            raise HTTPException(400, "Invalid engine_params JSON") from e

    temp_file = None
    try:
        reference_audio_path = None
        if reference_audio:
            suffix = Path(reference_audio.filename or ".wav").suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await reference_audio.read())
                temp_file = tmp
            reference_audio_path = temp_file.name

        result = await tts_engine.synthesize(
            text,
            voice=voice,
            speed=speed,
            reference_audio_path=reference_audio_path,
            reference_text=reference_text,
            **kwargs,
        )
        return result

    finally:
        if temp_file:
            Path(temp_file.name).unlink(missing_ok=True)


@router.post("/synthesize/stream")
async def synthesize_text_stream(
    text: str = Form(..., description="Text to synthesize"),
    reference_audio: UploadFile | None = File(
        None, description="Reference audio for voice cloning"
    ),
    reference_text: str | None = Form(
        None, description="Transcript of reference audio"
    ),
    engine_params: str | None = Form(None, description="JSON engine parameters"),
    voice: str | None = Query(None, description="Voice name/ID to use"),
    speed: float = Query(1.0, gt=0, le=3.0, description="Speech speed multiplier"),
    tts_engine: BaseTTSEngine = Depends(get_tts_engine),
):
    """
    Synthesize text to speech with streaming

    Query params:
    - engine: TTS engine name (required)
    - voice: Optional voice name/ID
    - speed: Speech speed multiplier (0 < speed <= 3.0)

    Form params:
    - text: Text to synthesize (required)
    - reference_audio: Optional reference audio file for voice cloning
    - reference_text: Optional transcript of reference audio
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

    temp_file = None
    reference_audio_path = None
    if reference_audio:
        suffix = Path(reference_audio.filename or ".wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await reference_audio.read())
            temp_file = tmp
        reference_audio_path = temp_file.name

    async def event_generator():
        try:
            async for result in tts_engine.synthesize_stream(
                text,
                voice=voice,
                speed=speed,
                reference_audio_path=reference_audio_path,
                reference_text=reference_text,
                **kwargs,
            ):
                if isinstance(result, TTSChunk):
                    yield {"event": "chunk", "data": result.model_dump_json()}
                elif isinstance(result, TTSResponse):
                    yield {"event": "complete", "data": result.model_dump_json()}
        finally:
            if temp_file:
                Path(temp_file.name).unlink(missing_ok=True)

    return EventSourceResponse(event_generator())
