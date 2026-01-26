import json
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    Form,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from sse_starlette.sse import EventSourceResponse

from app.api.deps import get_stt_engine, validate_audio_upload
from app.api.registry import EngineRegistry
from app.engines.base import BaseSTTEngine
from app.models.engine import STTChunk, STTResponse

router = APIRouter()


@router.post("/transcribe", response_model=STTResponse)
async def transcribe_audio(
    audio: Annotated[bytes, Depends(validate_audio_upload)],
    language: str | None = Query(None, description="Language hint"),
    engine_params: str | None = Form(None, description="JSON engine parameters"),
    stt_engine: BaseSTTEngine = Depends(get_stt_engine),
):
    """
    Transcribe audio to text (invoke mode)

    Query params:
    - engine: STT engine name (required, e.g., "whisper")
    - language: Optional language hint

    Form params:
    - engine_params: Optional JSON engine parameters

    Returns complete transcription with segments and metrics.
    """
    kwargs = {}
    if engine_params:
        try:
            kwargs = json.loads(engine_params)
        except json.JSONDecodeError as e:
            raise HTTPException(400, "Invalid engine_params JSON") from e

    result = await stt_engine.transcribe(audio, language=language, **kwargs)
    return result


@router.post("/transcribe/stream")
async def transcribe_audio_stream(
    audio: Annotated[bytes, Depends(validate_audio_upload)],
    language: str | None = Query(None, description="Language hint"),
    engine_params: str | None = Form(None, description="JSON engine parameters"),
    stt_engine: BaseSTTEngine = Depends(get_stt_engine),
):
    """
    Transcribe audio with Server-Sent Events streaming

    Query params:
    - engine: STT engine name (required, e.g., "whisper")
    - language: Optional language hint

    Form params:
    - engine_params: Optional JSON engine parameters

    Returns progressive chunks followed by final response.
    Event types: "chunk" (STTChunk), "complete" (STTResponse)
    """
    kwargs = {}
    if engine_params:
        try:
            kwargs = json.loads(engine_params)
        except json.JSONDecodeError as e:
            raise HTTPException(400, "Invalid engine_params JSON") from e

    async def event_generator():
        async for result in stt_engine.transcribe_stream(
            audio, language=language, **kwargs
        ):
            if isinstance(result, STTChunk):
                yield {"event": "chunk", "data": result.model_dump_json()}
            elif isinstance(result, STTResponse):
                yield {"event": "complete", "data": result.model_dump_json()}

    return EventSourceResponse(event_generator())


@router.websocket("/transcribe/ws")
async def transcribe_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time transcription

    Protocol:
    1. Client sends config (JSON): {"engine": "whisper", "language": "en", "engine_params": {...}}
    2. Client sends audio chunks (binary)
    3. Client sends "END" (text) or disconnects
    4. Server sends: {"type": "chunk", "data": {...}} and {"type": "complete", "data": {...}}
    """
    await websocket.accept()

    try:
        # Receive config
        config_data = await websocket.receive_json()
        engine_name = config_data.get("engine")
        language = config_data.get("language")
        engine_params = config_data.get("engine_params", {})

        if not engine_name:
            await websocket.send_json(
                {"type": "error", "message": "Missing 'engine' in config"}
            )
            await websocket.close(code=1008)
            return

        # Get engine from registry
        registry: EngineRegistry = websocket.app.state.engine_registry
        try:
            stt_engine = registry.get_stt(engine_name)
        except Exception as e:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close(code=1008)
            return

        # Accumulate audio
        audio_chunks = []
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break
            if message["type"] == "websocket.receive":
                if "bytes" in message:
                    audio_chunks.append(message["bytes"])
                elif "text" in message and message["text"] == "END":
                    break

        # Process
        if audio_chunks:
            audio_bytes = b"".join(audio_chunks)
            async for result in stt_engine.transcribe_stream(
                audio_bytes, language=language, **engine_params
            ):
                if isinstance(result, STTChunk):
                    await websocket.send_json(
                        {"type": "chunk", "data": result.model_dump()}
                    )
                elif isinstance(result, STTResponse):
                    await websocket.send_json(
                        {"type": "complete", "data": result.model_dump()}
                    )

        await websocket.close()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        await websocket.close(code=1011)
