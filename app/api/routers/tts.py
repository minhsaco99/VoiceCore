import json

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

from app.api.deps import get_optional_audio_upload, get_tts_engine
from app.api.registry import EngineRegistry
from app.engines.base import BaseTTSEngine
from app.models.engine import TTSChunk, TTSResponse

router = APIRouter()



@router.post("/synthesize", response_model=TTSResponse)
async def synthesize_text(
    text: str = Query(..., description="Text to synthesize"),
    prompt_text: str | None = Query(None, description="Text content of the reference audio (if provided)"),
    voice: str | None = Form(None, description="Voice ID to use"),
    audio: bytes | None = Depends(get_optional_audio_upload),
    speed: float = Query(1.0, description="Speed multiplier"),
    engine_params: str | None = Query(None, description="JSON engine parameters"),
    tts_engine: BaseTTSEngine = Depends(get_tts_engine),
):
    """
    Synthesize text to speech (invoke mode)

    Inputs:
    - **text** (Query): Text to synthesize (required)
    - **engine** (Query): TTS engine name (required)
    - **audio** (File): Optional reference audio for voice cloning (takes precedence over voice params)
    - **voice** (Form): Optional voice ID (used if audio is missing)
    - **speed** (Query): Speed multiplier (default 1.0)
    - **engine_params** (Query): Optional JSON string for extra engine params

    Returns complete audio with metrics.
    """
    # Validation: Must have either audio (speaker reference) or voice/default
    # Actually, some engines might have defaults, but user requested explicit "one of them must be present" logic?
    # User said: "Khi user gửi request thì bắt buộc 1 trong 2 phải có" (When user sends request, must have 1 of 2)
    
    if not audio and not voice:
        # Check if engine has a default voice? 
        # But user rule is strict: "Must have 1 of 2"
        raise HTTPException(
            400, "Either 'audio' file or 'voice' ID must be provided"
        )

    # Parse engine_params
    kwargs = {}
    if engine_params:
        try:
            kwargs = json.loads(engine_params)
        except json.JSONDecodeError as e:
            raise HTTPException(400, "Invalid engine_params JSON") from e

    if prompt_text:
        kwargs["prompt_text"] = prompt_text

    # Logic: Prioritize audio
    speaker_wav = audio if audio else None
    voice_id = voice if not audio else None

    result = await tts_engine.synthesize(
        text=text,
        voice=voice_id,
        speed=speed,
        speaker_wav=speaker_wav,
        **kwargs,
    )
    return result


@router.post("/synthesize/stream")
async def synthesize_text_stream(
    text: str = Query(..., description="Text to synthesize"),
    prompt_text: str | None = Query(None, description="Text content of the reference audio (if provided)"),
    voice: str | None = Form(None, description="Voice ID to use"),
    audio: bytes | None = Depends(get_optional_audio_upload),
    speed: float = Query(1.0, description="Speed multiplier"),
    engine_params: str | None = Query(None, description="JSON engine parameters"),
    tts_engine: BaseTTSEngine = Depends(get_tts_engine),
):
    """
    Synthesize text to speech with Server-Sent Events streaming

    Inputs:
    - **text** (Query): Text to synthesize (required)
    - **engine** (Query): TTS engine name (required)
    - **audio** (File): Optional reference audio
    - **voice** (Form): Optional voice ID
    - **speed** (Query): Speed multiplier
    - **engine_params** (Query): Optional JSON params

    Returns progressive audio chunks followed by final response.
    """
    if not audio and not voice:
        raise HTTPException(
            400, "Either 'audio' file or 'voice' ID must be provided"
        )

    parsed_params = {}
    if engine_params:
        try:
            parsed_params = json.loads(engine_params)
        except json.JSONDecodeError as e:
            raise HTTPException(400, "Invalid engine_params JSON") from e

    if prompt_text:
        parsed_params["prompt_text"] = prompt_text

    speaker_wav = audio if audio else None
    voice_id = voice if not audio else None

    async def event_generator():
        async for result in tts_engine.synthesize_stream(
            text=text,
            voice=voice_id,
            speed=speed,
            speaker_wav=speaker_wav,
            **parsed_params
        ):
            if isinstance(result, TTSChunk):
                yield {"event": "chunk", "data": result.model_dump_json()}
            elif isinstance(result, TTSResponse):
                yield {"event": "complete", "data": result.model_dump_json()}

    return EventSourceResponse(event_generator())


@router.websocket("/synthesize/ws")
async def synthesize_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time synthesis

    Protocol:
    1. Client sends config (JSON):
       {
         "engine": "voxcpm",
         "text": "Hello world",
         "voice": "nguyet",
         "speed": 1.0,
         "engine_params": {...}
       }
    2. Server sends: {"type": "chunk", "data": {...}} and {"type": "complete", "data": {...}}
    
    Note: WebSocket does not support multipart uploads easily. 
    For file-based cloning via WebSocket, client should send binary message first?
    Or stick to base64 in JSON?
    For now, keeping previous JSON-based config for WebSocket as user request focused on API (REST) inputs.
    """
    await websocket.accept()

    try:
        # Receive config
        config_data = await websocket.receive_json()
        engine_name = config_data.get("engine")
        text = config_data.get("text")
        
        if not engine_name:
            await websocket.send_json(
                {"type": "error", "message": "Missing 'engine' in config"}
            )
            await websocket.close(code=1008)
            return

        if not text:
            await websocket.send_json(
                {"type": "error", "message": "Missing 'text' in config"}
            )
            await websocket.close(code=1008)
            return

        # Optional params
        voice = config_data.get("voice")
        speed = config_data.get("speed", 1.0)
        engine_params = config_data.get("engine_params", {})
        
        # TODO: Handle speaker_wav for WebSocket if needed (e.g. base64 in config or binary msg)
        # For now, adhering to user request which implied "Form/File" which is REST-specific context.

        # Get engine from registry
        registry: EngineRegistry = websocket.app.state.engine_registry
        try:
            tts_engine = registry.get_tts(engine_name)
        except Exception as e:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close(code=1008)
            return

        # Stream
        async for result in tts_engine.synthesize_stream(
            text=text, 
            voice=voice, 
            speed=speed, 
            **engine_params
        ):
            if isinstance(result, TTSChunk):
                await websocket.send_json(
                    {"type": "chunk", "data": result.model_dump(mode='json')}
                )
            elif isinstance(result, TTSResponse):
                await websocket.send_json(
                    {"type": "complete", "data": result.model_dump(mode='json')}
                )

        await websocket.close()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
