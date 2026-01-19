from fastapi import APIRouter, HTTPException

from app.models.engine import TTSRequest

router = APIRouter()


@router.post("/synthesize")
async def synthesize_text(request: TTSRequest):
    """TTS synthesis (placeholder - not implemented yet)"""
    raise HTTPException(501, "TTS not implemented yet")


@router.post("/synthesize/stream")
async def synthesize_text_stream(request: TTSRequest):
    """TTS streaming (placeholder - not implemented yet)"""
    raise HTTPException(501, "TTS streaming not implemented yet")
