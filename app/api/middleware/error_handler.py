import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.exceptions import (
    EngineNotReadyError,
    InvalidAudioError,
    TimeoutError,
    TranscriptionError,
    VoiceEngineError,
)

logger = logging.getLogger(__name__)


class ExceptionHandlerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        try:
            return await call_next(request)
        except EngineNotReadyError as exc:
            return JSONResponse(
                status_code=503,
                content={"error": "service_unavailable", "message": str(exc)},
            )
        except InvalidAudioError as exc:
            return JSONResponse(
                status_code=400,
                content={"error": "invalid_audio", "message": str(exc)},
            )
        except TimeoutError as exc:
            return JSONResponse(
                status_code=504,
                content={"error": "timeout", "message": str(exc)},
            )
        except (TranscriptionError, VoiceEngineError) as exc:
            logger.error(f"Engine error: {exc}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": "processing_error", "message": str(exc)},
            )
        except Exception as exc:
            logger.error(f"Unhandled error: {exc}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "internal_server_error",
                    "message": "Unexpected error",
                },
            )
