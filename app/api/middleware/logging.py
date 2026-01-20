import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()

        logger.info(f"Request: {request.method} {request.url.path}")

        response = await call_next(request)

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Completed: {request.method} {request.url.path} "
            f"- {response.status_code} ({duration_ms:.2f}ms)"
        )

        response.headers["X-Process-Time"] = f"{duration_ms:.2f}ms"
        return response
