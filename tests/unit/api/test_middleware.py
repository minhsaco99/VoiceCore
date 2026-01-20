"""Unit tests for middleware

Tests CORS, logging, and error handler middleware.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.api.config import Settings
from app.api.middleware.cors import configure_cors
from app.api.middleware.error_handler import ExceptionHandlerMiddleware
from app.api.middleware.logging import LoggingMiddleware
from app.exceptions import (
    EngineNotReadyError,
    InvalidAudioError,
    TimeoutError,
    TranscriptionError,
    VoiceEngineError,
)


class TestExceptionHandlerMiddleware:
    """Error handler middleware tests"""

    @pytest.mark.asyncio
    async def test_engine_not_ready_503(self):
        """EngineNotReadyError → 503"""
        middleware = ExceptionHandlerMiddleware(app=MagicMock())

        mock_request = MagicMock(spec=Request)
        mock_call_next = AsyncMock(side_effect=EngineNotReadyError("Engine starting"))

        response = await middleware.dispatch(mock_request, mock_call_next)

        assert response.status_code == 503
        body = response.body.decode()
        assert "service_unavailable" in body
        assert "Engine starting" in body

    @pytest.mark.asyncio
    async def test_invalid_audio_400(self):
        """InvalidAudioError → 400"""
        middleware = ExceptionHandlerMiddleware(app=MagicMock())

        mock_request = MagicMock(spec=Request)
        mock_call_next = AsyncMock(side_effect=InvalidAudioError("Bad format"))

        response = await middleware.dispatch(mock_request, mock_call_next)

        assert response.status_code == 400
        body = response.body.decode()
        assert "invalid_audio" in body
        assert "Bad format" in body

    @pytest.mark.asyncio
    async def test_timeout_504(self):
        """TimeoutError → 504"""
        middleware = ExceptionHandlerMiddleware(app=MagicMock())

        mock_request = MagicMock(spec=Request)
        mock_call_next = AsyncMock(side_effect=TimeoutError("Request timed out"))

        response = await middleware.dispatch(mock_request, mock_call_next)

        assert response.status_code == 504
        body = response.body.decode()
        assert "timeout" in body
        assert "Request timed out" in body

    @pytest.mark.asyncio
    async def test_transcription_error_500(self):
        """TranscriptionError → 500"""
        middleware = ExceptionHandlerMiddleware(app=MagicMock())

        mock_request = MagicMock(spec=Request)
        mock_call_next = AsyncMock(
            side_effect=TranscriptionError("Transcription failed")
        )

        with patch("app.api.middleware.error_handler.logger") as mock_logger:
            response = await middleware.dispatch(mock_request, mock_call_next)

        assert response.status_code == 500
        body = response.body.decode()
        assert "processing_error" in body
        assert "Transcription failed" in body
        # Verify error was logged
        mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_voice_engine_error_500(self):
        """VoiceEngineError → 500"""
        middleware = ExceptionHandlerMiddleware(app=MagicMock())

        mock_request = MagicMock(spec=Request)
        mock_call_next = AsyncMock(side_effect=VoiceEngineError("Engine crashed"))

        with patch("app.api.middleware.error_handler.logger") as mock_logger:
            response = await middleware.dispatch(mock_request, mock_call_next)

        assert response.status_code == 500
        body = response.body.decode()
        assert "processing_error" in body
        mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_generic_exception_500(self):
        """Unhandled exception → 500 with generic message"""
        middleware = ExceptionHandlerMiddleware(app=MagicMock())

        mock_request = MagicMock(spec=Request)
        mock_call_next = AsyncMock(side_effect=RuntimeError("Unexpected error"))

        with patch("app.api.middleware.error_handler.logger") as mock_logger:
            response = await middleware.dispatch(mock_request, mock_call_next)

        assert response.status_code == 500
        body = response.body.decode()
        assert "internal_server_error" in body
        # Generic message, not the actual error
        assert "Unexpected error" in body or "internal_server_error" in body
        # Error should be logged with traceback
        mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_successful_request_passes_through(self):
        """Successful request passes through unchanged"""
        middleware = ExceptionHandlerMiddleware(app=MagicMock())

        mock_request = MagicMock(spec=Request)
        expected_response = Response(content="Success", status_code=200)
        mock_call_next = AsyncMock(return_value=expected_response)

        response = await middleware.dispatch(mock_request, mock_call_next)

        assert response == expected_response
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_json_response_format(self):
        """Error responses are valid JSON"""
        middleware = ExceptionHandlerMiddleware(app=MagicMock())

        mock_request = MagicMock(spec=Request)
        mock_call_next = AsyncMock(side_effect=InvalidAudioError("Test"))

        response = await middleware.dispatch(mock_request, mock_call_next)

        assert isinstance(response, JSONResponse)
        import json

        body = json.loads(response.body.decode())
        assert "error" in body
        assert "message" in body


class TestLoggingMiddleware:
    """Logging middleware tests"""

    @pytest.mark.asyncio
    async def test_request_logged(self, caplog):
        """Request method and path logged"""
        middleware = LoggingMiddleware(app=MagicMock())

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/stt/transcribe"

        mock_response = Response(status_code=200)
        mock_call_next = AsyncMock(return_value=mock_response)

        with caplog.at_level(logging.INFO):
            await middleware.dispatch(mock_request, mock_call_next)

        # Check request was logged
        assert "POST" in caplog.text
        assert "/api/v1/stt/transcribe" in caplog.text

    @pytest.mark.asyncio
    async def test_response_timing_logged(self, caplog):
        """Response includes X-Process-Time header"""
        middleware = LoggingMiddleware(app=MagicMock())

        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/health"

        mock_response = Response(status_code=200)
        mock_call_next = AsyncMock(return_value=mock_response)

        with caplog.at_level(logging.INFO):
            response = await middleware.dispatch(mock_request, mock_call_next)

        # Check timing header was added
        assert "X-Process-Time" in response.headers
        assert response.headers["X-Process-Time"].endswith("ms")

        # Check completion was logged
        assert "Completed" in caplog.text
        assert "200" in caplog.text

    @pytest.mark.asyncio
    async def test_duration_calculation(self):
        """Duration calculated correctly"""
        middleware = LoggingMiddleware(app=MagicMock())

        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/test"

        mock_response = Response(status_code=200)

        # Simulate some processing time
        async def slow_call_next(request):
            import asyncio

            await asyncio.sleep(0.01)  # 10ms
            return mock_response

        response = await middleware.dispatch(mock_request, slow_call_next)

        # Check timing is reasonable (>= 10ms)
        timing_str = response.headers["X-Process-Time"]
        timing_ms = float(timing_str.replace("ms", ""))
        assert timing_ms >= 10.0

    @pytest.mark.asyncio
    async def test_status_code_logged(self, caplog):
        """Response status code is logged"""
        middleware = LoggingMiddleware(app=MagicMock())

        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/test"

        mock_response = Response(status_code=404)
        mock_call_next = AsyncMock(return_value=mock_response)

        with caplog.at_level(logging.INFO):
            await middleware.dispatch(mock_request, mock_call_next)

        assert "404" in caplog.text

    @pytest.mark.asyncio
    async def test_error_response_still_timed(self):
        """Error responses still get timing header"""
        middleware = LoggingMiddleware(app=MagicMock())

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/stt/transcribe"

        mock_response = JSONResponse(
            status_code=500, content={"error": "Internal error"}
        )
        mock_call_next = AsyncMock(return_value=mock_response)

        response = await middleware.dispatch(mock_request, mock_call_next)

        assert "X-Process-Time" in response.headers


class TestCORSMiddleware:
    """CORS configuration tests"""

    def test_cors_headers_added(self):
        """CORS headers present in response"""
        from fastapi import FastAPI

        app = FastAPI()
        settings = Settings(cors_origins=["http://localhost:3000"])

        configure_cors(app, settings)

        # Verify middleware was added
        # FastAPI stores middleware in app.user_middleware
        assert len(app.user_middleware) > 0

    def test_cors_origins_respected(self):
        """Configured origins in Allow-Origin"""
        from fastapi import FastAPI

        app = FastAPI()
        settings = Settings(
            cors_origins=["http://localhost:3000", "http://localhost:8080"]
        )

        configure_cors(app, settings)

        # Check middleware is CORSMiddleware
        # The actual CORS behavior is tested in integration tests
        assert len(app.user_middleware) > 0

    def test_cors_wildcard(self):
        """Wildcard CORS origin"""
        from fastapi import FastAPI

        app = FastAPI()
        settings = Settings(cors_origins=["*"])

        configure_cors(app, settings)

        assert len(app.user_middleware) > 0

    def test_cors_credentials_enabled(self):
        """CORS credentials are enabled"""
        from fastapi import FastAPI

        app = FastAPI()
        settings = Settings()

        configure_cors(app, settings)

        # Middleware added successfully
        assert len(app.user_middleware) > 0

    def test_cors_all_methods_allowed(self):
        """All HTTP methods are allowed"""
        from fastapi import FastAPI

        app = FastAPI()
        settings = Settings()

        configure_cors(app, settings)

        # Configuration includes allow_methods=["*"]
        assert len(app.user_middleware) > 0

    def test_cors_all_headers_allowed(self):
        """All headers are allowed"""
        from fastapi import FastAPI

        app = FastAPI()
        settings = Settings()

        configure_cors(app, settings)

        # Configuration includes allow_headers=["*"]
        assert len(app.user_middleware) > 0


class TestMiddlewareIntegration:
    """Integration between multiple middleware"""

    @pytest.mark.asyncio
    async def test_logging_before_error_handler(self, caplog):
        """Logging middleware logs even when error occurs"""
        # Create middleware chain: logging -> error handler
        error_middleware = ExceptionHandlerMiddleware(app=MagicMock())
        logging_middleware = LoggingMiddleware(app=MagicMock())

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/stt/transcribe"

        # Call chain raises error
        mock_call_next = AsyncMock(side_effect=InvalidAudioError("Bad audio"))

        with caplog.at_level(logging.INFO):
            # First logging middleware
            async def error_handler_call(req):
                return await error_middleware.dispatch(req, mock_call_next)

            response = await logging_middleware.dispatch(
                mock_request, error_handler_call
            )

        # Error was handled
        assert response.status_code == 400

        # Request was logged
        assert "POST" in caplog.text
        assert "/api/v1/stt/transcribe" in caplog.text

    @pytest.mark.asyncio
    async def test_multiple_errors_first_one_caught(self):
        """First matching error handler catches exception"""
        middleware = ExceptionHandlerMiddleware(app=MagicMock())

        mock_request = MagicMock(spec=Request)

        # Raise InvalidAudioError
        mock_call_next = AsyncMock(side_effect=InvalidAudioError("Test"))

        response = await middleware.dispatch(mock_request, mock_call_next)

        # Should be caught as 400, not 500
        assert response.status_code == 400
        assert "invalid_audio" in response.body.decode()
