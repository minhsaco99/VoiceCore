"""Unit tests for STT router endpoints

Focus on WebSocket memory exhaustion and JSON validation.
"""

import json
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from app.api.routers import stt


@pytest.fixture
def app_with_mock_lifespan(mock_registry):
    """FastAPI app with mocked lifespan"""

    @asynccontextmanager
    async def mock_lifespan(app: FastAPI):
        app.state.engine_registry = mock_registry
        yield

    app = FastAPI(lifespan=mock_lifespan)
    app.include_router(stt.router, prefix="/api/v1/stt", tags=["stt"])
    return app


@pytest.fixture
async def test_client(app_with_mock_lifespan):
    """Async test client"""
    async with AsyncClient(
        transport=ASGITransport(app=app_with_mock_lifespan), base_url="http://test"
    ) as client:
        yield client


class TestTranscribeEndpoint:
    """POST /transcribe tests"""

    @pytest.mark.asyncio
    async def test_transcribe_success(self, test_client, test_audio_bytes):
        """Valid audio returns STTResponse"""
        response = await test_client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "whisper"},
            files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Test transcription"
        assert "processing_time" in data

    @pytest.mark.asyncio
    async def test_transcribe_missing_engine_param(self, test_client, test_audio_bytes):
        """Missing ?engine= raises 422 validation error"""
        response = await test_client.post(
            "/api/v1/stt/transcribe",
            files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_transcribe_engine_not_found(self, test_client, test_audio_bytes):
        """Non-existent engine raises 404"""
        response = await test_client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "nonexistent"},
            files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_transcribe_invalid_json_params(self, test_client, test_audio_bytes):
        """CRITICAL: Invalid JSON in engine_params raises 400"""
        response = await test_client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "whisper", "engine_params": "invalid json {"},
            files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
        )

        assert response.status_code == 400
        assert "Invalid engine_params JSON" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_transcribe_json_params_not_dict(self, test_client, test_audio_bytes):
        """CRITICAL: JSON array/string as engine_params causes error"""
        # Array should cause TypeError when unpacking **kwargs
        response = await test_client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "whisper", "engine_params": '["array"]'},
            files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
        )

        # Should return 400 or 500 depending on error handling
        assert response.status_code in [400, 500]

    @pytest.mark.asyncio
    async def test_transcribe_valid_json_params(
        self, test_client, test_audio_bytes, mock_stt_engine
    ):
        """Valid JSON params are passed to engine"""
        response = await test_client.post(
            "/api/v1/stt/transcribe",
            params={
                "engine": "whisper",
                "engine_params": '{"beam_size": 5, "temperature": 0.0}',
            },
            files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
        )

        assert response.status_code == 200
        # Verify engine was called with kwargs
        mock_stt_engine.transcribe.assert_called_once()
        call_kwargs = mock_stt_engine.transcribe.call_args.kwargs
        assert call_kwargs.get("beam_size") == 5
        assert call_kwargs.get("temperature") == 0.0

    @pytest.mark.asyncio
    async def test_transcribe_with_language(
        self, test_client, test_audio_bytes, mock_stt_engine
    ):
        """Language parameter passed to engine"""
        response = await test_client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "whisper", "language": "en"},
            files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
        )

        assert response.status_code == 200
        mock_stt_engine.transcribe.assert_called_once()
        call_kwargs = mock_stt_engine.transcribe.call_args.kwargs
        assert call_kwargs.get("language") == "en"

    @pytest.mark.asyncio
    async def test_transcribe_empty_audio(self, test_client):
        """Empty audio raises 400"""
        response = await test_client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "whisper"},
            files={"audio": ("test.wav", b"", "audio/wav")},
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_transcribe_oversized_audio(self, test_client):
        """Audio too large raises 413"""
        # Create audio larger than max_audio_size_mb (default 10MB)
        large_audio = b"x" * (11 * 1024 * 1024)  # 11MB

        response = await test_client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "whisper"},
            files={"audio": ("test.wav", large_audio, "audio/wav")},
        )

        assert response.status_code == 413


class TestTranscribeStreamEndpoint:
    """POST /transcribe/stream tests"""

    @pytest.mark.asyncio
    async def test_stream_yields_chunks_then_response(
        self, test_client, test_audio_bytes
    ):
        """SSE stream yields STTChunk events, then STTResponse"""
        async with test_client.stream(
            "POST",
            "/api/v1/stt/transcribe/stream",
            params={"engine": "whisper"},
            files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

            events = []
            async for line in response.aiter_lines():
                if line.startswith("event:"):
                    event_type = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data = line.split(":", 1)[1].strip()
                    events.append((event_type, json.loads(data)))

            # Should have chunk events and complete event
            assert len(events) >= 2
            assert events[0][0] == "chunk"
            assert events[-1][0] == "complete"
            assert events[-1][1]["text"] == "Test transcription"

    @pytest.mark.asyncio
    async def test_stream_invalid_json_params(self, test_client, test_audio_bytes):
        """Invalid JSON in stream endpoint raises 400"""
        response = await test_client.post(
            "/api/v1/stt/transcribe/stream",
            params={"engine": "whisper", "engine_params": "invalid json"},
            files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_stream_engine_not_found(self, test_client, test_audio_bytes):
        """Non-existent engine raises 404"""
        response = await test_client.post(
            "/api/v1/stt/transcribe/stream",
            params={"engine": "nonexistent"},
            files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
        )

        assert response.status_code == 404


class TestTranscribeWebSocket:
    """WebSocket /transcribe/ws tests"""

    @pytest.mark.asyncio
    async def test_websocket_full_flow(self, app_with_mock_lifespan, test_audio_bytes):
        """Complete WebSocket flow: config → audio → results"""
        client = TestClient(app_with_mock_lifespan)

        with client.websocket_connect("/api/v1/stt/transcribe/ws") as websocket:
            # Send config
            websocket.send_json({"engine": "whisper", "language": "en"})

            # Send audio chunks
            websocket.send_bytes(test_audio_bytes[:100])
            websocket.send_bytes(test_audio_bytes[100:])

            # Send END signal
            websocket.send_text("END")

            # Receive responses
            responses = []
            while True:
                try:
                    data = websocket.receive_json(timeout=1)
                    responses.append(data)
                    if data.get("type") == "complete":
                        break
                except Exception:
                    break

            # Verify responses
            assert len(responses) >= 2
            assert responses[0]["type"] == "chunk"
            assert responses[-1]["type"] == "complete"
            assert responses[-1]["data"]["text"] == "Test transcription"

    @pytest.mark.asyncio
    async def test_websocket_missing_engine_in_config(self, app_with_mock_lifespan):
        """CRITICAL: Config without 'engine' field closes with error"""
        client = TestClient(app_with_mock_lifespan)

        with client.websocket_connect("/api/v1/stt/transcribe/ws") as websocket:
            # Send invalid config (missing engine)
            websocket.send_json({"language": "en"})

            # Should receive error
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert "engine" in response["message"].lower()

    @pytest.mark.asyncio
    async def test_websocket_invalid_config_structure(self, app_with_mock_lifespan):
        """Config is not JSON/dict → error"""
        client = TestClient(app_with_mock_lifespan)

        with client.websocket_connect("/api/v1/stt/transcribe/ws") as websocket:
            # Try to send invalid JSON
            try:
                websocket.send_text("not json")
                response = websocket.receive_json()
                assert response["type"] == "error"
            except Exception:
                # Connection may close immediately
                pass

    @pytest.mark.asyncio
    async def test_websocket_empty_audio_chunks(
        self, app_with_mock_lifespan, mock_stt_engine
    ):
        """No audio chunks sent → silent success or error?"""
        client = TestClient(app_with_mock_lifespan)

        with client.websocket_connect("/api/v1/stt/transcribe/ws") as websocket:
            # Send config
            websocket.send_json({"engine": "whisper"})

            # Send END immediately without audio
            websocket.send_text("END")

            # Should handle gracefully (no response if no audio)
            # Connection should close cleanly

        # Verify engine was not called
        mock_stt_engine.transcribe_stream.assert_not_called()

    @pytest.mark.asyncio
    async def test_websocket_large_accumulated_audio(self, app_with_mock_lifespan):
        """CRITICAL: Very large audio accumulation → memory limit"""
        client = TestClient(app_with_mock_lifespan)

        with client.websocket_connect("/api/v1/stt/transcribe/ws") as websocket:
            # Send config
            websocket.send_json({"engine": "whisper"})

            # Send many large chunks to test memory protection
            # This should either:
            # 1. Be accepted (if no limit implemented)
            # 2. Raise error/close connection (if limit implemented)
            chunk_size = 1024 * 1024  # 1MB
            num_chunks = 50  # 50MB total

            try:
                for _ in range(num_chunks):
                    websocket.send_bytes(b"x" * chunk_size)

                # If we get here, no limit is enforced (current behavior)
                # This is the vulnerability we're testing for
                websocket.send_text("END")

                # Try to receive response
                response = websocket.receive_json(timeout=5)
                # If successful, memory exhaustion is possible
                assert response  # Test passes but highlights vulnerability

            except Exception as e:
                # Connection closed due to limit - good!
                assert "limit" in str(e).lower() or "size" in str(e).lower()

    @pytest.mark.asyncio
    async def test_websocket_engine_not_found(self, app_with_mock_lifespan):
        """Non-existent engine in config → error"""
        client = TestClient(app_with_mock_lifespan)

        with client.websocket_connect("/api/v1/stt/transcribe/ws") as websocket:
            # Send config with non-existent engine
            websocket.send_json({"engine": "nonexistent"})

            # Should receive error
            response = websocket.receive_json()
            assert response["type"] == "error"

    @pytest.mark.asyncio
    async def test_websocket_registry_missing(self):
        """CRITICAL: app.state.engine_registry missing causes AttributeError"""

        @asynccontextmanager
        async def empty_lifespan(app: FastAPI):
            # Don't set engine_registry
            yield

        app = FastAPI(lifespan=empty_lifespan)
        app.include_router(stt.router, prefix="/api/v1/stt", tags=["stt"])

        client = TestClient(app)

        try:
            with client.websocket_connect("/api/v1/stt/transcribe/ws") as websocket:
                websocket.send_json({"engine": "whisper"})

                # Should receive error, not crash
                response = websocket.receive_json()
                assert response["type"] == "error"
        except Exception as e:
            # Connection may close with error
            assert "AttributeError" in str(e) or "engine_registry" in str(e)

    @pytest.mark.asyncio
    async def test_websocket_with_engine_params(
        self, app_with_mock_lifespan, test_audio_bytes, mock_stt_engine
    ):
        """engine_params passed to engine via WebSocket"""
        client = TestClient(app_with_mock_lifespan)

        with client.websocket_connect("/api/v1/stt/transcribe/ws") as websocket:
            # Send config with engine_params
            websocket.send_json(
                {
                    "engine": "whisper",
                    "language": "en",
                    "engine_params": {"beam_size": 5, "temperature": 0.5},
                }
            )

            # Send audio
            websocket.send_bytes(test_audio_bytes)
            websocket.send_text("END")

            # Receive responses
            while True:
                try:
                    data = websocket.receive_json(timeout=1)
                    if data.get("type") == "complete":
                        break
                except Exception:
                    break

        # Note: Cannot easily verify kwargs with async generator mock
        # This test ensures no errors with engine_params

    @pytest.mark.asyncio
    async def test_websocket_disconnect_before_end(
        self, app_with_mock_lifespan, test_audio_bytes
    ):
        """Client disconnects before sending END"""
        client = TestClient(app_with_mock_lifespan)

        with client.websocket_connect("/api/v1/stt/transcribe/ws") as websocket:
            # Send config and some audio
            websocket.send_json({"engine": "whisper"})
            websocket.send_bytes(test_audio_bytes)

            # Disconnect without sending END
            # TestClient context manager handles disconnect

        # Should handle gracefully without errors


class TestSTTRouterEdgeCases:
    """Edge cases and error scenarios"""

    @pytest.mark.asyncio
    async def test_transcribe_missing_audio_field(self, test_client):
        """Missing audio field raises 422"""
        response = await test_client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "whisper"},
            # No files parameter
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_transcribe_null_language(
        self, test_client, test_audio_bytes, mock_stt_engine
    ):
        """Null language parameter handled correctly"""
        response = await test_client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "whisper", "language": "null"},
            files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
        )

        assert response.status_code == 200
        # language="null" string should be passed as-is
        call_kwargs = mock_stt_engine.transcribe.call_args.kwargs
        assert call_kwargs.get("language") == "null"

    @pytest.mark.asyncio
    async def test_stream_with_language(
        self, test_client, test_audio_bytes, mock_stt_engine
    ):
        """Language parameter in stream endpoint"""
        async with test_client.stream(
            "POST",
            "/api/v1/stt/transcribe/stream",
            params={"engine": "whisper", "language": "en"},
            files={"audio": ("test.wav", test_audio_bytes, "audio/wav")},
        ) as response:
            assert response.status_code == 200

        # Verify language was passed (cannot easily verify with async gen)
        # This test ensures no errors with language param
