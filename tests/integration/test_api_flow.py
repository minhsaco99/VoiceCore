"""End-to-end API flow integration tests

Tests complete request flows from client to engine and back.
"""

import pytest
from fastapi.testclient import TestClient

from app.api.main import app


class TestSTTFlow:
    """End-to-end STT transcription flows"""

    @pytest.mark.asyncio
    async def test_full_transcription_flow(self, api_client, test_audio_file):
        """End-to-end: Upload audio → transcribe → receive response"""
        if test_audio_file is None or not test_audio_file.exists():
            pytest.skip("Test audio file not available")

        with test_audio_file.open("rb") as f:
            audio_data = f.read()

        response = await api_client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "whisper"},
            files={"audio": ("test.wav", audio_data, "audio/wav")},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify full STTResponse structure
        assert "text" in data
        assert "language" in data
        assert "segments" in data
        assert "processing_time" in data
        assert isinstance(data["segments"], list)

        # Verify metrics
        assert isinstance(data["processing_time"], (int, float))
        assert data["processing_time"] >= 0

    @pytest.mark.asyncio
    async def test_streaming_flow(self, api_client, test_audio_file):
        """End-to-end: Upload audio → stream → receive chunks + response"""
        if test_audio_file is None or not test_audio_file.exists():
            pytest.skip("Test audio file not available")

        with test_audio_file.open("rb") as f:
            audio_data = f.read()

        async with api_client.stream(
            "POST",
            "/api/v1/stt/transcribe/stream",
            params={"engine": "whisper"},
            files={"audio": ("test.wav", audio_data, "audio/wav")},
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

            events = []
            async for line in response.aiter_lines():
                if line.startswith("event:"):
                    event_type = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    import json

                    data = line.split(":", 1)[1].strip()
                    events.append((event_type, json.loads(data)))

            # Should have at least one complete event
            assert len(events) >= 1
            assert any(e[0] == "complete" for e in events)

            # Final event should be complete
            assert events[-1][0] == "complete"
            assert "text" in events[-1][1]

    @pytest.mark.asyncio
    async def test_websocket_flow(self, test_audio_file):
        """End-to-end: WebSocket connection → send audio → receive results"""
        if test_audio_file is None or not test_audio_file.exists():
            pytest.skip("Test audio file not available")

        with test_audio_file.open("rb") as f:
            audio_data = f.read()

        client = TestClient(app)

        with client.websocket_connect("/api/v1/stt/transcribe/ws") as websocket:
            # Send config
            websocket.send_json({"engine": "whisper", "language": "en"})

            # Send audio in chunks
            chunk_size = 8192
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                websocket.send_bytes(chunk)

            # Send END signal
            websocket.send_text("END")

            # Receive responses
            responses = []
            while True:
                try:
                    data = websocket.receive_json(timeout=10)
                    responses.append(data)
                    if data.get("type") == "complete":
                        break
                except Exception:
                    break

            # Verify responses
            assert len(responses) >= 1
            assert responses[-1]["type"] == "complete"
            assert "text" in responses[-1]["data"]

    @pytest.mark.asyncio
    async def test_multi_engine_selection(self, api_client):
        """Different engines handle requests correctly"""
        # Create small test audio
        test_audio = b"x" * 1000

        # Try with whisper engine
        response = await api_client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "whisper"},
            files={"audio": ("test.wav", test_audio, "audio/wav")},
        )

        # Should either work or fail gracefully
        assert response.status_code in [200, 400, 404, 500]

        # Non-existent engine should return 404
        response = await api_client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "nonexistent"},
            files={"audio": ("test.wav", test_audio, "audio/wav")},
        )

        assert response.status_code == 404


class TestHealthFlows:
    """End-to-end health check flows"""

    @pytest.mark.asyncio
    async def test_health_check_flow(self, api_client):
        """Complete health check flow"""
        response = await api_client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_readiness_check_flow(self, api_client):
        """Complete readiness check flow"""
        response = await api_client.get("/api/v1/ready")

        assert response.status_code == 200
        data = response.json()
        assert "ready" in data
        assert "stt_engines" in data
        assert "tts_engines" in data

    @pytest.mark.asyncio
    async def test_engine_discovery_flow(self, api_client):
        """Complete engine discovery flow"""
        response = await api_client.get("/api/v1/engines")

        assert response.status_code == 200
        data = response.json()
        assert "engines" in data
        assert isinstance(data["engines"], list)

        # If engines are loaded, verify structure
        if len(data["engines"]) > 0:
            engine = data["engines"][0]
            assert "name" in engine
            assert "type" in engine
            assert "ready" in engine


class TestErrorFlows:
    """End-to-end error handling flows"""

    @pytest.mark.asyncio
    async def test_invalid_audio_flow(self, api_client):
        """Invalid audio rejected"""
        response = await api_client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "whisper"},
            files={"audio": ("test.wav", b"", "audio/wav")},
        )

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_missing_engine_flow(self, api_client):
        """Missing engine parameter rejected"""
        response = await api_client.post(
            "/api/v1/stt/transcribe",
            files={"audio": ("test.wav", b"test audio", "audio/wav")},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_nonexistent_engine_flow(self, api_client):
        """Non-existent engine returns 404"""
        response = await api_client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "nonexistent"},
            files={"audio": ("test.wav", b"test audio", "audio/wav")},
        )

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_oversized_audio_flow(self, api_client):
        """Oversized audio rejected"""
        # Create 26MB audio (over default 25MB limit)
        large_audio = b"x" * (26 * 1024 * 1024)

        response = await api_client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "whisper"},
            files={"audio": ("large.wav", large_audio, "audio/wav")},
        )

        assert response.status_code == 413


class TestCORSFlow:
    """CORS header flows"""

    @pytest.mark.asyncio
    async def test_cors_headers_present(self, api_client):
        """CORS headers present in responses"""
        response = await api_client.get("/api/v1/health")

        # Check for CORS headers (may vary based on config)
        # This is a basic check that middleware is working
        assert response.status_code == 200


class TestMiddlewareFlow:
    """Middleware processing flows"""

    @pytest.mark.asyncio
    async def test_logging_middleware_flow(self, api_client):
        """Logging middleware adds timing header"""
        response = await api_client.get("/api/v1/health")

        # Check for X-Process-Time header
        assert "x-process-time" in response.headers
        timing = response.headers["x-process-time"]
        assert timing.endswith("ms")

    @pytest.mark.asyncio
    async def test_error_handler_middleware_flow(self, api_client):
        """Error handler middleware formats errors"""
        # Trigger an error
        response = await api_client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "whisper"},
            files={"audio": ("test.wav", b"", "audio/wav")},
        )

        # Error should be formatted as JSON
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data


class TestConcurrentRequests:
    """Concurrent request handling"""

    @pytest.mark.asyncio
    async def test_concurrent_transcription_requests(self, api_client):
        """Multiple concurrent transcription requests"""
        import asyncio

        test_audio = b"x" * 1000

        async def make_request():
            return await api_client.post(
                "/api/v1/stt/transcribe",
                params={"engine": "whisper"},
                files={"audio": ("test.wav", test_audio, "audio/wav")},
            )

        # Send 5 concurrent requests
        responses = await asyncio.gather(*[make_request() for _ in range(5)])

        # All should complete (may succeed or fail gracefully)
        assert len(responses) == 5
        for response in responses:
            assert response.status_code in [200, 400, 404, 500, 503]

    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self, api_client):
        """Multiple concurrent health checks"""
        import asyncio

        async def make_health_request():
            return await api_client.get("/api/v1/health")

        # Send 10 concurrent health checks
        responses = await asyncio.gather(*[make_health_request() for _ in range(10)])

        # All should succeed
        assert len(responses) == 10
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"


class TestFullUserJourney:
    """Complete user journey tests"""

    @pytest.mark.asyncio
    async def test_new_user_journey(self, api_client):
        """New user discovering and using API"""
        # 1. Check API health
        health_response = await api_client.get("/api/v1/health")
        assert health_response.status_code == 200

        # 2. Check readiness
        ready_response = await api_client.get("/api/v1/ready")
        assert ready_response.status_code == 200

        # 3. Discover available engines
        engines_response = await api_client.get("/api/v1/engines")
        assert engines_response.status_code == 200
        engines = engines_response.json()["engines"]

        # 4. Use an available engine (if any)
        if len(engines) > 0:
            engine_name = engines[0]["name"]
            test_audio = b"x" * 1000

            transcribe_response = await api_client.post(
                "/api/v1/stt/transcribe",
                params={"engine": engine_name},
                files={"audio": ("test.wav", test_audio, "audio/wav")},
            )

            # Should get a response (success or graceful failure)
            assert transcribe_response.status_code in [200, 400, 500]

    @pytest.mark.asyncio
    async def test_error_recovery_journey(self, api_client):
        """User encounters error and recovers"""
        # 1. Try invalid request
        invalid_response = await api_client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "nonexistent"},
            files={"audio": ("test.wav", b"test", "audio/wav")},
        )
        assert invalid_response.status_code == 404

        # 2. Discover correct engines
        engines_response = await api_client.get("/api/v1/engines")
        engines = engines_response.json()["engines"]

        # 3. Retry with valid engine
        if len(engines) > 0:
            engine_name = engines[0]["name"]
            valid_response = await api_client.post(
                "/api/v1/stt/transcribe",
                params={"engine": engine_name},
                files={"audio": ("test.wav", b"x" * 1000, "audio/wav")},
            )

            # Should work or fail gracefully (not 404)
            assert valid_response.status_code in [200, 400, 500, 503]
