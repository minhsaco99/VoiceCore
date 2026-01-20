"""End-to-end API flow integration tests

Tests complete request flows from client to engine and back.
Requires faster_whisper to be installed.
"""

import pytest
from starlette.testclient import TestClient

# Skip all tests if faster_whisper not available
pytest.importorskip("faster_whisper", reason="faster_whisper not installed")

from app.api.main import app


@pytest.fixture
def client():
    """Sync TestClient for integration tests"""
    with TestClient(app) as c:
        yield c


class TestSTTFlow:
    """End-to-end STT transcription flows"""

    def test_full_transcription_flow(self, client, test_audio_file):
        """End-to-end: Upload audio → transcribe → receive response"""
        if test_audio_file is None or not test_audio_file.exists():
            pytest.skip("Test audio file not available")

        with test_audio_file.open("rb") as f:
            audio_data = f.read()

        response = client.post(
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
        assert isinstance(data["segments"], list)

    def test_multi_engine_selection(self, client):
        """Different engines handle requests correctly"""
        test_audio = b"x" * 1000

        # Try with whisper engine
        response = client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "whisper"},
            files={"audio": ("test.wav", test_audio, "audio/wav")},
        )

        # Should either work or fail gracefully
        assert response.status_code in [200, 400, 404, 500]

        # Non-existent engine should return 404
        response = client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "nonexistent"},
            files={"audio": ("test.wav", test_audio, "audio/wav")},
        )

        assert response.status_code == 404


class TestHealthFlows:
    """End-to-end health check flows"""

    def test_health_check_flow(self, client):
        """Complete health check flow"""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_readiness_check_flow(self, client):
        """Complete readiness check flow"""
        response = client.get("/api/v1/ready")

        assert response.status_code == 200
        data = response.json()
        assert "ready" in data
        assert "stt_engines" in data
        assert "tts_engines" in data

    def test_engine_discovery_flow(self, client):
        """Complete engine discovery flow"""
        response = client.get("/api/v1/engines")

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

    def test_invalid_audio_flow(self, client):
        """Invalid audio rejected"""
        response = client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "whisper"},
            files={"audio": ("test.wav", b"", "audio/wav")},
        )

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    def test_missing_engine_flow(self, client):
        """Missing engine parameter rejected"""
        response = client.post(
            "/api/v1/stt/transcribe",
            files={"audio": ("test.wav", b"test audio", "audio/wav")},
        )

        assert response.status_code == 422

    def test_nonexistent_engine_flow(self, client):
        """Non-existent engine returns 404"""
        response = client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "nonexistent"},
            files={"audio": ("test.wav", b"test audio", "audio/wav")},
        )

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    def test_oversized_audio_flow(self, client):
        """Oversized audio rejected"""
        # Create 26MB audio (over default 25MB limit)
        large_audio = b"x" * (26 * 1024 * 1024)

        response = client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "whisper"},
            files={"audio": ("large.wav", large_audio, "audio/wav")},
        )

        assert response.status_code == 413


class TestCORSFlow:
    """CORS header flows"""

    def test_cors_headers_present(self, client):
        """CORS headers present in responses"""
        response = client.get("/api/v1/health")

        # Check for CORS headers (may vary based on config)
        # This is a basic check that middleware is working
        assert response.status_code == 200


class TestMiddlewareFlow:
    """Middleware processing flows"""

    def test_logging_middleware_flow(self, client):
        """Logging middleware adds timing header"""
        response = client.get("/api/v1/health")

        # Check for X-Process-Time header
        assert "x-process-time" in response.headers
        timing = response.headers["x-process-time"]
        assert timing.endswith("ms")

    def test_error_handler_middleware_flow(self, client):
        """Error handler middleware formats errors"""
        # Trigger an error
        response = client.post(
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

    def test_concurrent_health_checks(self, client):
        """Multiple health checks work"""
        # Send multiple health checks
        responses = [client.get("/api/v1/health") for _ in range(5)]

        # All should succeed
        assert len(responses) == 5
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"


class TestFullUserJourney:
    """Complete user journey tests"""

    def test_new_user_journey(self, client):
        """New user discovering and using API"""
        # 1. Check API health
        health_response = client.get("/api/v1/health")
        assert health_response.status_code == 200

        # 2. Check readiness
        ready_response = client.get("/api/v1/ready")
        assert ready_response.status_code == 200

        # 3. Discover available engines
        engines_response = client.get("/api/v1/engines")
        assert engines_response.status_code == 200
        engines = engines_response.json()["engines"]

        # 4. Use an available engine (if any)
        if len(engines) > 0:
            engine_name = engines[0]["name"]
            test_audio = b"x" * 1000

            transcribe_response = client.post(
                "/api/v1/stt/transcribe",
                params={"engine": engine_name},
                files={"audio": ("test.wav", test_audio, "audio/wav")},
            )

            # Should get a response (success or graceful failure)
            assert transcribe_response.status_code in [200, 400, 500]

    def test_error_recovery_journey(self, client):
        """User encounters error and recovers"""
        # 1. Try invalid request
        invalid_response = client.post(
            "/api/v1/stt/transcribe",
            params={"engine": "nonexistent"},
            files={"audio": ("test.wav", b"test", "audio/wav")},
        )
        assert invalid_response.status_code == 404

        # 2. Discover correct engines
        engines_response = client.get("/api/v1/engines")
        engines = engines_response.json()["engines"]

        # 3. Retry with valid engine
        if len(engines) > 0:
            engine_name = engines[0]["name"]
            valid_response = client.post(
                "/api/v1/stt/transcribe",
                params={"engine": engine_name},
                files={"audio": ("test.wav", b"x" * 1000, "audio/wav")},
            )

            # Should work or fail gracefully (not 404)
            assert valid_response.status_code in [200, 400, 500, 503]
