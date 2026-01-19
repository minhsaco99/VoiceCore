"""Unit tests for TTS router endpoints (stubs)

Current TTS endpoints are placeholders returning 501 Not Implemented.
"""

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.routers import tts


@pytest.fixture
def app_with_tts_router():
    """FastAPI app with TTS router"""
    app = FastAPI()
    app.include_router(tts.router, prefix="/api/v1/tts", tags=["tts"])
    return app


@pytest.fixture
async def tts_client(app_with_tts_router):
    """Async test client for TTS endpoints"""
    async with AsyncClient(
        transport=ASGITransport(app=app_with_tts_router), base_url="http://test"
    ) as client:
        yield client


class TestTTSRouter:
    """TTS router stub endpoint tests"""

    @pytest.mark.asyncio
    async def test_synthesize_returns_501(self, tts_client):
        """POST /synthesize returns 501 Not Implemented"""
        response = await tts_client.post(
            "/api/v1/tts/synthesize",
            json={"text": "Hello world", "engine": "coqui", "voice": "default"},
        )

        assert response.status_code == 501
        assert "not implemented" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_synthesize_stream_returns_501(self, tts_client):
        """POST /synthesize/stream returns 501 Not Implemented"""
        response = await tts_client.post(
            "/api/v1/tts/synthesize/stream",
            json={"text": "Hello world", "engine": "coqui", "voice": "default"},
        )

        assert response.status_code == 501
        assert "not implemented" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_endpoints_exist(self, tts_client):
        """TTS endpoints are registered and reachable"""
        # Both endpoints should exist and return errors, not 404
        response1 = await tts_client.post(
            "/api/v1/tts/synthesize",
            json={"text": "Test", "engine": "test", "voice": "test"},
        )
        assert response1.status_code != 404

        response2 = await tts_client.post(
            "/api/v1/tts/synthesize/stream",
            json={"text": "Test", "engine": "test", "voice": "test"},
        )
        assert response2.status_code != 404

    @pytest.mark.asyncio
    async def test_synthesize_with_minimal_request(self, tts_client):
        """Synthesize with minimal TTSRequest"""
        response = await tts_client.post(
            "/api/v1/tts/synthesize",
            json={"text": "Test", "engine": "test"},
        )

        # Should still return 501
        assert response.status_code == 501

    @pytest.mark.asyncio
    async def test_synthesize_with_invalid_request(self, tts_client):
        """Invalid request structure raises 422"""
        response = await tts_client.post(
            "/api/v1/tts/synthesize",
            json={},  # Missing required fields
        )

        # Should be validation error, not 501
        assert response.status_code == 422


class TestTTSFutureImplementation:
    """Tests to guide future TTS implementation"""

    @pytest.mark.skip(reason="TTS not implemented yet")
    @pytest.mark.asyncio
    async def test_synthesize_success(self, tts_client):
        """Future test: Valid request returns audio"""
        response = await tts_client.post(
            "/api/v1/tts/synthesize",
            json={"text": "Hello world", "engine": "coqui", "voice": "default"},
        )

        assert response.status_code == 200
        assert "audio_data" in response.json()

    @pytest.mark.skip(reason="TTS not implemented yet")
    @pytest.mark.asyncio
    async def test_synthesize_stream_success(self, tts_client):
        """Future test: Streaming TTS returns audio chunks"""
        response = await tts_client.post(
            "/api/v1/tts/synthesize/stream",
            json={"text": "Hello world", "engine": "coqui", "voice": "default"},
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

    @pytest.mark.skip(reason="TTS not implemented yet")
    @pytest.mark.asyncio
    async def test_synthesize_engine_not_found(self, tts_client):
        """Future test: Non-existent engine raises 404"""
        response = await tts_client.post(
            "/api/v1/tts/synthesize",
            json={"text": "Test", "engine": "nonexistent", "voice": "default"},
        )

        assert response.status_code == 404

    @pytest.mark.skip(reason="TTS not implemented yet")
    @pytest.mark.asyncio
    async def test_synthesize_voice_not_supported(self, tts_client):
        """Future test: Unsupported voice raises error"""
        response = await tts_client.post(
            "/api/v1/tts/synthesize",
            json={"text": "Test", "engine": "coqui", "voice": "nonexistent_voice"},
        )

        assert response.status_code in [400, 404]

    @pytest.mark.skip(reason="TTS not implemented yet")
    @pytest.mark.asyncio
    async def test_synthesize_empty_text(self, tts_client):
        """Future test: Empty text raises validation error"""
        response = await tts_client.post(
            "/api/v1/tts/synthesize",
            json={"text": "", "engine": "coqui", "voice": "default"},
        )

        assert response.status_code == 400

    @pytest.mark.skip(reason="TTS not implemented yet")
    @pytest.mark.asyncio
    async def test_synthesize_with_speed_param(self, tts_client):
        """Future test: Speed parameter is respected"""
        response = await tts_client.post(
            "/api/v1/tts/synthesize",
            json={
                "text": "Test",
                "engine": "coqui",
                "voice": "default",
                "speed": 1.5,
            },
        )

        assert response.status_code == 200
