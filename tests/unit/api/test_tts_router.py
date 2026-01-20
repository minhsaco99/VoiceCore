"""Unit tests for TTS router endpoints (stubs)"""


class TestTTSRouter:
    """TTS router stub endpoint tests"""

    def test_synthesize_returns_501(self, client):
        """POST /synthesize returns 501"""
        response = client.post(
            "/api/v1/tts/synthesize",
            json={"text": "Hello", "engine": "default"},
        )

        assert response.status_code == 501

    def test_synthesize_stream_returns_501(self, client):
        """POST /synthesize/stream returns 501"""
        response = client.post(
            "/api/v1/tts/synthesize/stream",
            json={"text": "Hello", "engine": "default"},
        )

        assert response.status_code == 501

    def test_endpoints_exist(self, client):
        """TTS endpoints exist (not 404)"""
        r1 = client.post("/api/v1/tts/synthesize", json={"text": "x", "engine": "x"})
        r2 = client.post(
            "/api/v1/tts/synthesize/stream", json={"text": "x", "engine": "x"}
        )

        assert r1.status_code != 404
        assert r2.status_code != 404

    def test_synthesize_with_minimal_request(self, client):
        """Synthesize with minimal request"""
        response = client.post(
            "/api/v1/tts/synthesize",
            json={"text": "Test", "engine": "test"},
        )

        assert response.status_code == 501

    def test_synthesize_missing_fields_returns_422(self, client):
        """Missing required fields returns 422"""
        response = client.post(
            "/api/v1/tts/synthesize",
            json={},
        )

        assert response.status_code == 422
