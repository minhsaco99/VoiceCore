"""Unit tests for TTS router endpoints (stubs)"""


class TestTTSRouter:
    """TTS router stub endpoint tests"""

    def test_synthesize_returns_501(self, client_both):
        """POST /synthesize returns 501"""
        response = client_both.post(
            "/api/v1/tts/synthesize",
            params={"text": "Hello", "engine": "default"},
        )

        assert response.status_code == 501

    def test_synthesize_stream_returns_501(self, client_both):
        """POST /synthesize/stream returns 501"""
        response = client_both.post(
            "/api/v1/tts/synthesize/stream",
            params={"text": "Hello", "engine": "default"},
        )

        assert response.status_code == 501

    def test_endpoints_exist(self, client_both):
        """TTS endpoints exist (not 404)"""
        r1 = client_both.post(
            "/api/v1/tts/synthesize", params={"text": "x", "engine": "default"}
        )
        r2 = client_both.post(
            "/api/v1/tts/synthesize/stream", params={"text": "x", "engine": "default"}
        )

        assert r1.status_code != 404
        assert r2.status_code != 404

    def test_synthesize_with_minimal_request(self, client_both):
        """Synthesize with minimal request"""
        response = client_both.post(
            "/api/v1/tts/synthesize",
            params={"text": "Test", "engine": "default"},
        )

        assert response.status_code == 501

    def test_synthesize_missing_fields_returns_422(self, client_both):
        """Missing required fields returns 422"""
        response = client_both.post(
            "/api/v1/tts/synthesize",
            params={},
        )

        assert response.status_code == 422
