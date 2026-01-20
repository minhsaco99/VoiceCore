"""Unit tests for health router endpoints"""


class TestHealthEndpoint:
    """GET /health tests"""

    def test_returns_200(self, client):
        """Returns 200 OK"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_returns_status_healthy(self, client):
        """Returns status healthy"""
        response = client.get("/api/v1/health")
        assert response.json()["status"] == "healthy"

    def test_returns_version(self, client):
        """Returns version"""
        response = client.get("/api/v1/health")
        assert "version" in response.json()


class TestReadinessEndpoint:
    """GET /ready tests"""

    def test_returns_200(self, client):
        """Returns 200 OK"""
        response = client.get("/api/v1/ready")
        assert response.status_code == 200

    def test_ready_true_when_engine_ready(self, client):
        """ready=True when engine is ready"""
        response = client.get("/api/v1/ready")
        data = response.json()

        assert data["ready"] is True
        assert "default" in data["stt_engines"]
        assert data["stt_engines"]["default"] is True

    def test_ready_false_when_engine_not_ready(self, app_with_stt, mock_stt_engine):
        """ready=False when engine not ready"""
        from fastapi.testclient import TestClient

        mock_stt_engine.is_ready.return_value = False
        with TestClient(app_with_stt) as client:
            response = client.get("/api/v1/ready")
            data = response.json()

            assert data["ready"] is False
            assert data["stt_engines"]["default"] is False

    def test_ready_true_when_empty_registry(self, client_empty):
        """ready=True when empty registry (all([]) is True)"""
        response = client_empty.get("/api/v1/ready")
        data = response.json()

        # Note: This is a potential bug - empty registry returns ready=True
        assert data["ready"] is True
        assert data["stt_engines"] == {}
        assert data["tts_engines"] == {}

    def test_includes_both_engine_types(self, client_both):
        """Includes both STT and TTS status"""
        response = client_both.get("/api/v1/ready")
        data = response.json()

        assert "default" in data["stt_engines"]
        assert "default" in data["tts_engines"]

    def test_returns_503_when_no_registry(self, client_no_registry):
        """Returns 503 when registry not available"""
        response = client_no_registry.get("/api/v1/ready")
        assert response.status_code == 503


class TestEnginesEndpoint:
    """GET /engines tests"""

    def test_returns_200(self, client):
        """Returns 200 OK"""
        response = client.get("/api/v1/engines")
        assert response.status_code == 200

    def test_lists_stt_engine(self, client):
        """Lists STT engine"""
        response = client.get("/api/v1/engines")
        data = response.json()

        assert len(data["engines"]) == 1
        engine = data["engines"][0]
        assert engine["name"] == "default"
        assert engine["type"] == "stt"
        assert engine["ready"] is True

    def test_lists_both_engines(self, client_both):
        """Lists both STT and TTS engines"""
        response = client_both.get("/api/v1/engines")
        data = response.json()

        assert len(data["engines"]) == 2
        types = [e["type"] for e in data["engines"]]
        assert "stt" in types
        assert "tts" in types

    def test_empty_list_when_no_engines(self, client_empty):
        """Returns empty list when no engines"""
        response = client_empty.get("/api/v1/engines")
        data = response.json()

        assert data["engines"] == []

    def test_includes_engine_metadata(self, client):
        """Includes engine metadata"""
        response = client.get("/api/v1/engines")
        engine = response.json()["engines"][0]

        assert "name" in engine
        assert "type" in engine
        assert "ready" in engine
        assert "engine_name" in engine
        assert "supported_formats" in engine

    def test_returns_503_when_no_registry(self, client_no_registry):
        """Returns 503 when registry not available"""
        response = client_no_registry.get("/api/v1/engines")
        assert response.status_code == 503


class TestMetricsEndpoint:
    """GET /metrics tests"""

    def test_returns_200(self, client):
        """Returns 200 OK"""
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200

    def test_returns_placeholder(self, client):
        """Returns placeholder message"""
        response = client.get("/api/v1/metrics")
        assert "message" in response.json()
