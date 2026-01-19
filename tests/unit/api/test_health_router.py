"""Unit tests for health router endpoints

Tests health check, readiness check, and engine discovery.
"""

from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.routers import health


@pytest.fixture
def app_with_health_router(mock_registry):
    """FastAPI app with health router and mocked registry"""

    @asynccontextmanager
    async def mock_lifespan(app: FastAPI):
        app.state.engine_registry = mock_registry
        yield

    app = FastAPI(lifespan=mock_lifespan)
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    return app


@pytest.fixture
async def health_client(app_with_health_router):
    """Async test client for health endpoints"""
    async with AsyncClient(
        transport=ASGITransport(app=app_with_health_router), base_url="http://test"
    ) as client:
        yield client


class TestHealthEndpoint:
    """GET /health tests"""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, health_client):
        """GET /health returns 200"""
        response = await health_client.get("/api/v1/health")

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_includes_version(self, health_client):
        """Health response includes version"""
        response = await health_client.get("/api/v1/health")
        data = response.json()

        assert "status" in data
        assert "version" in data
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"  # Default version

    @pytest.mark.asyncio
    async def test_health_always_healthy(self, health_client):
        """Health endpoint always returns healthy"""
        response = await health_client.get("/api/v1/health")
        data = response.json()

        assert data["status"] == "healthy"


class TestReadinessEndpoint:
    """GET /ready tests"""

    @pytest.mark.asyncio
    async def test_ready_all_engines_ready(self, health_client, mock_stt_engine):
        """All engines ready → ready=True"""
        mock_stt_engine.is_ready.return_value = True

        response = await health_client.get("/api/v1/ready")
        data = response.json()

        assert response.status_code == 200
        assert data["ready"] is True
        assert "whisper" in data["stt_engines"]
        assert data["stt_engines"]["whisper"] is True

    @pytest.mark.asyncio
    async def test_ready_one_engine_not_ready(self, mock_stt_engine):
        """One engine not ready → ready=False"""
        # Set engine as not ready
        mock_stt_engine.is_ready.return_value = False

        # Create registry with not-ready engine
        from app.api.registry import EngineRegistry

        registry = EngineRegistry()
        registry._stt_engines = {"whisper": mock_stt_engine}

        @asynccontextmanager
        async def mock_lifespan(app: FastAPI):
            app.state.engine_registry = registry
            yield

        app = FastAPI(lifespan=mock_lifespan)
        app.include_router(health.router, prefix="/api/v1", tags=["health"])

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/ready")
            data = response.json()

            assert response.status_code == 200
            assert data["ready"] is False
            assert data["stt_engines"]["whisper"] is False

    @pytest.mark.asyncio
    async def test_ready_empty_registry(self):
        """CRITICAL: Empty registry → ready=True (bug!)"""
        # This is Critical Issue #5 from the plan
        from app.api.registry import EngineRegistry

        registry = EngineRegistry()
        # No engines registered

        @asynccontextmanager
        async def mock_lifespan(app: FastAPI):
            app.state.engine_registry = registry
            yield

        app = FastAPI(lifespan=mock_lifespan)
        app.include_router(health.router, prefix="/api/v1", tags=["health"])

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/ready")
            data = response.json()

            assert response.status_code == 200
            # BUG: all([]) and all([]) both return True
            assert data["ready"] is True
            assert data["stt_engines"] == {}
            assert data["tts_engines"] == {}

    @pytest.mark.asyncio
    async def test_ready_includes_engine_status(self, mock_stt_engine, mock_tts_engine):
        """Response includes individual engine statuses"""
        from app.api.registry import EngineRegistry

        registry = EngineRegistry()
        registry._stt_engines = {"whisper": mock_stt_engine}
        registry._tts_engines = {"coqui": mock_tts_engine}

        mock_stt_engine.is_ready.return_value = True
        mock_tts_engine.is_ready.return_value = True

        @asynccontextmanager
        async def mock_lifespan(app: FastAPI):
            app.state.engine_registry = registry
            yield

        app = FastAPI(lifespan=mock_lifespan)
        app.include_router(health.router, prefix="/api/v1", tags=["health"])

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/ready")
            data = response.json()

            assert data["ready"] is True
            assert "whisper" in data["stt_engines"]
            assert "coqui" in data["tts_engines"]
            assert data["stt_engines"]["whisper"] is True
            assert data["tts_engines"]["coqui"] is True

    @pytest.mark.asyncio
    async def test_ready_mixed_status(self, mock_stt_engine, mock_tts_engine):
        """Some engines ready, some not → ready=False"""
        from app.api.registry import EngineRegistry

        registry = EngineRegistry()
        registry._stt_engines = {"whisper": mock_stt_engine}
        registry._tts_engines = {"coqui": mock_tts_engine}

        mock_stt_engine.is_ready.return_value = True
        mock_tts_engine.is_ready.return_value = False  # TTS not ready

        @asynccontextmanager
        async def mock_lifespan(app: FastAPI):
            app.state.engine_registry = registry
            yield

        app = FastAPI(lifespan=mock_lifespan)
        app.include_router(health.router, prefix="/api/v1", tags=["health"])

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/ready")
            data = response.json()

            assert data["ready"] is False
            assert data["stt_engines"]["whisper"] is True
            assert data["tts_engines"]["coqui"] is False


class TestEnginesEndpoint:
    """GET /engines tests"""

    @pytest.mark.asyncio
    async def test_list_engines(self, health_client):
        """List all registered engines with metadata"""
        response = await health_client.get("/api/v1/engines")
        data = response.json()

        assert response.status_code == 200
        assert "engines" in data
        assert len(data["engines"]) >= 1

        # Check first engine structure
        engine = data["engines"][0]
        assert "name" in engine
        assert "type" in engine
        assert "ready" in engine
        assert "engine_name" in engine

    @pytest.mark.asyncio
    async def test_engines_empty_registry(self):
        """Empty registry returns empty list"""
        from app.api.registry import EngineRegistry

        registry = EngineRegistry()

        @asynccontextmanager
        async def mock_lifespan(app: FastAPI):
            app.state.engine_registry = registry
            yield

        app = FastAPI(lifespan=mock_lifespan)
        app.include_router(health.router, prefix="/api/v1", tags=["health"])

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/engines")
            data = response.json()

            assert response.status_code == 200
            assert data["engines"] == []

    @pytest.mark.asyncio
    async def test_engine_metadata_complete(self, health_client):
        """Engine info includes name, type, ready, formats/voices"""
        response = await health_client.get("/api/v1/engines")
        data = response.json()

        engine = data["engines"][0]
        assert engine["name"] == "whisper"
        assert engine["type"] == "stt"
        assert engine["ready"] is True
        assert engine["engine_name"] == "mock-stt"
        assert "supported_formats" in engine
        assert engine["supported_formats"] == ["wav", "mp3"]

    @pytest.mark.asyncio
    async def test_engines_includes_both_stt_and_tts(
        self, mock_stt_engine, mock_tts_engine
    ):
        """Engines list includes both STT and TTS"""
        from app.api.registry import EngineRegistry

        registry = EngineRegistry()
        registry._stt_engines = {"whisper": mock_stt_engine}
        registry._tts_engines = {"coqui": mock_tts_engine}

        @asynccontextmanager
        async def mock_lifespan(app: FastAPI):
            app.state.engine_registry = registry
            yield

        app = FastAPI(lifespan=mock_lifespan)
        app.include_router(health.router, prefix="/api/v1", tags=["health"])

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/engines")
            data = response.json()

            assert len(data["engines"]) == 2

            # Check STT engine
            stt_engine = next(e for e in data["engines"] if e["type"] == "stt")
            assert stt_engine["name"] == "whisper"
            assert "supported_formats" in stt_engine
            assert stt_engine["supported_voices"] is None

            # Check TTS engine
            tts_engine = next(e for e in data["engines"] if e["type"] == "tts")
            assert tts_engine["name"] == "coqui"
            assert "supported_voices" in tts_engine
            assert tts_engine["supported_formats"] is None

    @pytest.mark.asyncio
    async def test_engines_not_ready_included(self, mock_stt_engine):
        """Not-ready engines still appear in list"""
        mock_stt_engine.is_ready.return_value = False

        from app.api.registry import EngineRegistry

        registry = EngineRegistry()
        registry._stt_engines = {"whisper": mock_stt_engine}

        @asynccontextmanager
        async def mock_lifespan(app: FastAPI):
            app.state.engine_registry = registry
            yield

        app = FastAPI(lifespan=mock_lifespan)
        app.include_router(health.router, prefix="/api/v1", tags=["health"])

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/engines")
            data = response.json()

            assert len(data["engines"]) == 1
            assert data["engines"][0]["ready"] is False


class TestMetricsEndpoint:
    """GET /metrics tests"""

    @pytest.mark.asyncio
    async def test_metrics_placeholder(self, health_client):
        """Metrics endpoint returns placeholder"""
        response = await health_client.get("/api/v1/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "placeholder" in data["message"].lower()


class TestHealthRouterEdgeCases:
    """Edge cases for health router"""

    @pytest.mark.asyncio
    async def test_health_no_registry(self):
        """Health check works even without registry"""

        @asynccontextmanager
        async def empty_lifespan(app: FastAPI):
            # Don't set registry
            yield

        app = FastAPI(lifespan=empty_lifespan)
        app.include_router(health.router, prefix="/api/v1", tags=["health"])

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Health should work (doesn't need registry)
            response = await client.get("/api/v1/health")
            assert response.status_code == 200

            # Ready should fail (needs registry)
            response = await client.get("/api/v1/ready")
            assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_ready_multiple_engines_same_type(self):
        """Multiple engines of same type"""
        engine1 = MagicMock()
        engine1.is_ready.return_value = True
        engine1.engine_name = "engine1"

        engine2 = MagicMock()
        engine2.is_ready.return_value = True
        engine2.engine_name = "engine2"

        from app.api.registry import EngineRegistry

        registry = EngineRegistry()
        registry._stt_engines = {"whisper": engine1, "google": engine2}

        @asynccontextmanager
        async def mock_lifespan(app: FastAPI):
            app.state.engine_registry = registry
            yield

        app = FastAPI(lifespan=mock_lifespan)
        app.include_router(health.router, prefix="/api/v1", tags=["health"])

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/ready")
            data = response.json()

            assert data["ready"] is True
            assert len(data["stt_engines"]) == 2
            assert data["stt_engines"]["whisper"] is True
            assert data["stt_engines"]["google"] is True
