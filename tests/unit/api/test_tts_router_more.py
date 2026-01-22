import pytest
from fastapi.testclient import TestClient

from app.api.routers import tts as tts_router


class DummyEngine:
    async def synthesize(self, text, voice=None, speed=1.0, speaker_wav=None, **kwargs):
        from app.models.engine import TTSResponse

        return TTSResponse(
            audio_data=b"a",
            sample_rate=16000,
            duration_seconds=0.001,
            format="wav",
            performance_metrics=None,
        )

    async def synthesize_stream(
        self, text, voice=None, speed=1.0, speaker_wav=None, **kwargs
    ):
        from app.models.engine import TTSChunk, TTSResponse

        yield TTSChunk(audio_data=b"c", sequence_number=0, chunk_latency_ms=1.0)
        yield TTSResponse(
            audio_data=b"f",
            sample_rate=16000,
            duration_seconds=0.002,
            format="wav",
            performance_metrics=None,
        )


@pytest.fixture
def client():
    from app.api.main import app

    # minimal registry to avoid 503
    class _Reg:
        def get_tts(self, name):
            return DummyEngine()

    app.state.engine_registry = _Reg()
    return TestClient(app)


def test_synthesize_invalid_engine_params_returns_400(client):
    # invalid JSON for engine_params should return 400
    r = client.post(
        "/api/v1/tts/synthesize?text=hi&voice=v&engine=voxcpm&engine_params=notjson"
    )
    assert r.status_code == 400


def test_synthesize_stream_invalid_engine_params_returns_400(client):
    r = client.post(
        "/api/v1/tts/synthesize/stream?text=hi&voice=v&engine=voxcpm&engine_params=xxx"
    )
    assert r.status_code == 400


def test_synthesize_with_audio_upload(monkeypatch, client):
    # override dependency to return DummyEngine
    async def get_engine():
        return DummyEngine()

    client.app.dependency_overrides[tts_router.get_tts_engine] = get_engine

    files = {"audio": ("a.wav", b"RIFF" + b"0" * 100, "audio/wav")}
    r = client.post("/api/v1/tts/synthesize?text=hello&engine=voxcpm", files=files)
    assert r.status_code == 200


def test_websocket_missing_engine_returns_error(client):
    with client.websocket_connect("/api/v1/tts/synthesize/ws") as ws:
        ws.send_json({"text": "hi"})
        msg = ws.receive_json()
        assert msg.get("type") == "error"


def test_websocket_missing_text_returns_error(client):
    with client.websocket_connect("/api/v1/tts/synthesize/ws") as ws:
        ws.send_json({"engine": "voxcpm"})
        msg = ws.receive_json()
        assert msg.get("type") == "error"


def test_websocket_registry_error_returns_error(client):
    # Make registry raise
    class BadReg:
        def get_tts(self, name):
            raise RuntimeError("nope")

    client.app.state.engine_registry = BadReg()

    with client.websocket_connect("/api/v1/tts/synthesize/ws") as ws:
        ws.send_json({"engine": "voxcpm", "text": "hi"})
        msg = ws.receive_json()
        assert msg.get("type") == "error"


def test_websocket_streaming_exception_reports_error(client):
    # Engine whose synthesize_stream raises immediately
    class BadEngine:
        async def synthesize_stream(self, **kwargs):
            raise RuntimeError("stream fail")
            if False:
                yield

    class Reg:
        def get_tts(self, name):
            return BadEngine()

    client.app.state.engine_registry = Reg()

    with client.websocket_connect("/api/v1/tts/synthesize/ws") as ws:
        ws.send_json({"engine": "voxcpm", "text": "hi"})
        msg = ws.receive_json()
        assert msg.get("type") == "error"
