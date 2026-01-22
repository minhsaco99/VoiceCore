import pytest
from fastapi.testclient import TestClient

from app.api.routers import tts as tts_router
from app.models.engine import TTSChunk, TTSResponse


class DummyEngine:
    def __init__(self):
        self.called = False

    async def synthesize(self, text, voice=None, speed=1.0, speaker_wav=None, **kwargs):
        self.called = True
        return TTSResponse(
            audio_data=b"\x00\x01",
            sample_rate=16000,
            duration_seconds=0.001,
            format="wav",
            performance_metrics=None,
        )

    async def synthesize_stream(
        self, text, voice=None, speed=1.0, speaker_wav=None, **kwargs
    ):
        # yield two chunks then final response
        for i in range(2):
            yield TTSChunk(
                audio_data=b"chunk%d" % i, sequence_number=i, chunk_latency_ms=1.0
            )
        yield TTSResponse(
            audio_data=b"final",
            sample_rate=16000,
            duration_seconds=0.002,
            format="wav",
            performance_metrics=None,
        )


@pytest.fixture
def client():
    from app.api.main import app

    # Ensure an engine registry exists on app.state to avoid 503 from dependency
    class _Reg:
        def get_tts(self, name):
            # simple placeholder engine that will error if used unexpectedly
            class _E:
                async def synthesize(self, *a, **k):
                    return None

                async def synthesize_stream(self, *a, **k):
                    if False:
                        yield

            return _E()

    app.state.engine_registry = _Reg()
    return TestClient(app)


def test_synthesize_requires_voice_or_audio(monkeypatch, client):
    # No audio and no voice -> 501
    r = client.post("/api/v1/tts/synthesize?text=hello&engine=voxcpm")
    assert r.status_code == 501


def test_synthesize_with_voice(monkeypatch, client):
    # Inject dummy engine via dependency override
    dummy = DummyEngine()

    async def _get_engine():
        return dummy

    app = client.app
    app.dependency_overrides[tts_router.get_tts_engine] = _get_engine

    r = client.post("/api/v1/tts/synthesize?text=hello&voice=any&engine=voxcpm")
    assert r.status_code == 200
    data = r.json()
    assert "audio_data" in data or data.get("format") == "wav"


def test_synthesize_stream_sse(monkeypatch, client):
    dummy = DummyEngine()

    async def _get_engine():
        return dummy

    app = client.app
    app.dependency_overrides[tts_router.get_tts_engine] = _get_engine

    with client.stream(
        "POST", "/api/v1/tts/synthesize/stream?text=hi&voice=v&engine=voxcpm"
    ) as resp:
        assert resp.status_code == 200
        # ensure some SSE events are present
        got = False
        for chunk in resp.iter_text():
            if "event" in chunk or "chunk" in chunk or "data" in chunk:
                got = True
                break
        assert got


def test_websocket_flow(monkeypatch, client):
    # Use TestClient websocket
    dummy = DummyEngine()

    class Registry:
        def get_tts(self, name):
            return dummy

    # Inject registry into app state
    app = client.app
    app.state.engine_registry = Registry()

    with client.websocket_connect("/api/v1/tts/synthesize/ws") as ws:
        ws.send_json({"engine": "voxcpm", "text": "hello", "voice": "v"})
        msgs = []
        try:
            while True:
                m = ws.receive_json()
                msgs.append(m)
        except Exception:
            pass

    assert any(m.get("type") in ("chunk", "complete") for m in msgs)
