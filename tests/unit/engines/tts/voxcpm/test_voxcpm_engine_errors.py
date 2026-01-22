import numpy as np
import pytest

from app.engines.tts.voxcpm.config import VoxCPMConfig
from app.engines.tts.voxcpm.engine import VoxCPMEngine
from app.exceptions import SynthesisError


class BrokenModel:
    def __init__(self):
        class T:
            sample_rate = 16000

        self.tts_model = T()

    def generate(self, **kwargs):
        raise RuntimeError("boom")

    def generate_streaming(self, **kwargs):
        raise RuntimeError("stream boom")


class StreamingRaisesModel:
    def __init__(self):
        class T:
            sample_rate = 16000

        self.tts_model = T()

    def generate(self, **kwargs):
        return np.zeros(1600, dtype=np.float32)

    def generate_streaming(self, **kwargs):
        # yield one good chunk then raise
        yield np.zeros(800, dtype=np.float32)
        raise RuntimeError("later")


@pytest.mark.asyncio
async def test_synthesize_raises_synthesis_error_on_model_failure(monkeypatch):
    class DummyV:
        @classmethod
        def from_pretrained(cls, name):
            return BrokenModel()

    monkeypatch.setattr("app.engines.tts.voxcpm.engine.VoxCPM", DummyV)
    cfg = VoxCPMConfig(model_name="x")
    engine = VoxCPMEngine(cfg)

    await engine._initialize()
    with pytest.raises(SynthesisError):
        await engine.synthesize(text="hi")


@pytest.mark.asyncio
async def test_synthesize_stream_raises_when_producer_errors(monkeypatch):
    # model whose generate_streaming yields then raises; synthesize_stream should raise SynthesisError
    class DummyV:
        @classmethod
        def from_pretrained(cls, name):
            return StreamingRaisesModel()

    monkeypatch.setattr("app.engines.tts.voxcpm.engine.VoxCPM", DummyV)
    cfg = VoxCPMConfig(model_name="x")
    engine = VoxCPMEngine(cfg)

    await engine._initialize()
    with pytest.raises(SynthesisError):
        async for _ in engine.synthesize_stream(text="hi"):
            pass
