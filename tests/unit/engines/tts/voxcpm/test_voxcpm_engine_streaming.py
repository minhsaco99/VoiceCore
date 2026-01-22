import numpy as np
import pytest
import soundfile as sf

from app.engines.tts.voxcpm.config import VoxCPMConfig
from app.engines.tts.voxcpm.engine import VoxCPMEngine


class DummyTTSModel:
    def __init__(self, sample_rate=16000):
        class TTS:
            pass

        self.tts_model = TTS()
        self.tts_model.sample_rate = sample_rate

    def generate(self, **kwargs):
        # return a short numpy array
        return np.zeros(1600, dtype=np.float32)

    def generate_streaming(self, **kwargs):
        for _ in range(3):
            yield np.zeros(800, dtype=np.float32)


class DummyVoxCPM:
    @classmethod
    def from_pretrained(cls, name):
        return DummyTTSModel()


def test_safe_load_works_with_small_wav(tmp_path):
    # Create a small wav file
    p = tmp_path / "test.wav"
    data = np.zeros((1600,), dtype=np.float32)
    sf.write(str(p), data, 16000)

    # torchaudio.load is monkeypatched in module; ensure it returns tensor and sr
    import torchaudio

    tensor, sr = torchaudio.load(str(p))
    assert sr == 16000
    assert tensor.ndim >= 1


@pytest.mark.asyncio
async def test_initialize_and_synthesize_with_speaker_wav(monkeypatch, tmp_path):
    # Patch VoxCPM to dummy
    monkeypatch.setattr("app.engines.tts.voxcpm.engine.VoxCPM", DummyVoxCPM)

    cfg = VoxCPMConfig(model_name="dummy")
    engine = VoxCPMEngine(cfg)

    await engine._initialize()
    # call synthesize with speaker_wav bytes
    wav_bytes = b"RIFF" + b"0" * 100
    resp = await engine.synthesize(text="hi", speaker_wav=wav_bytes)
    assert resp.format == "wav"


@pytest.mark.asyncio
async def test_synthesize_stream_yields_chunks_and_final(monkeypatch):
    monkeypatch.setattr("app.engines.tts.voxcpm.engine.VoxCPM", DummyVoxCPM)

    cfg = VoxCPMConfig(model_name="dummy")
    engine = VoxCPMEngine(cfg)

    await engine._initialize()

    chunks = []
    async for item in engine.synthesize_stream(text="hello"):
        chunks.append(item)

    # Last item should be TTSResponse
    assert len(chunks) >= 1
    assert hasattr(chunks[-1], "audio_data")


def test_initialize_raises_when_dependency_missing(monkeypatch):
    # Force VoxCPM to None
    monkeypatch.setattr("app.engines.tts.voxcpm.engine.VoxCPM", None)
    cfg = VoxCPMConfig(model_name="x")
    engine = VoxCPMEngine(cfg)

    with pytest.raises(ImportError):
        import asyncio

        asyncio.get_event_loop().run_until_complete(engine._initialize())
