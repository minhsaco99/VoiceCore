"""Unit tests for VoxCPM TTS engine (structured like project tests)."""

import importlib
from unittest.mock import patch

import numpy as np
import pytest

from app.engines.tts.voxcpm.config import VoxCPMConfig
from app.models.engine import EngineConfig


class FakeVoxCPM:
    class TTSModel:
        sample_rate = 22050

    def __init__(self):
        self.tts_model = FakeVoxCPM.TTSModel()

    @classmethod
    def from_pretrained(cls, model_name: str):
        return cls()

    def generate(self, **kwargs):
        # return 1 second of zeros (float32)
        return np.zeros(self.tts_model.sample_rate, dtype=np.float32)

    def generate_streaming(self, **kwargs):
        for _ in range(3):
            yield np.zeros(1000, dtype=np.float32)


class TestVoxCPMConfig:
    def test_voxcpm_config_extends_engine_config(self):
        assert issubclass(VoxCPMConfig, EngineConfig)

    def test_voxcpm_config_defaults(self):
        cfg = VoxCPMConfig(model_name="openbmb/VoxCPM-0.5B")
        # Default numeric and boolean fields exist
        assert isinstance(cfg.cfg_value, float)
        assert isinstance(cfg.inference_timesteps, int)
        assert isinstance(cfg.normalize, bool)


class TestVoxCPMEngine:
    @pytest.fixture
    def vox_config(self):
        return VoxCPMConfig(model_name="fake-model")

    def test_engine_not_initialized_on_creation(self, vox_config):
        from app.engines.tts.voxcpm.engine import VoxCPMEngine

        engine = VoxCPMEngine(vox_config)
        assert not engine.is_ready()

    @pytest.mark.asyncio
    async def test_initialize_raises_if_dependency_missing(
        self, vox_config, monkeypatch
    ):
        # Simulate missing package
        engine_mod = importlib.import_module("app.engines.tts.voxcpm.engine")
        monkeypatch.setattr(engine_mod, "VoxCPM", None)

        from app.engines.tts.voxcpm.engine import VoxCPMEngine

        engine = VoxCPMEngine(vox_config)
        with pytest.raises(ImportError):
            await engine.initialize()

    @pytest.mark.asyncio
    async def test_explicit_initialize_calls_from_pretrained(
        self, vox_config, monkeypatch
    ):
        # Patch VoxCPM.from_pretrained to ensure it's called
        engine_mod = importlib.import_module("app.engines.tts.voxcpm.engine")
        monkeypatch.setattr(engine_mod, "VoxCPM", FakeVoxCPM)

        from app.engines.tts.voxcpm.engine import VoxCPMEngine

        engine = VoxCPMEngine(vox_config)

        with patch("app.engines.tts.voxcpm.engine.VoxCPM.from_pretrained") as mocked:
            mocked.return_value = FakeVoxCPM()
            await engine.initialize()
            mocked.assert_called_once()

        assert engine.is_ready()
        await engine.close()

    @pytest.mark.asyncio
    async def test_synthesize_and_stream(self, vox_config, monkeypatch):
        # Use fake implementation
        engine_mod = importlib.import_module("app.engines.tts.voxcpm.engine")
        monkeypatch.setattr(engine_mod, "VoxCPM", FakeVoxCPM)

        from app.engines.tts.voxcpm.engine import VoxCPMEngine
        from app.models.engine import TTSChunk, TTSResponse

        engine = VoxCPMEngine(vox_config)
        await engine.initialize()

        # Batch synthesize
        result = await engine.synthesize("Hello world", speaker_wav=b"fake-wav-bytes")
        assert isinstance(result, TTSResponse)
        assert isinstance(result.audio_data, (bytes, bytearray))
        assert result.sample_rate == 22050
        assert result.duration_seconds > 0

        # Streaming synthesize
        collected = []
        async for item in engine.synthesize_stream("Streaming test"):
            collected.append(item)

        assert len(collected) >= 2
        assert any(isinstance(x, TTSChunk) for x in collected)
        assert isinstance(collected[-1], TTSResponse)

        await engine.close()
