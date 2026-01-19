"""Shared fixtures for integration tests"""

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
async def api_client():
    """Async HTTP client for integration tests"""
    from app.api.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
def test_audio_file():
    """Path to real test audio file"""
    # Use existing test audio from engine tests
    test_audio_path = (
        Path(__file__).parent.parent / "fixtures" / "audio" / "test_audio.wav"
    )
    if test_audio_path.exists():
        return test_audio_path
    # Fallback to any available test audio
    return None


@pytest.fixture
def temp_engines_yaml(tmp_path):
    """Create temporary engines.yaml for testing"""
    yaml_content = """stt:
  whisper:
    enabled: true
    engine_class: "app.engines.stt.whisper.engine.WhisperSTTEngine"
    config:
      model_name: "base"
      device: "cpu"
      compute_type: "int8"
      timeout_seconds: 300

tts: {}
"""
    yaml_file = tmp_path / "engines.yaml"
    yaml_file.write_text(yaml_content)
    return yaml_file
