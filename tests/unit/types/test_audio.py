"""Tests for audio type definitions"""

import io
import pathlib

import numpy as np


class TestAudioInputTypeAlias:
    """Test AudioInput type alias definition"""

    def test_audio_input_type_alias_exists(self):
        """AudioInput type alias should be defined and importable"""
        from app.types.audio import AudioInput

        # Type alias should exist
        assert AudioInput is not None

    def test_audio_input_runtime_type_check(self):
        """Runtime helper should validate AudioInput types correctly"""
        from app.types.audio import is_valid_audio_input

        # Valid types
        assert is_valid_audio_input(b"fake audio data")
        assert is_valid_audio_input(np.array([1, 2, 3]))
        assert is_valid_audio_input(pathlib.Path("/fake/path.wav"))
        assert is_valid_audio_input(io.BytesIO(b"fake data"))

        # Invalid types
        assert not is_valid_audio_input("string")
        assert not is_valid_audio_input(123)
        assert not is_valid_audio_input([1, 2, 3])
        assert not is_valid_audio_input(None)
