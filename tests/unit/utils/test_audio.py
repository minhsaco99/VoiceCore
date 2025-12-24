"""Tests for audio processing utilities"""
import io
import pathlib
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

from app.exceptions import InvalidAudioError, UnsupportedFormatError


class TestAudioProcessorToNumpy:
    """Test AudioProcessor.to_numpy() method"""

    def test_to_numpy_from_bytes_wav(self):
        """Should convert WAV bytes to numpy array with sample rate"""
        from app.utils.audio import AudioProcessor

        # Mock WAV data (16kHz, mono)
        fake_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        fake_sr = 16000

        processor = AudioProcessor()

        with patch("soundfile.read") as mock_read:
            mock_read.return_value = (fake_audio, fake_sr)

            audio, sr = processor.to_numpy(b"fake wav data")

            assert isinstance(audio, np.ndarray)
            assert sr == 16000
            np.testing.assert_array_equal(audio, fake_audio)
            mock_read.assert_called_once()

    def test_to_numpy_from_numpy_passthrough(self):
        """Should return numpy array as-is with default 16kHz sample rate"""
        from app.utils.audio import AudioProcessor

        fake_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        processor = AudioProcessor()

        audio, sr = processor.to_numpy(fake_audio)

        assert isinstance(audio, np.ndarray)
        assert sr == 16000  # Default sample rate
        np.testing.assert_array_equal(audio, fake_audio)

    def test_to_numpy_from_path(self):
        """Should load audio file from Path to numpy"""
        from app.utils.audio import AudioProcessor

        fake_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        fake_sr = 22050
        fake_path = pathlib.Path("/fake/audio.wav")

        processor = AudioProcessor()

        with patch("soundfile.read") as mock_read:
            mock_read.return_value = (fake_audio, fake_sr)

            audio, sr = processor.to_numpy(fake_path)

            assert isinstance(audio, np.ndarray)
            assert sr == 22050
            np.testing.assert_array_equal(audio, fake_audio)
            mock_read.assert_called_once_with(fake_path)

    def test_to_numpy_from_bytesio(self):
        """Should load audio from BytesIO to numpy"""
        from app.utils.audio import AudioProcessor

        fake_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        fake_sr = 16000
        fake_buffer = io.BytesIO(b"fake audio data")

        processor = AudioProcessor()

        with patch("soundfile.read") as mock_read:
            mock_read.return_value = (fake_audio, fake_sr)

            audio, sr = processor.to_numpy(fake_buffer)

            assert isinstance(audio, np.ndarray)
            assert sr == 16000
            np.testing.assert_array_equal(audio, fake_audio)
            mock_read.assert_called_once_with(fake_buffer)

    def test_to_numpy_raises_on_unsupported_type(self):
        """Should raise UnsupportedFormatError for invalid types like str/int"""
        from app.utils.audio import AudioProcessor

        processor = AudioProcessor()

        with pytest.raises(UnsupportedFormatError):
            processor.to_numpy("invalid_string")

        with pytest.raises(UnsupportedFormatError):
            processor.to_numpy(123)

        with pytest.raises(UnsupportedFormatError):
            processor.to_numpy([1, 2, 3])

    def test_to_numpy_raises_on_invalid_audio(self):
        """Should raise InvalidAudioError for corrupted data"""
        from app.utils.audio import AudioProcessor

        processor = AudioProcessor()

        with patch("soundfile.read") as mock_read:
            # Simulate soundfile failing to read corrupted data
            mock_read.side_effect = RuntimeError("Invalid audio file")

            with pytest.raises(InvalidAudioError):
                processor.to_numpy(b"corrupted data")


class TestAudioProcessorResample:
    """Test AudioProcessor.resample_to_16khz() method"""

    def test_resample_to_16khz(self):
        """Should resample 44.1kHz audio to 16kHz"""
        from app.utils.audio import AudioProcessor

        # Create fake 44.1kHz audio (1 second)
        fake_audio_44k = np.random.randn(44100).astype(np.float32)
        processor = AudioProcessor()

        with patch("librosa.resample") as mock_resample:
            # Mock returns 16kHz audio (shorter)
            fake_audio_16k = np.random.randn(16000).astype(np.float32)
            mock_resample.return_value = fake_audio_16k

            resampled = processor.resample_to_16khz(fake_audio_44k, 44100)

            assert isinstance(resampled, np.ndarray)
            mock_resample.assert_called_once()
            # Check that librosa.resample was called with correct params
            call_args = mock_resample.call_args
            assert call_args[1]["orig_sr"] == 44100
            assert call_args[1]["target_sr"] == 16000

    def test_resample_already_16khz_passthrough(self):
        """Should not resample if already 16kHz"""
        from app.utils.audio import AudioProcessor

        fake_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        processor = AudioProcessor()

        resampled = processor.resample_to_16khz(fake_audio, 16000)

        # Should return the same audio without modification
        np.testing.assert_array_equal(resampled, fake_audio)

    def test_resample_mono_conversion(self):
        """Should convert stereo to mono during resample"""
        from app.utils.audio import AudioProcessor

        # Create fake stereo audio (2 channels)
        fake_stereo = np.random.randn(2, 44100).astype(np.float32)
        processor = AudioProcessor()

        with patch("librosa.to_mono") as mock_to_mono, \
             patch("librosa.resample") as mock_resample:

            fake_mono = np.random.randn(44100).astype(np.float32)
            mock_to_mono.return_value = fake_mono
            mock_resample.return_value = np.random.randn(16000).astype(np.float32)

            resampled = processor.resample_to_16khz(fake_stereo, 44100)

            # Should have called to_mono first
            mock_to_mono.assert_called_once()


class TestAudioProcessorDuration:
    """Test AudioProcessor.get_duration_ms() method"""

    def test_get_duration_ms(self):
        """Should calculate audio duration in milliseconds correctly"""
        from app.utils.audio import AudioProcessor

        processor = AudioProcessor()

        # 16000 samples at 16kHz = 1 second = 1000ms
        audio_1s = np.zeros(16000)
        duration = processor.get_duration_ms(audio_1s, 16000)
        assert duration == 1000.0

        # 8000 samples at 16kHz = 0.5 seconds = 500ms
        audio_half_s = np.zeros(8000)
        duration = processor.get_duration_ms(audio_half_s, 16000)
        assert duration == 500.0

        # 44100 samples at 44.1kHz = 1 second = 1000ms
        audio_44k = np.zeros(44100)
        duration = processor.get_duration_ms(audio_44k, 44100)
        assert duration == 1000.0

    def test_get_duration_ms_raises_on_empty(self):
        """Should raise error for empty/invalid audio"""
        from app.utils.audio import AudioProcessor

        processor = AudioProcessor()

        # Empty audio
        with pytest.raises(InvalidAudioError):
            processor.get_duration_ms(np.array([]), 16000)

        # Invalid sample rate (zero or negative)
        with pytest.raises(InvalidAudioError):
            processor.get_duration_ms(np.array([1, 2, 3]), 0)

        with pytest.raises(InvalidAudioError):
            processor.get_duration_ms(np.array([1, 2, 3]), -1000)
