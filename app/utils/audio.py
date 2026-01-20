"""Audio processing utilities for voice engine"""

import io
import pathlib

import librosa
import numpy as np
import soundfile as sf

from app.exceptions import InvalidAudioError, UnsupportedFormatError
from app.types.audio import AudioInput


class AudioProcessor:
    """Handles audio format conversion, resampling, and validation"""

    def to_numpy(self, audio_input: AudioInput) -> tuple[np.ndarray, int]:
        """
        Convert any AudioInput type to numpy array

        Args:
            audio_input: Audio in bytes, numpy, Path, or BytesIO format

        Returns:
            Tuple of (audio_array, sample_rate)

        Raises:
            UnsupportedFormatError: If audio_input type is not supported
            InvalidAudioError: If audio data is corrupted or invalid
        """
        try:
            if isinstance(audio_input, np.ndarray):
                # Numpy array passthrough with default 16kHz
                return audio_input, 16000

            elif isinstance(audio_input, bytes):
                # Load from bytes buffer
                buffer = io.BytesIO(audio_input)
                audio, sr = sf.read(buffer)
                return audio, sr

            elif isinstance(audio_input, pathlib.Path):
                # Load from file path
                audio, sr = sf.read(audio_input)
                return audio, sr

            elif isinstance(audio_input, io.BytesIO):
                # Load from BytesIO buffer
                audio, sr = sf.read(audio_input)
                return audio, sr

            else:
                raise UnsupportedFormatError(
                    f"Unsupported audio input type: {type(audio_input)}"
                )

        except (RuntimeError, Exception) as e:
            if isinstance(e, UnsupportedFormatError):
                raise
            raise InvalidAudioError(f"Failed to process audio data: {e}") from e

    def resample_to_16khz(self, audio: np.ndarray, current_sr: int) -> np.ndarray:
        """
        Resample audio to 16kHz (Whisper requirement)

        Args:
            audio: Audio numpy array
            current_sr: Current sample rate

        Returns:
            Resampled audio at 16kHz
        """
        # If already 16kHz, return as-is
        if current_sr == 16000:
            return audio

        # Convert stereo to mono if needed
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)

        # Resample to 16kHz
        resampled = librosa.resample(audio, orig_sr=current_sr, target_sr=16000)

        return resampled

    def get_duration_ms(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Calculate audio duration in milliseconds

        Args:
            audio: Audio numpy array
            sample_rate: Sample rate in Hz

        Returns:
            Duration in milliseconds

        Raises:
            InvalidAudioError: If audio is empty or sample_rate is invalid
        """
        if len(audio) == 0:
            raise InvalidAudioError("Audio is empty")

        if sample_rate <= 0:
            raise InvalidAudioError(f"Invalid sample rate: {sample_rate}")

        duration_seconds = len(audio) / sample_rate
        return duration_seconds * 1000.0
