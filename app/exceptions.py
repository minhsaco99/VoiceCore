"""
Voice Engine custom exceptions

Granular exception hierarchy for precise error handling
"""


class VoiceEngineError(Exception):
    """Base exception for all voice engine errors"""

    pass


# Initialization & Lifecycle Errors
class EngineInitializationError(VoiceEngineError):
    """Failed to initialize engine"""

    pass


class ModelLoadError(EngineInitializationError):
    """Failed to load ML model"""

    pass


class EngineNotReadyError(VoiceEngineError):
    """Engine not initialized or has been closed"""

    pass


# Audio & Input Errors
class AudioError(VoiceEngineError):
    """Base for audio-related errors"""

    pass


class UnsupportedFormatError(AudioError):
    """Audio format not supported by engine"""

    pass


class InvalidAudioError(AudioError):
    """Audio data is corrupted or invalid"""

    pass


class AudioTooLargeError(AudioError):
    """Audio exceeds size limits"""

    pass


class AudioTooShortError(AudioError):
    """Audio too short to process"""

    pass


# Processing Errors
class ProcessingError(VoiceEngineError):
    """Error during audio processing"""

    pass


class TranscriptionError(ProcessingError):
    """STT-specific processing error"""

    pass


class SynthesisError(ProcessingError):
    """TTS-specific processing error"""

    pass


# Streaming Errors
class StreamingError(VoiceEngineError):
    """Streaming-related errors"""

    pass


class StreamingNotSupportedError(StreamingError):
    """Engine doesn't support streaming"""

    pass


class StreamInterruptedError(StreamingError):
    """Stream was interrupted"""

    pass


# Configuration & Resources
class ConfigurationError(VoiceEngineError):
    """Invalid configuration"""

    pass


class ResourceExhaustedError(VoiceEngineError):
    """Out of memory, GPU, etc."""

    pass


class TimeoutError(VoiceEngineError):
    """Processing timeout"""

    pass


class TranscriptionTimeoutError(TimeoutError):
    """STT transcription timeout"""

    pass


# Registry
class EngineNotFoundError(VoiceEngineError):
    """Engine not registered"""

    pass
