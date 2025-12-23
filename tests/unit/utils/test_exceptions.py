from app.utils.exceptions import (
    # Audio & Input
    AudioError,
    AudioTooLargeError,
    AudioTooShortError,
    # Configuration & Resources
    ConfigurationError,
    # Initialization & Lifecycle
    EngineInitializationError,
    # Registry
    EngineNotFoundError,
    EngineNotReadyError,
    InvalidAudioError,
    ModelLoadError,
    # Processing
    ProcessingError,
    ResourceExhaustedError,
    # Streaming
    StreamingError,
    StreamingNotSupportedError,
    StreamInterruptedError,
    SynthesisError,
    TimeoutError,
    TranscriptionError,
    UnsupportedFormatError,
    VoiceEngineError,
)


class TestExceptionHierarchy:
    """Test that exception inheritance is correct"""

    def test_all_exceptions_inherit_from_base(self):
        """All custom exceptions should inherit from VoiceEngineError"""
        exceptions = [
            EngineInitializationError,
            ModelLoadError,
            EngineNotReadyError,
            AudioError,
            UnsupportedFormatError,
            InvalidAudioError,
            AudioTooLargeError,
            AudioTooShortError,
            ProcessingError,
            TranscriptionError,
            SynthesisError,
            StreamingError,
            StreamingNotSupportedError,
            StreamInterruptedError,
            ConfigurationError,
            ResourceExhaustedError,
            TimeoutError,
            EngineNotFoundError,
        ]

        for exc_class in exceptions:
            assert issubclass(exc_class, VoiceEngineError)

    def test_model_load_error_inherits_from_init_error(self):
        """ModelLoadError should inherit from EngineInitializationError"""
        assert issubclass(ModelLoadError, EngineInitializationError)

    def test_audio_specific_errors_inherit_from_audio_error(self):
        """Audio-specific errors should inherit from AudioError"""
        assert issubclass(UnsupportedFormatError, AudioError)
        assert issubclass(InvalidAudioError, AudioError)
        assert issubclass(AudioTooLargeError, AudioError)
        assert issubclass(AudioTooShortError, AudioError)

    def test_processing_specific_errors_inherit_from_processing_error(self):
        """Processing-specific errors should inherit from ProcessingError"""
        assert issubclass(TranscriptionError, ProcessingError)
        assert issubclass(SynthesisError, ProcessingError)

    def test_streaming_specific_errors_inherit_from_streaming_error(self):
        """Streaming-specific errors should inherit from StreamingError"""
        assert issubclass(StreamingNotSupportedError, StreamingError)
        assert issubclass(StreamInterruptedError, StreamingError)


class TestExceptionMessages:
    """Test that exceptions can be raised with custom messages"""

    def test_exceptions_support_custom_messages(self):
        """All exceptions should support custom error messages"""
        msg = "Custom error message"
        exc = VoiceEngineError(msg)
        assert str(exc) == msg

    def test_audio_too_large_error_with_details(self):
        """AudioTooLargeError should support detailed messages"""
        exc = AudioTooLargeError("Audio size 50MB exceeds limit of 25MB")
        assert "50MB" in str(exc)
        assert "25MB" in str(exc)
