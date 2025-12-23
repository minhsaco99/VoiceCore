"""
Abstract base classes for STT and TTS engines

Key design principles:
- Async everywhere
- Stateless engines
- Lifecycle management (initialize, close, is_ready)
- Auto-initialization on first use
- Context manager support
- Standardized I/O via EngineConfig, STTOutput, TTSOutput
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.models.engine import EngineConfig, STTChunk, STTOutput, TTSChunk, TTSOutput
from app.utils.exceptions import EngineNotReadyError


class BaseEngine(ABC):
    """
    Abstract base class for all voice engines (STT and TTS)

    Provides shared lifecycle management:
    - Initialization and cleanup
    - State tracking (_initialized, _closed)
    - Auto-initialization on first use
    - Context manager support
    - Idempotent operations

    Subclasses must implement:
    - _initialize() - Load models/resources
    - _cleanup() - Cleanup resources
    - engine_name property
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self._initialized = False
        self._closed = False

    async def initialize(self) -> None:
        """
        Initialize engine (idempotent)
        Safe to call multiple times
        """
        if self._initialized:
            return
        if self._closed:
            raise EngineNotReadyError(
                "Engine has been closed and cannot be reinitialized"
            )

        await self._initialize()
        self._initialized = True

    async def close(self) -> None:
        """
        Close engine and cleanup resources (idempotent)
        After calling this, engine cannot be used again
        """
        if self._closed:
            return

        if self._initialized:
            await self._cleanup()

        self._initialized = False
        self._closed = True

    def is_ready(self) -> bool:
        """Check if engine is initialized and ready"""
        return self._initialized and not self._closed

    async def _ensure_ready(self) -> None:
        """Ensure engine is ready (auto-initialize if needed)"""
        if self._closed:
            raise EngineNotReadyError("Engine has been closed")
        if not self._initialized:
            await self.initialize()

    # Context manager support
    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False

    # Abstract methods subclasses must implement
    @abstractmethod
    async def _initialize(self) -> None:
        """Initialize engine resources (load models, etc.)"""
        pass

    @abstractmethod
    async def _cleanup(self) -> None:
        """Cleanup engine resources"""
        pass

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Engine name (e.g., 'whisper', 'coqui')"""
        pass


class BaseSTTEngine(BaseEngine):
    """
    Abstract base class for Speech-to-Text engines

    All STT engines must inherit from this class and implement:
    - _initialize() - Load models/resources
    - _cleanup() - Cleanup resources
    - transcribe() - Batch processing (returns STTOutput with STTInvokePerformanceMetrics)
    - transcribe_stream() - Streaming (yields STTChunk with STTStreamPerformanceMetrics)
    - supported_formats property
    - engine_name property

    Lifecycle:
    1. Create instance: engine = WhisperSTTEngine(config)
    2. Auto-initializes on first use OR explicit: await engine.initialize()
    3. Use: result = await engine.transcribe(audio)
    4. Cleanup: await engine.close() OR use async with
    """

    @abstractmethod
    async def transcribe(
        self, audio_data: bytes, language: str | None = None
    ) -> STTOutput:
        """
        Transcribe audio (invoke/batch mode)

        Args:
            audio_data: Audio file as bytes
            language: Optional language hint (e.g., "en", "es")

        Returns:
            STTOutput with text and STTInvokePerformanceMetrics
        """
        pass

    @abstractmethod
    async def transcribe_stream(
        self, audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[STTChunk]:
        """
        Transcribe audio stream (streaming mode)

        Args:
            audio_stream: Async iterator of audio chunks

        Yields:
            STTChunk with partial/final text and STTStreamPerformanceMetrics
        """
        pass

    @property
    @abstractmethod
    def supported_formats(self) -> list[str]:
        """List of supported audio formats (e.g., ['wav', 'mp3'])"""
        pass


class BaseTTSEngine(BaseEngine):
    """
    Abstract base class for Text-to-Speech engines

    All TTS engines must implement:
    - _initialize() - Load models/resources
    - _cleanup() - Cleanup resources
    - synthesize() - Batch processing (returns TTSOutput with TTSInvokePerformanceMetrics)
    - synthesize_stream() - Streaming (yields TTSChunk with TTSStreamPerformanceMetrics)
    - supported_voices property
    - engine_name property

    Lifecycle same as BaseSTTEngine
    """

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> TTSOutput:
        """
        Synthesize text to speech (invoke/batch mode)

        Args:
            text: Text to synthesize
            voice: Optional voice name (overrides config default)
            speed: Speech speed (1.0 = normal, overrides config default)

        Returns:
            TTSOutput with audio and TTSInvokePerformanceMetrics
        """
        pass

    @abstractmethod
    async def synthesize_stream(self, text: str, **kwargs) -> AsyncIterator[TTSChunk]:
        """
        Synthesize text to speech (streaming mode)

        Args:
            text: Text to synthesize
            **kwargs: Engine-specific params (voice, speed, etc.)

        Yields:
            TTSChunk with audio and TTSStreamPerformanceMetrics
        """
        pass

    @property
    @abstractmethod
    def supported_voices(self) -> list[str]:
        """List of available voices"""
        pass
