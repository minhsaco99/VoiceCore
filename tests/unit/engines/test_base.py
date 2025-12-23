from collections.abc import AsyncIterator

import pytest

from app.engines.base import BaseSTTEngine, BaseTTSEngine
from app.models.engine import EngineConfig, STTChunk, STTResponse, TTSChunk, TTSResponse
from app.models.metrics import (
    STTPerformanceMetrics,
    TTSPerformanceMetrics,
)
from app.utils.exceptions import EngineNotReadyError


# Create concrete test implementations
class MockSTTEngine(BaseSTTEngine):
    """Mock STT engine for testing"""

    def __init__(self, config: EngineConfig, should_fail_init: bool = False):
        self.should_fail_init = should_fail_init
        self.init_called = False
        self.cleanup_called = False
        super().__init__(config)

    async def _initialize(self) -> None:
        if self.should_fail_init:
            raise Exception("Mock initialization failed")
        self.init_called = True

    async def _cleanup(self) -> None:
        self.cleanup_called = True

    async def transcribe(
        self, audio_data: bytes, language: str | None = None
    ) -> STTResponse:
        await self._ensure_ready()  # Auto-initialize if needed
        return STTResponse(
            text="mock transcription",
            performance_metrics=STTPerformanceMetrics(
                latency_ms=100.0, processing_time_ms=95.0
            ),
        )

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[STTChunk]:
        yield STTChunk(
            text="mock",
            is_final=True,
            chunk_latency_ms=50.0,
        )

    @property
    def supported_formats(self) -> list[str]:
        return ["wav", "mp3"]

    @property
    def engine_name(self) -> str:
        return "mock_stt"


class MockTTSEngine(BaseTTSEngine):
    """Mock TTS engine for testing"""

    def __init__(self, config: EngineConfig):
        self.init_called = False
        self.cleanup_called = False
        super().__init__(config)

    async def _initialize(self) -> None:
        self.init_called = True

    async def _cleanup(self) -> None:
        self.cleanup_called = True

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
    ) -> TTSResponse:
        await self._ensure_ready()  # Auto-initialize if needed
        return TTSResponse(
            audio_data=b"mock audio",
            sample_rate=22050,
            duration_seconds=1.0,
            performance_metrics=TTSPerformanceMetrics(
                latency_ms=100.0, processing_time_ms=95.0
            ),
        )

    async def synthesize_stream(self, text: str, **kwargs) -> AsyncIterator[TTSChunk]:
        yield TTSChunk(
            audio_data=b"mock",
            sequence_number=0,
            is_final=True,
            chunk_latency_ms=30.0,
        )

    @property
    def supported_voices(self) -> list[str]:
        return ["voice1", "voice2"]

    @property
    def engine_name(self) -> str:
        return "mock_tts"


@pytest.fixture
def engine_config():
    """Fixture for engine config"""
    return EngineConfig(model_name="test-model")


class TestBaseEngineLifecycle:
    """Test base engine lifecycle (init, ready, close)"""

    def test_engine_initialization_on_construction(self, engine_config):
        """Engine should NOT auto-initialize on construction"""
        engine = MockSTTEngine(engine_config)

        # Should not be initialized yet
        assert not engine.is_ready()
        assert not engine.init_called

    @pytest.mark.asyncio
    async def test_explicit_initialization(self, engine_config):
        """Engine should initialize when explicitly called"""
        engine = MockSTTEngine(engine_config)

        await engine.initialize()

        assert engine.is_ready()
        assert engine.init_called

    @pytest.mark.asyncio
    async def test_initialize_is_idempotent(self, engine_config):
        """Calling initialize multiple times should be safe"""
        engine = MockSTTEngine(engine_config)

        await engine.initialize()
        init_count_after_first = engine.init_called

        await engine.initialize()
        await engine.initialize()

        # Should still only have been called once
        assert engine.init_called == init_count_after_first

    @pytest.mark.asyncio
    async def test_close_engine(self, engine_config):
        """Should cleanup resources when closed"""
        engine = MockSTTEngine(engine_config)
        await engine.initialize()

        assert engine.is_ready()

        await engine.close()

        assert not engine.is_ready()
        assert engine.cleanup_called

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, engine_config):
        """Calling close multiple times should be safe"""
        engine = MockSTTEngine(engine_config)
        await engine.initialize()

        await engine.close()
        cleanup_count_after_first = engine.cleanup_called

        await engine.close()
        await engine.close()

        # Should not fail, cleanup called once
        assert engine.cleanup_called == cleanup_count_after_first

    @pytest.mark.asyncio
    async def test_cannot_use_closed_engine(self, engine_config):
        """Using a closed engine should raise error"""
        engine = MockSTTEngine(engine_config)
        await engine.initialize()
        await engine.close()

        with pytest.raises(EngineNotReadyError):
            await engine.initialize()  # Cannot re-initialize after close

    @pytest.mark.asyncio
    async def test_cannot_reinitialize_after_close(self, engine_config):
        """Cannot reinitialize a closed engine"""
        engine = MockSTTEngine(engine_config)
        await engine.initialize()
        await engine.close()

        with pytest.raises(EngineNotReadyError):
            await engine.initialize()


class TestBaseEngineAutoInitialization:
    """Test auto-initialization on first use"""

    @pytest.mark.asyncio
    async def test_auto_initialize_on_transcribe(self, engine_config):
        """Engine should auto-initialize on first transcribe call"""
        engine = MockSTTEngine(engine_config)

        assert not engine.is_ready()

        # First call should trigger initialization
        result = await engine.transcribe(b"fake audio")

        assert engine.is_ready()
        assert engine.init_called
        assert result.text == "mock transcription"

    @pytest.mark.asyncio
    async def test_transcribe_fails_if_closed(self, engine_config):
        """Transcribe should fail on closed engine"""
        engine = MockSTTEngine(engine_config)
        await engine.initialize()
        await engine.close()

        with pytest.raises(EngineNotReadyError):
            await engine.transcribe(b"fake audio")


class TestBaseEngineContextManager:
    """Test async context manager support"""

    @pytest.mark.asyncio
    async def test_context_manager_lifecycle(self, engine_config):
        """Should support async with statement"""
        engine = MockSTTEngine(engine_config)

        async with engine:
            # Should be initialized inside context
            assert engine.is_ready()

            result = await engine.transcribe(b"fake audio")
            assert result.text == "mock transcription"

        # Should be closed after context
        assert not engine.is_ready()
        assert engine.cleanup_called

    @pytest.mark.asyncio
    async def test_context_manager_cleans_up_on_error(self, engine_config):
        """Should cleanup even if error occurs in context"""
        engine = MockSTTEngine(engine_config)

        with pytest.raises(ValueError):
            async with engine:
                assert engine.is_ready()
                raise ValueError("Test error")

        # Should still be closed
        assert not engine.is_ready()
        assert engine.cleanup_called


class TestBaseTTSEngine:
    """Test TTS engine lifecycle and methods"""

    @pytest.mark.asyncio
    async def test_tts_engine_lifecycle(self, engine_config):
        """TTS engine should follow same lifecycle as STT"""
        engine = MockTTSEngine(engine_config)

        assert not engine.is_ready()

        await engine.initialize()
        assert engine.is_ready()

        await engine.close()
        assert not engine.is_ready()
        assert engine.cleanup_called

    @pytest.mark.asyncio
    async def test_tts_synthesize(self, engine_config):
        """Should synthesize text to audio"""
        engine = MockTTSEngine(engine_config)

        result = await engine.synthesize("Hello world")

        assert result.audio_data == b"mock audio"
        assert result.sample_rate == 22050
        assert engine.is_ready()  # Auto-initialized

    @pytest.mark.asyncio
    async def test_tts_synthesize_with_params(self, engine_config):
        """Should accept voice and speed parameters"""
        engine = MockTTSEngine(engine_config)

        result = await engine.synthesize("Hello", voice="voice1", speed=1.5)

        assert result.audio_data == b"mock audio"

    @pytest.mark.asyncio
    async def test_tts_context_manager(self, engine_config):
        """TTS should support context manager"""
        engine = MockTTSEngine(engine_config)

        async with engine:
            result = await engine.synthesize("Test")
            assert result.audio_data == b"mock audio"

        assert not engine.is_ready()
