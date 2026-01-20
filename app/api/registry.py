import logging

from app.engines.base import BaseSTTEngine, BaseTTSEngine
from app.exceptions import VoiceEngineError

logger = logging.getLogger(__name__)


class EngineNotFoundError(VoiceEngineError):
    """Raised when requested engine not found in registry"""

    pass


class EngineRegistry:
    """
    Centralized registry for managing multiple STT and TTS engines

    Provides:
    - Registration of engines by name
    - Retrieval of engines by name
    - Listing available engines
    - Engine metadata access
    """

    def __init__(self):
        self._stt_engines: dict[str, BaseSTTEngine] = {}
        self._tts_engines: dict[str, BaseTTSEngine] = {}

    def register_stt(self, name: str, engine: BaseSTTEngine) -> None:
        """Register an STT engine"""
        if name in self._stt_engines:
            raise ValueError(f"STT engine '{name}' already registered")
        self._stt_engines[name] = engine
        logger.info(f"✓ Registered STT engine: {name} ({engine.engine_name})")

    def register_tts(self, name: str, engine: BaseTTSEngine) -> None:
        """Register a TTS engine"""
        if name in self._tts_engines:
            raise ValueError(f"TTS engine '{name}' already registered")
        self._tts_engines[name] = engine
        logger.info(f"✓ Registered TTS engine: {name} ({engine.engine_name})")

    def get_stt(self, name: str) -> BaseSTTEngine:
        """Get STT engine by name"""
        if name not in self._stt_engines:
            available = list(self._stt_engines.keys())
            raise EngineNotFoundError(
                f"STT engine '{name}' not found. Available: {available}"
            )
        engine = self._stt_engines[name]
        if not engine.is_ready():
            raise EngineNotFoundError(f"STT engine '{name}' not ready")
        return engine

    def get_tts(self, name: str) -> BaseTTSEngine:
        """Get TTS engine by name"""
        if name not in self._tts_engines:
            available = list(self._tts_engines.keys())
            raise EngineNotFoundError(
                f"TTS engine '{name}' not found. Available: {available}"
            )
        engine = self._tts_engines[name]
        if not engine.is_ready():
            raise EngineNotFoundError(f"TTS engine '{name}' not ready")
        return engine

    def list_stt_engines(self) -> dict[str, BaseSTTEngine]:
        """List all registered STT engines"""
        return dict(self._stt_engines)

    def list_tts_engines(self) -> dict[str, BaseTTSEngine]:
        """List all registered TTS engines"""
        return dict(self._tts_engines)

    def get_all_engines(self) -> list[BaseSTTEngine | BaseTTSEngine]:
        """Get all engines for cleanup"""
        return list(self._stt_engines.values()) + list(self._tts_engines.values())
