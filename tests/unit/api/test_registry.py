"""Unit tests for EngineRegistry

Focus on thread safety and concurrent access scenarios.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from app.api.registry import EngineNotFoundError, EngineRegistry
from app.engines.base import BaseSTTEngine


class TestEngineRegistryBasic:
    """Basic engine registry functionality"""

    def test_register_stt_engine(self, mock_stt_engine):
        """Register STT engine successfully"""
        registry = EngineRegistry()
        registry.register_stt("test-engine", mock_stt_engine)

        assert "test-engine" in registry.list_stt_engines()
        assert registry.get_stt("test-engine") == mock_stt_engine

    def test_register_tts_engine(self, mock_tts_engine):
        """Register TTS engine successfully"""
        registry = EngineRegistry()
        registry.register_tts("test-engine", mock_tts_engine)

        assert "test-engine" in registry.list_tts_engines()
        assert registry.get_tts("test-engine") == mock_tts_engine

    def test_duplicate_registration_raises(self, mock_stt_engine):
        """Registering same engine twice raises ValueError"""
        registry = EngineRegistry()
        registry.register_stt("test-engine", mock_stt_engine)

        with pytest.raises(ValueError, match="already registered"):
            registry.register_stt("test-engine", mock_stt_engine)

    def test_get_stt_by_name(self, mock_stt_engine):
        """Retrieve registered STT engine"""
        registry = EngineRegistry()
        registry.register_stt("whisper", mock_stt_engine)

        engine = registry.get_stt("whisper")
        assert engine == mock_stt_engine
        assert engine.engine_name == "mock-stt"

    def test_get_stt_not_found(self):
        """Get non-existent engine raises EngineNotFoundError"""
        registry = EngineRegistry()

        with pytest.raises(EngineNotFoundError, match="STT engine 'nonexistent'"):
            registry.get_stt("nonexistent")

        # Verify error message includes available engines
        registry.register_stt(
            "whisper", MagicMock(spec=BaseSTTEngine, is_ready=lambda: True)
        )
        with pytest.raises(EngineNotFoundError, match=r"Available.*whisper"):
            registry.get_stt("nonexistent")

    def test_get_stt_not_ready(self, mock_stt_engine):
        """Get engine that's not ready raises error"""
        mock_stt_engine.is_ready.return_value = False
        registry = EngineRegistry()
        registry.register_stt("not-ready", mock_stt_engine)

        with pytest.raises(EngineNotFoundError, match="not ready"):
            registry.get_stt("not-ready")

    def test_list_stt_engines(self, mock_stt_engine):
        """List all registered STT engines"""
        registry = EngineRegistry()
        registry.register_stt("engine1", mock_stt_engine)
        registry.register_stt("engine2", mock_stt_engine)

        engines = registry.list_stt_engines()
        assert len(engines) == 2
        assert "engine1" in engines
        assert "engine2" in engines

    def test_list_empty_registry(self):
        """List engines when none registered"""
        registry = EngineRegistry()

        assert registry.list_stt_engines() == {}
        assert registry.list_tts_engines() == {}

    def test_get_all_engines(self, mock_stt_engine, mock_tts_engine):
        """Get all engines for cleanup"""
        registry = EngineRegistry()
        registry.register_stt("stt1", mock_stt_engine)
        registry.register_tts("tts1", mock_tts_engine)

        all_engines = registry.get_all_engines()
        assert len(all_engines) == 2
        assert mock_stt_engine in all_engines
        assert mock_tts_engine in all_engines


class TestEngineRegistryThreadSafety:
    """CRITICAL: Thread safety and concurrent access tests"""

    @pytest.mark.asyncio
    async def test_concurrent_registration(self):
        """CRITICAL: Multiple engines registered concurrently"""
        registry = EngineRegistry()

        # Create multiple mock engines
        engines = []
        for i in range(10):
            engine = MagicMock(spec=BaseSTTEngine)
            engine.is_ready.return_value = True
            engine.engine_name = f"engine-{i}"
            engines.append((f"engine-{i}", engine))

        # Register all engines concurrently
        async def register_engine(name, engine):
            await asyncio.sleep(0.001)  # Simulate async work
            registry.register_stt(name, engine)

        await asyncio.gather(*[register_engine(name, eng) for name, eng in engines])

        # Verify all engines registered
        registered = registry.list_stt_engines()
        assert len(registered) == 10
        for name, _ in engines:
            assert name in registered

    @pytest.mark.asyncio
    async def test_concurrent_get_during_registration(self):
        """CRITICAL: Get engine while another is being registered"""
        registry = EngineRegistry()

        # Pre-register one engine
        existing_engine = MagicMock(spec=BaseSTTEngine)
        existing_engine.is_ready.return_value = True
        existing_engine.engine_name = "existing"
        registry.register_stt("existing", existing_engine)

        # Register new engines and get existing ones concurrently
        async def register_new_engine(i):
            engine = MagicMock(spec=BaseSTTEngine)
            engine.is_ready.return_value = True
            engine.engine_name = f"new-{i}"
            await asyncio.sleep(0.001)
            registry.register_stt(f"new-{i}", engine)

        async def get_existing_engine():
            await asyncio.sleep(0.001)
            return registry.get_stt("existing")

        # Run 5 registrations and 5 gets concurrently
        tasks = []
        for i in range(5):
            tasks.append(register_new_engine(i))
            tasks.append(get_existing_engine())

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check no exceptions occurred
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent access raised exception: {result}")

        # Verify state consistency
        assert len(registry.list_stt_engines()) == 6  # existing + 5 new

    @pytest.mark.asyncio
    async def test_concurrent_list_during_modification(self):
        """List engines while registry is being modified"""
        registry = EngineRegistry()

        async def add_engines():
            for i in range(10):
                engine = MagicMock(spec=BaseSTTEngine)
                engine.is_ready.return_value = True
                registry.register_stt(f"engine-{i}", engine)
                await asyncio.sleep(0.001)

        async def list_engines():
            lists = []
            for _ in range(20):
                lists.append(registry.list_stt_engines())
                await asyncio.sleep(0.001)
            return lists

        # Run concurrently
        add_task = asyncio.create_task(add_engines())
        list_task = asyncio.create_task(list_engines())

        await asyncio.gather(add_task, list_task, return_exceptions=True)

        # Verify final state
        assert len(registry.list_stt_engines()) == 10

    @pytest.mark.asyncio
    async def test_concurrent_get_with_ready_state_changes(self):
        """Get engines while ready state changes"""
        registry = EngineRegistry()

        engine = MagicMock(spec=BaseSTTEngine)
        engine.is_ready.return_value = True
        engine.engine_name = "flaky"
        registry.register_stt("flaky", engine)

        async def toggle_ready_state():
            for i in range(10):
                engine.is_ready.return_value = i % 2 == 0
                await asyncio.sleep(0.001)

        async def try_get_engine():
            results = []
            for _ in range(20):
                try:
                    eng = registry.get_stt("flaky")
                    results.append(("success", eng))
                except EngineNotFoundError:
                    results.append(("not_ready", None))
                await asyncio.sleep(0.001)
            return results

        # Run concurrently
        toggle_task = asyncio.create_task(toggle_ready_state())
        get_task = asyncio.create_task(try_get_engine())

        await asyncio.gather(toggle_task, get_task)

        # No assertion - just ensuring no crashes or race conditions


class TestEngineRegistryEdgeCases:
    """Edge cases and error scenarios"""

    def test_get_tts_not_found(self):
        """Get non-existent TTS engine raises error"""
        registry = EngineRegistry()

        with pytest.raises(EngineNotFoundError, match="TTS engine"):
            registry.get_tts("nonexistent")

    def test_register_none_engine(self):
        """Cannot register None as engine"""
        registry = EngineRegistry()

        # Registration fails immediately when accessing engine.engine_name
        with pytest.raises(AttributeError):
            registry.register_stt("none-engine", None)

    def test_list_engines_returns_copy(self):
        """List engines returns names, not mutable reference"""
        registry = EngineRegistry()
        engine = MagicMock(spec=BaseSTTEngine)
        engine.is_ready.return_value = True
        registry.register_stt("engine1", engine)

        list1 = registry.list_stt_engines()
        list2 = registry.list_stt_engines()

        # Lists should be equal but not the same object
        assert list1 == list2
        assert list1 is not list2

    def test_get_all_engines_returns_new_list(self):
        """get_all_engines returns new list each time"""
        registry = EngineRegistry()
        engine = MagicMock(spec=BaseSTTEngine)
        engine.is_ready.return_value = True
        registry.register_stt("engine1", engine)

        all1 = registry.get_all_engines()
        all2 = registry.get_all_engines()

        assert all1 == all2
        assert all1 is not all2
