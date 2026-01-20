import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.config import Settings
from app.api.engine_config import create_engine_instance, load_engine_config
from app.api.registry import EngineRegistry

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Async lifespan: initialize engines at startup, cleanup at shutdown

    Loads engines from engines.yaml config file and initializes all enabled engines.
    """
    settings = Settings()

    # Load engine configuration
    engine_config = load_engine_config(settings.engine_config_path)

    # Create registry
    registry = EngineRegistry()

    # Initialize all enabled STT engines
    for name, config_entry in engine_config.stt.items():
        if config_entry.enabled:
            try:
                engine = create_engine_instance(config_entry)
                await engine.initialize()
                registry.register_stt(name, engine)
            except Exception as e:
                logger.error(f"Failed to initialize STT engine '{name}': {e}")
                # Continue loading other engines

    # Initialize all enabled TTS engines
    for name, config_entry in engine_config.tts.items():
        if config_entry.enabled:
            try:
                engine = create_engine_instance(config_entry)
                await engine.initialize()
                registry.register_tts(name, engine)
            except Exception as e:
                logger.error(f"Failed to initialize TTS engine '{name}': {e}")

    # Store registry in app.state
    app.state.engine_registry = registry

    logger.info(
        f"✓ Engine registry initialized with {len(registry.list_stt_engines())} STT and {len(registry.list_tts_engines())} TTS engines"
    )

    yield

    # Shutdown: Cleanup all engines
    for engine in registry.get_all_engines():
        try:
            if engine.is_ready():
                await engine.close()
                logger.info(f"✓ Closed engine: {engine.engine_name}")
        except Exception as e:
            logger.error(f"Error closing engine: {e}")
