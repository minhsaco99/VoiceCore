import importlib
import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class EngineConfigEntry(BaseModel):
    """Single engine configuration"""

    enabled: bool
    engine_class: str  # e.g., "app.engines.stt.whisper.engine.WhisperSTTEngine"
    config: dict[str, Any]  # Engine-specific config (model_name, device, etc.)


class EngineConfig(BaseModel):
    """Full engine configuration from YAML"""

    stt: dict[str, EngineConfigEntry] = {}
    tts: dict[str, EngineConfigEntry] = {}


def load_engine_config(config_path: str | Path) -> EngineConfig:
    """Load engine configuration from YAML file"""
    config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(f"Engine config not found: {config_path}, using defaults")
        return EngineConfig()

    with config_path.open() as f:
        data = yaml.safe_load(f)
    return EngineConfig(**data)


def create_engine_instance(engine_config: EngineConfigEntry):
    """
    Dynamically create engine instance from config

    Args:
        engine_config: Engine configuration with class path and config dict

    Returns:
        Instantiated engine (not initialized)
    """
    # Parse class path
    module_path, class_name = engine_config.engine_class.rsplit(".", 1)

    # Import module and get class
    module = importlib.import_module(module_path)
    engine_class = getattr(module, class_name)

    # Try to import config class from config.py in the same package
    config_class = None
    try:
        package_path = module_path.rsplit(".", 1)[0]
        config_module = importlib.import_module(f"{package_path}.config")

        # Look for a config class (skip base EngineConfig)
        for attr_name in dir(config_module):
            if (
                attr_name.endswith("Config")
                and attr_name != "EngineConfig"
                and not attr_name.startswith("_")
            ):
                attr = getattr(config_module, attr_name)
                if isinstance(attr, type):
                    config_class = attr
                    break
    except (ImportError, AttributeError):
        pass

    # Create config object
    if config_class:
        engine_config_obj = config_class(**engine_config.config)
    else:
        # Fallback: Use base EngineConfig
        from app.models.engine import EngineConfig as BaseEngineConfig

        engine_config_obj = BaseEngineConfig(**engine_config.config)

    # Instantiate engine
    return engine_class(engine_config_obj)
