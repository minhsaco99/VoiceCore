"""Audio type definitions for voice engine"""

import io
import pathlib

import numpy as np

# Type alias for audio input - supports multiple formats
AudioInput = bytes | np.ndarray | pathlib.Path | io.BytesIO


def is_valid_audio_input(value: object) -> bool:
    """
    Runtime type check for AudioInput

    Args:
        value: Value to check

    Returns:
        True if value matches AudioInput type, False otherwise
    """
    return isinstance(value, (bytes, np.ndarray, pathlib.Path, io.BytesIO))
