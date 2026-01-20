# Custom Engine Development Guide

This guide explains how to create custom STT and TTS engines for Voice Engine API.

## Architecture Overview

The framework uses a plugin pattern with abstract base classes:

```
BaseEngine (shared lifecycle management)
├── BaseSTTEngine (speech-to-text interface)
└── BaseTTSEngine (text-to-speech interface)
```

Each engine:
- Has a configuration class extending `EngineConfig`
- Implements async lifecycle methods (`_initialize`, `_cleanup`)
- Implements processing methods (`transcribe`/`synthesize`)
- Is registered in `engines.yaml`

---

## Creating an STT Engine

### Step 1: Add Engine Dependencies to pyproject.toml

Each engine should have its dependencies in a separate dependency group in `pyproject.toml`. This keeps engine-specific packages isolated and allows users to install only what they need.

```toml
# pyproject.toml

[dependency-groups]
# Existing groups
whisper = [
    "faster-whisper>=1.2.1",
]

# Add your engine's dependencies as a new group
myengine = [
    "my-stt-library>=1.0.0",
    "some-other-dependency>=2.0.0",
]

# For cloud-based engines
google-stt = [
    "google-cloud-speech>=2.0.0",
]

azure-stt = [
    "azure-cognitiveservices-speech>=1.30.0",
]

# TTS engines
coqui-tts = [
    "TTS>=0.20.0",
]
```

**Install engine dependencies:**

```bash
# Install specific engine
uv sync --group myengine

# Install multiple engines
uv sync --group whisper --group myengine

# Install all groups (not recommended for production)
uv sync --all-groups
```

### Step 2: Create Directory Structure

```
app/engines/stt/myengine/
├── __init__.py
├── config.py      # Configuration class
└── engine.py      # Engine implementation
```

### Step 3: Create Configuration Class

```python
# app/engines/stt/myengine/config.py

from pydantic import Field, field_validator
from app.models.engine import EngineConfig


class MyEngineConfig(EngineConfig):
    """Configuration for My STT Engine"""

    # Override model_name validation if needed
    @field_validator("model_name")
    @classmethod
    def validate_model(cls, v: str) -> str:
        valid = ["model-small", "model-large"]
        if v not in valid:
            raise ValueError(f"Model must be one of {valid}")
        return v

    # Add engine-specific fields
    api_key: str | None = Field(None, description="API key for cloud service")
    sample_rate: int = Field(default=16000, description="Expected sample rate")
    custom_param: str = Field(default="value", description="Custom parameter")
```

### Step 4: Implement Engine Class

```python
# app/engines/stt/myengine/engine.py

import time
from collections.abc import AsyncIterator

from app.engines.base import BaseSTTEngine
from app.engines.stt.myengine.config import MyEngineConfig
from app.exceptions import (
    EngineNotReadyError,
    InvalidAudioError,
    TranscriptionError,
)
from app.models.engine import Segment, STTChunk, STTResponse
from app.models.metrics import STTPerformanceMetrics
from app.types.audio import AudioInput
from app.utils.audio import AudioProcessor


class MySTTEngine(BaseSTTEngine):
    """My custom STT engine implementation"""

    def __init__(self, config: MyEngineConfig):
        super().__init__(config)
        self.my_config = config
        self._model = None
        self._audio_processor = AudioProcessor()

    async def _initialize(self) -> None:
        """
        Load models/resources.
        Called once when engine starts.
        """
        # Load your model here
        # self._model = load_model(self.my_config.model_name)
        pass

    async def _cleanup(self) -> None:
        """
        Cleanup resources.
        Called when engine shuts down.
        """
        self._model = None

    async def transcribe(
        self,
        audio_data: AudioInput,
        language: str | None = None,
        **kwargs
    ) -> STTResponse:
        """
        Transcribe audio (batch mode).

        Args:
            audio_data: Audio in bytes, numpy, Path, or BytesIO format
            language: Optional language hint
            **kwargs: Additional parameters from engine_params

        Returns:
            STTResponse with transcription and metrics
        """
        start_time = time.time()

        # Ensure engine is ready (auto-initializes if needed)
        await self._ensure_ready()

        if self._model is None:
            raise EngineNotReadyError("Model not loaded")

        # Convert audio to numpy
        audio_array, sample_rate = self._audio_processor.to_numpy(audio_data)

        if len(audio_array) == 0:
            raise InvalidAudioError("Audio is empty")

        # Get audio duration
        audio_duration_ms = self._audio_processor.get_duration_ms(
            audio_array, sample_rate
        )

        # Process audio
        processing_start = time.time()
        try:
            # Your transcription logic here
            text = "transcribed text"
            detected_language = language or "en"
            segments = []  # Word-level segments if available

        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}") from e

        processing_end = time.time()
        end_time = time.time()

        # Calculate metrics
        latency_ms = (end_time - start_time) * 1000
        processing_time_ms = (processing_end - processing_start) * 1000
        real_time_factor = (
            processing_time_ms / audio_duration_ms
            if audio_duration_ms > 0 else None
        )

        metrics = STTPerformanceMetrics(
            latency_ms=latency_ms,
            processing_time_ms=processing_time_ms,
            audio_duration_ms=audio_duration_ms,
            real_time_factor=real_time_factor,
        )

        return STTResponse(
            text=text,
            language=detected_language,
            segments=segments if segments else None,
            performance_metrics=metrics,
        )

    async def transcribe_stream(
        self,
        audio_data: AudioInput,
        language: str | None = None,
        **kwargs
    ) -> AsyncIterator[STTChunk | STTResponse]:
        """
        Streaming transcription.

        Yields STTChunk for partial results, then STTResponse for final.
        """
        start_time = time.time()
        first_token_time = None
        accumulated_text = []
        total_chunks = 0

        await self._ensure_ready()

        if self._model is None:
            raise EngineNotReadyError("Model not loaded")

        try:
            audio_array, sample_rate = self._audio_processor.to_numpy(audio_data)

            if len(audio_array) == 0:
                raise InvalidAudioError("Audio is empty")

            audio_duration_ms = self._audio_processor.get_duration_ms(
                audio_array, sample_rate
            )

            # Your streaming logic here
            # For each partial result:
            for partial_text in ["Hello", "world"]:
                chunk_start = time.time()

                if first_token_time is None:
                    first_token_time = time.time()

                accumulated_text.append(partial_text)
                total_chunks += 1

                chunk_latency_ms = (time.time() - chunk_start) * 1000

                yield STTChunk(
                    text=partial_text,
                    timestamp=None,
                    confidence=None,
                    chunk_latency_ms=chunk_latency_ms,
                )

            # Final response
            end_time = time.time()
            total_duration_ms = (end_time - start_time) * 1000
            time_to_first_token_ms = (
                (first_token_time - start_time) * 1000
                if first_token_time else None
            )

            metrics = STTPerformanceMetrics(
                latency_ms=total_duration_ms,
                processing_time_ms=total_duration_ms,
                audio_duration_ms=audio_duration_ms,
                real_time_factor=(
                    total_duration_ms / audio_duration_ms
                    if audio_duration_ms > 0 else None
                ),
                time_to_first_token_ms=time_to_first_token_ms,
                total_stream_duration_ms=total_duration_ms,
                total_chunks=total_chunks,
            )

            yield STTResponse(
                text=" ".join(accumulated_text),
                language=language,
                segments=None,
                performance_metrics=metrics,
            )

        except Exception as e:
            if isinstance(e, (InvalidAudioError, TranscriptionError, EngineNotReadyError)):
                raise
            raise TranscriptionError(f"Stream failed: {e}") from e

    @property
    def supported_formats(self) -> list[str]:
        """List of supported audio formats"""
        return ["wav", "mp3", "flac"]

    @property
    def engine_name(self) -> str:
        """Engine name for identification"""
        return "my-engine"
```

### Step 5: Register in engines.yaml

```yaml
stt:
  myengine:
    enabled: true
    engine_class: "app.engines.stt.myengine.engine.MySTTEngine"
    config:
      model_name: "model-small"
      device: "cpu"
      api_key: "${MY_ENGINE_API_KEY}"
      custom_param: "custom_value"
```

---

## Creating a TTS Engine

### Step 1: Add Engine Dependencies to pyproject.toml

Same as STT engines, add your TTS dependencies as a separate group:

```toml
# pyproject.toml

[dependency-groups]
# TTS engine example
mytts = [
    "my-tts-library>=1.0.0",
]

coqui-tts = [
    "TTS>=0.20.0",
]
```

```bash
# Install TTS engine dependencies
uv sync --group mytts
```

### Step 2: Create Directory Structure

```
app/engines/tts/myengine/
├── __init__.py
├── config.py
└── engine.py
```

### Step 3: Create Configuration Class

```python
# app/engines/tts/myengine/config.py

from pydantic import Field
from app.models.engine import EngineConfig


class MyTTSConfig(EngineConfig):
    """Configuration for My TTS Engine"""

    default_voice: str = Field(default="en-US-1", description="Default voice")
    default_speed: float = Field(default=1.0, description="Default speech speed")
    output_sample_rate: int = Field(default=22050, description="Output sample rate")
```

### Step 4: Implement Engine Class

```python
# app/engines/tts/myengine/engine.py

import time
from collections.abc import AsyncIterator

from app.engines.base import BaseTTSEngine
from app.engines.tts.myengine.config import MyTTSConfig
from app.exceptions import EngineNotReadyError, SynthesisError
from app.models.engine import TTSChunk, TTSResponse
from app.models.metrics import TTSPerformanceMetrics


class MyTTSEngine(BaseTTSEngine):
    """My custom TTS engine implementation"""

    def __init__(self, config: MyTTSConfig):
        super().__init__(config)
        self.tts_config = config
        self._model = None

    async def _initialize(self) -> None:
        """Load TTS model"""
        # self._model = load_tts_model(self.tts_config.model_name)
        pass

    async def _cleanup(self) -> None:
        """Cleanup resources"""
        self._model = None

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        **kwargs
    ) -> TTSResponse:
        """
        Synthesize text to speech (batch mode).

        Args:
            text: Text to synthesize
            voice: Voice name (uses default if None)
            speed: Speech speed multiplier
            **kwargs: Additional engine parameters

        Returns:
            TTSResponse with audio and metrics
        """
        start_time = time.time()

        await self._ensure_ready()

        if self._model is None:
            raise EngineNotReadyError("Model not loaded")

        voice = voice or self.tts_config.default_voice

        processing_start = time.time()
        try:
            # Your synthesis logic here
            audio_data = b"..."  # Generated audio bytes
            duration_seconds = 1.0  # Calculated duration

        except Exception as e:
            raise SynthesisError(f"Synthesis failed: {e}") from e

        processing_end = time.time()
        end_time = time.time()

        latency_ms = (end_time - start_time) * 1000
        processing_time_ms = (processing_end - processing_start) * 1000

        metrics = TTSPerformanceMetrics(
            latency_ms=latency_ms,
            processing_time_ms=processing_time_ms,
            audio_duration_ms=duration_seconds * 1000,
            real_time_factor=(
                processing_time_ms / (duration_seconds * 1000)
                if duration_seconds > 0 else None
            ),
            characters_processed=len(text),
        )

        return TTSResponse(
            audio_data=audio_data,
            sample_rate=self.tts_config.output_sample_rate,
            duration_seconds=duration_seconds,
            format="wav",
            performance_metrics=metrics,
        )

    async def synthesize_stream(
        self,
        text: str,
        **kwargs
    ) -> AsyncIterator[TTSChunk | TTSResponse]:
        """
        Streaming synthesis.

        Yields TTSChunk for audio chunks, then TTSResponse for final.
        """
        start_time = time.time()
        total_chunks = 0
        total_audio = b""

        await self._ensure_ready()

        if self._model is None:
            raise EngineNotReadyError("Model not loaded")

        try:
            # Stream audio chunks
            for i, audio_chunk in enumerate([b"chunk1", b"chunk2"]):
                chunk_start = time.time()
                total_chunks += 1
                total_audio += audio_chunk

                yield TTSChunk(
                    audio_data=audio_chunk,
                    sequence_number=i,
                    chunk_latency_ms=(time.time() - chunk_start) * 1000,
                )

            # Final response
            end_time = time.time()
            duration_seconds = len(total_audio) / self.tts_config.output_sample_rate

            metrics = TTSPerformanceMetrics(
                latency_ms=(end_time - start_time) * 1000,
                processing_time_ms=(end_time - start_time) * 1000,
                audio_duration_ms=duration_seconds * 1000,
                characters_processed=len(text),
                total_chunks=total_chunks,
            )

            yield TTSResponse(
                audio_data=total_audio,
                sample_rate=self.tts_config.output_sample_rate,
                duration_seconds=duration_seconds,
                format="wav",
                performance_metrics=metrics,
            )

        except Exception as e:
            raise SynthesisError(f"Stream synthesis failed: {e}") from e

    @property
    def supported_voices(self) -> list[str]:
        """List of available voices"""
        return ["en-US-1", "en-US-2", "en-GB-1"]

    @property
    def engine_name(self) -> str:
        """Engine name"""
        return "my-tts-engine"
```

---

## Base Class Reference

### BaseEngine

Common lifecycle management for all engines.

```python
class BaseEngine(ABC):
    def __init__(self, config: EngineConfig):
        self.config = config
        self._initialized = False
        self._closed = False

    async def initialize(self) -> None:
        """Initialize engine (idempotent, safe to call multiple times)"""

    async def close(self) -> None:
        """Close engine and cleanup (idempotent)"""

    def is_ready(self) -> bool:
        """Check if engine is ready for processing"""

    async def _ensure_ready(self) -> None:
        """Auto-initialize if needed, raise if closed"""

    # Context manager support
    async def __aenter__(self): ...
    async def __aexit__(self, ...): ...

    # Abstract methods - must implement
    @abstractmethod
    async def _initialize(self) -> None: ...

    @abstractmethod
    async def _cleanup(self) -> None: ...

    @property
    @abstractmethod
    def engine_name(self) -> str: ...
```

### BaseSTTEngine

Additional methods for STT engines.

```python
class BaseSTTEngine(BaseEngine):
    @abstractmethod
    async def transcribe(
        self,
        audio_data: AudioInput,
        language: str | None = None,
        **kwargs
    ) -> STTResponse: ...

    @abstractmethod
    async def transcribe_stream(
        self,
        audio_data: AudioInput,
        language: str | None = None,
        **kwargs
    ) -> AsyncIterator[STTChunk | STTResponse]: ...

    @property
    @abstractmethod
    def supported_formats(self) -> list[str]: ...
```

### BaseTTSEngine

Additional methods for TTS engines.

```python
class BaseTTSEngine(BaseEngine):
    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        **kwargs
    ) -> TTSResponse: ...

    @abstractmethod
    async def synthesize_stream(
        self,
        text: str,
        **kwargs
    ) -> AsyncIterator[TTSChunk | TTSResponse]: ...

    @property
    @abstractmethod
    def supported_voices(self) -> list[str]: ...
```

---

## AudioInput Types

The `AudioInput` type alias accepts multiple formats:

```python
AudioInput = bytes | np.ndarray | Path | BytesIO
```

Use `AudioProcessor` for conversion:

```python
from app.utils.audio import AudioProcessor

processor = AudioProcessor()

# Convert any input to numpy array
audio_array, sample_rate = processor.to_numpy(audio_data)

# Resample to 16kHz (common for STT)
audio_16k = processor.resample_to_16khz(audio_array, sample_rate)

# Get duration in milliseconds
duration_ms = processor.get_duration_ms(audio_array, sample_rate)
```

---

## Exception Types

Use these exceptions for proper error handling:

```python
from app.exceptions import (
    EngineNotReadyError,   # Engine not initialized or closed
    InvalidAudioError,     # Invalid or empty audio
    UnsupportedFormatError,# Unsupported audio format
    TranscriptionError,    # STT processing failed
    SynthesisError,        # TTS processing failed
)
```

---

## Testing Your Engine

Create tests under `tests/unit/engines/`:

```python
# tests/unit/engines/test_myengine.py

import pytest
from app.engines.stt.myengine.config import MyEngineConfig
from app.engines.stt.myengine.engine import MySTTEngine


@pytest.fixture
def config():
    return MyEngineConfig(model_name="model-small")


@pytest.fixture
async def engine(config):
    engine = MySTTEngine(config)
    await engine.initialize()
    yield engine
    await engine.close()


@pytest.mark.asyncio
async def test_transcribe(engine):
    audio_data = b"..."  # Test audio
    result = await engine.transcribe(audio_data)
    assert result.text
    assert result.performance_metrics


@pytest.mark.asyncio
async def test_transcribe_stream(engine):
    audio_data = b"..."
    chunks = []
    async for item in engine.transcribe_stream(audio_data):
        chunks.append(item)

    # Should end with STTResponse
    assert isinstance(chunks[-1], STTResponse)
```

Run tests:

```bash
make test-unit
```
