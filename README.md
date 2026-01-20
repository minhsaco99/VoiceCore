# Voice Engine API

A modular, async-native FastAPI framework for Speech-to-Text (STT) and Text-to-Speech (TTS) engines.

![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)

## Features

- **Multi-engine architecture** - Plugin pattern for swappable STT/TTS engines
- **Multiple API modes** - REST, SSE streaming, and WebSocket real-time
- **YAML-based configuration** - Declarative engine setup via `engines.yaml`
- **Async-native** - Built on FastAPI with proper lifecycle management
- **Production-ready** - Health checks, readiness probes, and performance metrics

## Supported Engines

### Speech-to-Text (STT)

| Engine | Status | Description |
|--------|--------|-------------|
| Whisper (faster-whisper) | Supported | Local STT via faster-whisper with word-level timestamps |

### Text-to-Speech (TTS)

| Engine | Status | Description |
|--------|--------|-------------|
| (Coming soon) | Planned | - |

### Roadmap

- [ ] Google Cloud Speech-to-Text
- [ ] Azure Speech Services
- [ ] Coqui TTS
- [ ] OpenAI Whisper API

## Quick Start

### Installation

```bash
# Clone and install
git clone <repository-url>
cd voice-engine-api

# Install with uv (recommended)
uv sync

# Install Whisper engine dependencies
uv sync --group whisper
```

### Run Server

```bash
# Development mode with auto-reload
make dev-api

# Or directly with uvicorn
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Test It

```bash
# Health check
curl http://localhost:8000/api/v1/health

# List available engines
curl http://localhost:8000/api/v1/engines

# Transcribe audio (batch mode)
curl -X POST "http://localhost:8000/api/v1/stt/transcribe?engine=whisper" \
  -F "file=@audio.wav"
```

## API Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/ready` | Readiness check (engine status) |
| GET | `/api/v1/engines` | List available engines |
| POST | `/api/v1/stt/transcribe` | Batch transcription |
| POST | `/api/v1/stt/transcribe/stream` | SSE streaming transcription |
| WS | `/api/v1/stt/transcribe/ws` | WebSocket real-time transcription |

See [docs/api.md](docs/api.md) for full API documentation.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `DEBUG` | `false` | Debug mode |
| `ENGINE_CONFIG_PATH` | `engines.yaml` | Path to engine config |
| `MAX_AUDIO_SIZE_MB` | `25` | Max upload size |
| `CORS_ORIGINS` | `["*"]` | Allowed CORS origins |

### Engine Configuration (engines.yaml)

```yaml
stt:
  whisper:
    enabled: true
    engine_class: "app.engines.stt.whisper.engine.WhisperSTTEngine"
    config:
      model_name: "base"  # tiny, base, small, medium, large, large-v2, large-v3
      device: "cpu"       # cpu, cuda, mps
      compute_type: "int8"

tts: {}
```

See [docs/configuration.md](docs/configuration.md) for full configuration reference.

## Adding Custom Engines

The framework uses a plugin pattern for engines:

1. Create engine directory under `app/engines/stt/` or `app/engines/tts/`
2. Implement `BaseSTTEngine` or `BaseTTSEngine` interface
3. Create a config class extending `EngineConfig`
4. Register in `engines.yaml`

```python
from app.engines.base import BaseSTTEngine
from app.models.engine import EngineConfig, STTResponse

class MySTTEngine(BaseSTTEngine):
    async def _initialize(self) -> None:
        # Load models/resources
        pass

    async def _cleanup(self) -> None:
        # Cleanup resources
        pass

    async def transcribe(self, audio_data, language=None, **kwargs) -> STTResponse:
        # Implement transcription
        pass

    # ... implement remaining abstract methods
```

See [docs/custom-engines.md](docs/custom-engines.md) for the complete guide.

## Development

```bash
# Install with dev dependencies
make dev

# Run all tests
make test

# Run unit tests only
make test-unit

# Run tests with coverage
make test-cov

# Lint code
make lint

# Format code
make format

# Run all checks
make check
```

## Project Structure

```
voice-engine-api/
├── app/
│   ├── api/              # FastAPI application
│   │   ├── main.py       # App entry point
│   │   ├── routers/      # API endpoints
│   │   ├── middleware/   # CORS, logging, error handling
│   │   └── deps.py       # Dependencies
│   ├── engines/          # Engine implementations
│   │   ├── base.py       # Abstract base classes
│   │   └── stt/          # STT engines (whisper, etc.)
│   ├── models/           # Pydantic models
│   └── types/            # Type definitions
├── docs/                 # Documentation
├── tests/
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
├── engines.yaml          # Engine configuration
├── pyproject.toml        # Project dependencies
└── Makefile              # Development commands
```

## License

Apache 2.0 License
