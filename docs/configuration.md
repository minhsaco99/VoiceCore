# Configuration Reference

Complete configuration reference for Voice Engine API.

## Environment Variables

Configure the application using environment variables or a `.env` file.

### Application Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `APP_NAME` | string | `"Voice Engine API"` | Application name |
| `VERSION` | string | `"1.0.0"` | Application version |
| `DEBUG` | bool | `false` | Enable debug mode |

### Server Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `HOST` | string | `"0.0.0.0"` | Server bind address |
| `PORT` | int | `8000` | Server port |

### CORS Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CORS_ORIGINS` | list | `["*"]` | Allowed CORS origins (JSON array) |

### Engine Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENGINE_CONFIG_PATH` | string | `"engines.yaml"` | Path to engine configuration file |

### Limits

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MAX_AUDIO_SIZE_MB` | int | `25` | Maximum audio upload size in MB |
| `REQUEST_TIMEOUT_SECONDS` | int | `300` | Request timeout in seconds |

### Example .env File

```bash
# App Configuration
APP_NAME="Voice Engine API"
VERSION="1.0.0"
DEBUG=false

# Server
HOST=0.0.0.0
PORT=8000

# CORS
CORS_ORIGINS=["http://localhost:3000", "https://myapp.com"]

# Engine Configuration
ENGINE_CONFIG_PATH=engines.yaml

# Limits
MAX_AUDIO_SIZE_MB=25
REQUEST_TIMEOUT_SECONDS=300

# Engine-specific environment variables (referenced in engines.yaml)
# GOOGLE_API_KEY=your_google_api_key_here
```

---

## Engine Configuration (engines.yaml)

The `engines.yaml` file defines which STT and TTS engines to load at startup.

### Structure

```yaml
stt:
  <engine_name>:
    enabled: true|false
    engine_class: "fully.qualified.class.Path"
    config:
      # Engine-specific configuration

tts:
  <engine_name>:
    enabled: true|false
    engine_class: "fully.qualified.class.Path"
    config:
      # Engine-specific configuration
```

### Full Example

```yaml
# Engine Configuration
# Define which STT/TTS engines to load at startup

stt:
  whisper:
    enabled: true
    engine_class: "app.engines.stt.whisper.engine.WhisperSTTEngine"
    config:
      model_name: "base"
      device: "cpu"
      compute_type: "int8"
      timeout_seconds: 300

  # Example: Google STT (disabled by default)
  # google:
  #   enabled: false
  #   engine_class: "app.engines.stt.google.engine.GoogleSTTEngine"
  #   config:
  #     api_key: "${GOOGLE_API_KEY}"  # Read from env var
  #     language: "en-US"

tts: {}
  # Example: Coqui TTS (disabled)
  # coqui:
  #   enabled: false
  #   engine_class: "app.engines.tts.coqui.engine.CoquiTTSEngine"
  #   config:
  #     model_name: "tts_models/en/ljspeech/tacotron2-DDC"
  #     device: "cpu"
```

### Environment Variable Substitution

You can reference environment variables in `engines.yaml` using `${VAR_NAME}` syntax:

```yaml
stt:
  google:
    enabled: true
    engine_class: "app.engines.stt.google.engine.GoogleSTTEngine"
    config:
      api_key: "${GOOGLE_API_KEY}"
```

---

## Engine-Specific Configuration

### Base Engine Configuration

All engines inherit from `EngineConfig` with these common fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | string | **required** | Model name/path to use |
| `device` | string | `"cpu"` | Device: `cpu`, `cuda`, `mps` |
| `max_workers` | int | `1` | Maximum parallel workers |
| `timeout_seconds` | int | `300` | Processing timeout |

### Whisper (faster-whisper)

Configuration class: `WhisperConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | string | **required** | Whisper model: `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3` |
| `device` | string | `"cpu"` | Device: `cpu`, `cuda`, `mps` |
| `compute_type` | string | `"int8"` | Compute type: `int8`, `float16`, `float32` |
| `beam_size` | int | `5` | Beam search size |
| `language` | string | `null` | Default language hint (e.g., `"en"`) |
| `temperature` | float | `0.0` | Sampling temperature (0.0 = greedy) |
| `vad_filter` | bool | `false` | Enable Voice Activity Detection filter |
| `condition_on_previous_text` | bool | `true` | Condition on previous text for context |
| `compression_ratio_threshold` | float | `2.4` | Compression ratio threshold |
| `log_prob_threshold` | float | `-1.0` | Log probability threshold |
| `no_speech_threshold` | float | `0.6` | No speech probability threshold |
| `timeout_seconds` | int | `300` | Processing timeout |

**Example:**

```yaml
stt:
  whisper:
    enabled: true
    engine_class: "app.engines.stt.whisper.engine.WhisperSTTEngine"
    config:
      model_name: "large-v3"
      device: "cuda"
      compute_type: "float16"
      beam_size: 5
      language: "en"
      vad_filter: true
      temperature: 0.0
```

### Model Size Guide

| Model | Parameters | Relative Speed | VRAM (GPU) | Description |
|-------|------------|----------------|------------|-------------|
| `tiny` | 39M | ~32x | ~1 GB | Fastest, lowest accuracy |
| `base` | 74M | ~16x | ~1 GB | Good balance for short audio |
| `small` | 244M | ~6x | ~2 GB | Good accuracy |
| `medium` | 769M | ~2x | ~5 GB | High accuracy |
| `large` | 1550M | 1x | ~10 GB | Best accuracy (original) |
| `large-v2` | 1550M | 1x | ~10 GB | Improved accuracy |
| `large-v3` | 1550M | 1x | ~10 GB | Latest, best accuracy |

### Compute Type Performance

| Type | Description | Speed | Accuracy |
|------|-------------|-------|----------|
| `int8` | 8-bit integer quantization | Fastest | Slightly reduced |
| `float16` | 16-bit floating point | Fast | Full (GPU recommended) |
| `float32` | 32-bit floating point | Slowest | Full |

---

## Configuration Best Practices

### Development

```yaml
stt:
  whisper:
    enabled: true
    engine_class: "app.engines.stt.whisper.engine.WhisperSTTEngine"
    config:
      model_name: "base"  # Fast for development
      device: "cpu"
      compute_type: "int8"
```

### Production (CPU)

```yaml
stt:
  whisper:
    enabled: true
    engine_class: "app.engines.stt.whisper.engine.WhisperSTTEngine"
    config:
      model_name: "small"  # Balance of speed and accuracy
      device: "cpu"
      compute_type: "int8"
      vad_filter: true  # Reduce processing of silence
      timeout_seconds: 600
```

### Production (GPU)

```yaml
stt:
  whisper:
    enabled: true
    engine_class: "app.engines.stt.whisper.engine.WhisperSTTEngine"
    config:
      model_name: "large-v3"  # Best accuracy
      device: "cuda"
      compute_type: "float16"
      vad_filter: true
      beam_size: 5
```

---

## Multiple Engines

You can configure multiple engines simultaneously. Clients select which engine to use via the `engine` query parameter.

```yaml
stt:
  whisper-fast:
    enabled: true
    engine_class: "app.engines.stt.whisper.engine.WhisperSTTEngine"
    config:
      model_name: "tiny"
      device: "cpu"
      compute_type: "int8"

  whisper-accurate:
    enabled: true
    engine_class: "app.engines.stt.whisper.engine.WhisperSTTEngine"
    config:
      model_name: "large-v3"
      device: "cuda"
      compute_type: "float16"
```

**Usage:**

```bash
# Use fast engine
curl -X POST "http://localhost:8000/api/v1/stt/transcribe?engine=whisper-fast" -F "file=@audio.wav"

# Use accurate engine
curl -X POST "http://localhost:8000/api/v1/stt/transcribe?engine=whisper-accurate" -F "file=@audio.wav"
```
