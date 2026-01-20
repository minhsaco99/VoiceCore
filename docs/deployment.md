# Deployment Guide

This guide explains how to deploy the Voice Engine API using Docker and Docker Compose.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed
- [Docker Compose](https://docs.docker.com/compose/install/) installed (usually included with Docker Desktop/Engine)

## Quick Start (Whisper Engine)

The repository includes a pre-configured setup for the Whisper STT engine.

1. **Build and Start**

   ```bash
   docker-compose up -d --build
   ```

   This will:
   - Build the image using `docker/Dockerfile.whisper`
   - Start the service on port `8000`
   - Mount a volume `huggingface_cache` to persist downloaded models

2. **Check Logs**

   ```bash
   docker-compose logs -f
   ```

3. **Verify API**

   ```bash
   curl http://localhost:8000/api/v1/health
   # {"status":"ok",...}
   ```

4. **Stop**

   ```bash
   docker-compose down
   ```

### Volume Persistence

The `huggingface_cache` volume ensures that large model files (downloaded by faster-whisper) are persisted between container restarts. To clear the cache:

```bash
docker volume rm voice-engine-api_huggingface_cache
```

## Custom Engine Deployment

Since the Voice Engine API is modular, you might want to package it with different dependencies (e.g., specific TTS engines, different STT libraries).

### 1. Create a Dockerfile

Use `docker/Dockerfile.template` as a base.

```dockerfile
# docker/Dockerfile.myengine

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./

# Install your specific dependency group defined in pyproject.toml
RUN uv sync --frozen --no-dev --group my-engine-group

FROM python:3.12-slim AS runtime
# Install system deps...
# ... (rest of the standard setup)
```

### 2. Update docker-compose.yml

Point `dockerfile` to your new file:

```yaml
services:
  voice-engine:
    build:
      context: .
      dockerfile: docker/Dockerfile.myengine
    # ...
```

## Production Considerations

- **GPU Support**: To use GPU acceleration (e.g., for Whisper), ensure you use the NVIDIA Container Runtime and appropriate base image (e.g., `nvidia/cuda:12.x.x-cudnn8-runtime-ubuntu22.04`). You will need to install Python and necessary libs in that base image, or copy the venv.
- **Security**: The default Dockerfile runs as root. for high-security environments, consider creating a non-root user.
- **Env Vars**: Inject sensitive configuration (API keys) via environment variables or `.env` file, do not commit them to the image.
