# Contributing to VoiceCore

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/VoiceCore.git
   cd VoiceCore
   ```
3. Install dependencies:
   ```bash
   uv sync --group dev
   ```
4. Create a branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Branch Naming Convention

- `feature/` - New features (e.g., `feature/add-azure-tts`)
- `fix/` - Bug fixes (e.g., `fix/websocket-timeout`)
- `docs/` - Documentation changes (e.g., `docs/update-api-guide`)
- `refactor/` - Code refactoring (e.g., `refactor/engine-registry`)

## Development Setup

```bash
# Install all dev dependencies
make dev

# Install pre-commit hooks
pre-commit install

# Run tests
make test

# Run linting
make lint

# Format code
make format
```

## Code Style

- We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting
- Run `make format` before committing
- All code must pass `make lint`

## Pull Request Process

1. Ensure all tests pass: `make test`
2. Update documentation if needed
3. Add tests for new features
4. Follow the PR template when submitting

## Adding New Engines

See [docs/custom-engines.md](docs/custom-engines.md) for a guide on implementing new STT/TTS engines.

## Reporting Issues

- Use the issue templates for bugs and feature requests
- Include reproduction steps for bugs
- Check existing issues before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
