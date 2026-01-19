.PHONY: help test test-cov test-watch test-unit test-integration clean install format lint check dev run

# Default target - show help
help:
	@echo "Voice Engine API - Available Make Commands"
	@echo "=========================================="
	@echo ""
	@echo "Testing:"
	@echo "  make test          - Run all tests"
	@echo "  make test-cov      - Run tests with coverage report"
	@echo "  make test-unit     - Run unit tests only"
	@echo "  make test-watch    - Run tests in watch mode (requires pytest-watch)"
	@echo ""
	@echo "Development:"
	@echo "  make install       - Install all dependencies"
	@echo "  make dev           - Install dev dependencies"
	@echo "  make run           - Run the application"
	@echo "  make dev-api       - Run FastAPI dev server with auto-reload"
	@echo "  make run-api       - Run FastAPI production server"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          - Run linting (ruff)"
	@echo "  make format        - Format code (ruff format)"
	@echo "  make check         - Run lint + format check"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         - Remove cache and build artifacts"
	@echo "  make clean-all     - Remove cache, build artifacts, and venv"
	@echo ""

# Testing
test:
	@echo "Running all tests..."
	uv run pytest tests/

test-cov:
	@echo "Running tests with coverage..."
	uv run pytest tests/unit/ --cov=app --cov-report=term-missing --cov-report=html -v

test-unit:
	@echo "Running unit tests..."
	uv run pytest tests/unit/ -v

test-integration:
	@echo "Running integration tests..."
	uv run pytest tests/integration/ -v

test-watch:
	@echo "Running tests in watch mode..."
	uv run ptw tests/

# Installation
install:
	@echo "Installing dependencies..."
	uv sync
	@echo "Installing pre-commit hooks..."
	uv run pre-commit install

dev:
	@echo "Installing dev dependencies..."
	uv sync --dev
	@echo "Installing pre-commit hooks..."
	uv run pre-commit install

# Code Quality
lint:
	@echo "Running linter..."
	uv run ruff check app/ tests/

format:
	@echo "Formatting code..."
	uv run ruff format app/ tests/

format-check:
	@echo "Checking code format..."
	uv run ruff format --check app/ tests/

check: lint format-check
	@echo "All checks passed!"

# Run application
run:
	@echo "Starting application..."
	uv run python main.py

dev-api:
	@echo "Starting FastAPI dev server..."
	uv run uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

run-api:
	@echo "Starting FastAPI production server..."
	uv run python main.py

# Cleanup
clean:
	@echo "Cleaning up cache and build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ 2>/dev/null || true
	rm -rf dist/ 2>/dev/null || true
	rm -rf build/ 2>/dev/null || true
	@echo "Cleanup complete!"

clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf .venv/
	@echo "Full cleanup complete!"
