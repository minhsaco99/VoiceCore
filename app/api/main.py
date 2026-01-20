import logging

from fastapi import FastAPI

from app.api.config import Settings
from app.api.lifespan import lifespan
from app.api.middleware.cors import configure_cors
from app.api.middleware.error_handler import ExceptionHandlerMiddleware
from app.api.middleware.logging import LoggingMiddleware
from app.api.routers import health, stt, tts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = Settings()

# Create FastAPI app with async lifespan
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Async-native voice processing API with STT and TTS engines",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS (must be first)
configure_cors(app, settings)

# Add middleware (reverse order of execution)
app.add_middleware(ExceptionHandlerMiddleware)
app.add_middleware(LoggingMiddleware)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(stt.router, prefix="/api/v1/stt", tags=["Speech-to-Text"])
app.include_router(tts.router, prefix="/api/v1/tts", tags=["Text-to-Speech"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.version,
        "docs": "/docs",
        "health": "/api/v1/health",
    }
