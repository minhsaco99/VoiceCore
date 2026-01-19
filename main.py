import uvicorn

from app.api.config import Settings

if __name__ == "__main__":
    settings = Settings()
    uvicorn.run(
        "app.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        loop="asyncio",  # Async-optimized
    )
