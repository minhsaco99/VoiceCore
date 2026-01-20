from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    app_name: str = "Voice Engine API"
    version: str = "1.0.0"
    debug: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS
    cors_origins: list[str] = ["*"]

    # Engine Configuration
    engine_config_path: str = "engines.yaml"

    # Limits
    max_audio_size_mb: int = 25
    request_timeout_seconds: int = 300

    class Config:
        env_file = ".env"
