from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings"""

    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Car Damage Assisstant"

    # OpenAI
    OPENAI_API_KEY: str

    # Model Settings
    VISION_MODEL_NAME: str = "google/vit-base-patch16-224"

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()