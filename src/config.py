from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings"""

    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Car Damage Assisstant"

    # OpenAI
    OPENAI_API_KEY: str

    # Model paths
    MODEL_CHECKPOINT_PATH: str = "checkpoints/best_model_refined.pt"
    BASE_MODEL_NAME: str = "google/vit-base-patch16-224"

    # Model inference settings
    USE_TTA: bool = True
    CONFIDENCE_THRESHOLD: float = 0.3

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()