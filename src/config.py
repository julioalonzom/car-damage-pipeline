# src/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

class Settings(BaseSettings):
    """Application settings"""

    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Car Damage Assistant"

    # OpenAI
    OPENAI_API_KEY: str

    # Model Settings
    MODEL_CACHE_DIR: str = str(PROJECT_ROOT / "model_cache")
    VISION_MODEL_NAME: str = "google/vit-base-patch16-224"
    MODEL_CHECKPOINT_PATH: str = str(PROJECT_ROOT / "checkpoints" / "best_model_refined.pt")
    BASE_MODEL_NAME: str = "google/vit-base-patch16-224"

    # Model inference settings
    USE_TTA: bool = True
    CONFIDENCE_THRESHOLD: float = 0.3

    # Logging
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True
    )

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()