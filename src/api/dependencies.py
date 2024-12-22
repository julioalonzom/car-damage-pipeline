# src/api/dependencies.py
from functools import lru_cache
from src.vision import DamageClassifier
from src.config import get_settings

@lru_cache()
def get_damage_classifier() -> DamageClassifier:
    settings = get_settings()
    return DamageClassifier(
        model_name=settings.BASE_MODEL_NAME,
        confidence_threshold=settings.CONFIDENCE_THRESHOLD,
        use_tta=settings.USE_TTA,
        checkpoint_path=settings.MODEL_CHECKPOINT_PATH
    )