from functools import lru_cache
from src.vision import DamageClassifier
from src.cost import CostEstimator
from src.llm import ReportGenerator
from src.config import get_settings

@lru_cache()
def get_damage_classifier() -> DamageClassifier:
    """Get singleton instance of damage classifier."""
    settings = get_settings()
    return DamageClassifier(
        model_name=settings.VISION_MODEL_NAME,
        confidence_threshold=settings.CONFIDENCE_THRESHOLD,
        use_tta=settings.USE_TTA,
        checkpoint_path=settings.MODEL_CHECKPOINT_PATH
    )

@lru_cache()
def get_cost_estimator() -> CostEstimator:
    """Get singleton instance of cost estimator."""
    return CostEstimator()

@lru_cache()
def get_report_generator() -> ReportGenerator:
    """Get singleton instance of report generator."""
    return ReportGenerator()