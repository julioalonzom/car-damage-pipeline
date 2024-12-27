# src/api/dependencies.py
from functools import lru_cache
from openai import OpenAI
from src.vision import DamageClassifier
from src.cost import CostEstimator
from src.llm import ReportGenerator
from src.config import get_settings

# Store instances for testing
_classifier_instance = None
_estimator_instance = None
_generator_instance = None

def reset_dependencies():
    """Reset all dependencies (useful for testing)."""
    global _classifier_instance, _estimator_instance, _generator_instance
    _classifier_instance = None
    _estimator_instance = None
    _generator_instance = None

@lru_cache()
def get_damage_classifier() -> DamageClassifier:
    """Get singleton instance of damage classifier."""
    global _classifier_instance
    if _classifier_instance is not None:
        return _classifier_instance
        
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
    global _estimator_instance
    if _estimator_instance is not None:
        return _estimator_instance
    return CostEstimator()

@lru_cache()
def get_report_generator() -> ReportGenerator:
    """Get singleton instance of report generator."""
    global _generator_instance
    if _generator_instance is not None:
        return _generator_instance
        
    settings = get_settings()
    return ReportGenerator(client=OpenAI(api_key=settings.OPENAI_API_KEY))

def set_test_instances(classifier=None, estimator=None, generator=None):
    """Set test instances for dependencies."""
    global _classifier_instance, _estimator_instance, _generator_instance
    if classifier is not None:
        _classifier_instance = classifier
    if estimator is not None:
        _estimator_instance = estimator
    if generator is not None:
        _generator_instance = generator