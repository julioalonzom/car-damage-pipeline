import pytest
from PIL import Image
import numpy as np
from src.vision.model import DamageClassifier
from src.vision.schemas import DamageType

@pytest.fixture
def dummy_image():
    # Create a dummy RGB image
    return Image.fromarray(
        (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    )

@pytest.fixture
def classifier():
    return DamageClassifier(
        model_name="google/vit-base-patch16-224",
        confidence_threshold=0.3
    )

def test_classifier_prediction(classifier, dummy_image):
    result = classifier.classify(dummy_image)
    assert isinstance(result.predictions, dict)
    assert isinstance(result.primary_damage, DamageType)
    assert sum(result.predictions.values()) > 0