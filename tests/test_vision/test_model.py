# tests/test_vision/test_model.py
import pytest
import torch
import tempfile
import json
from pathlib import Path
from src.vision import DamageClassifier

@pytest.fixture
def mock_checkpoint():
    """Create a temporary mock checkpoint file."""
    state_dict = {
        'model_state_dict': {
            'classifier.weight': torch.randn(5, 768),
            'classifier.bias': torch.randn(5)
        },
        'optimizer_state_dict': {},
        'loss': 0.5
    }
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(state_dict, f.name)
        yield f.name
    Path(f.name).unlink()

def test_damage_classification(mock_checkpoint):
    """Test model classification with mocked checkpoint."""
    classifier = DamageClassifier(
        model_name="google/vit-base-patch16-224",
        checkpoint_path=mock_checkpoint,  # Use the mock checkpoint
        confidence_threshold=0.3
    )
    
    # Create a simple test image
    test_image = Image.new('RGB', (224, 224))
    result = classifier.classify(test_image)
    
    assert hasattr(result, 'part_damages')
    assert hasattr(result, 'most_damaged_part')
    assert isinstance(result.part_damages, dict)