# tests/test_vision/test_inference.py
import torch
from src.vision.inference import TTAInference

def test_tta_transforms():
    """Test TTA transform functions."""
    # Create dummy tensor
    image = torch.randn(1, 3, 224, 224)
    
    # Test horizontal flip
    flipped = TTAInference.horizontal_flip(image)
    assert flipped.shape == image.shape
    
    # Test rotation
    rotated = TTAInference.rotate(image, 5)
    assert rotated.shape == image.shape
    
    # Test all transforms
    transforms = TTAInference.get_tta_transforms(image)
    assert len(transforms) == 4  # Original + flipped + 2 rotations