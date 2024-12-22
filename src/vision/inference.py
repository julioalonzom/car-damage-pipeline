import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
import torchvision.transforms.functional as TF
from PIL import Image

from src.vision.schemas import DamageAnalysis, CarPart
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TTAInference:
    """Handles test-time augmentation and prediction aggregation"""
    
    @staticmethod
    def horizontal_flip(image: torch.Tensor) -> torch.Tensor:
        """Flip image horizontally"""
        return TF.hflip(image)
    
    @staticmethod
    def rotate(image: torch.Tensor, angle: float) -> torch.Tensor:
        """Rotate image by angle degrees"""
        return TF.rotate(image, angle)
    
    @staticmethod
    def get_tta_transforms(image: torch.Tensor) -> List[torch.Tensor]:
        """Apply TTA transforms to image"""
        transforms = [
            image,  # Original
            TTAInference.horizontal_flip(image),  # Flipped
            TTAInference.rotate(image, 5),  # Slight clockwise
            TTAInference.rotate(image, -5),  # Slight counter-clockwise
        ]
        return transforms
    
    @staticmethod
    def aggregate_predictions(predictions: List[torch.Tensor]) -> torch.Tensor:
        """Aggregate predictions from multiple augmentations"""
        # Stack and average predictions
        stacked = torch.stack(predictions)
        return torch.mean(stacked, dim=0)