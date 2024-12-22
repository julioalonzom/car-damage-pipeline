# src/vision/preprocessing.py
from PIL import Image
import torch
from torchvision import transforms
from typing import Tuple

class ImagePreprocessor:
    """Handles imaage preprocessing for our vision model."""

    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess an image for model inference.
        
        Args:
            Image: PIL Image to preprocess
            
        Returns:
            Preprocessed image tensor
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        return self.transform(image).unsqueeze(0) # Add batch dimension
