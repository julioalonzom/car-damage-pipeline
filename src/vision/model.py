import torch
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from typing import Dict, List, Optional

from src.vision.preprocessing import ImagePreprocessor
from src.vision.schemas import DamageAnalysis, DamageType
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DamageClassifier:
    """Classifies car damage from images."""

    def __init__(
        self,
        model_name: str,
        confidence_threshold: float = 0.3,
        device: Optional[str] = None
    ):
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Initializing DamageClassifier with model: {model_name}")
        logger.info(f"Using device: {self.device}")

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name).to(self.device)
        self.model.to(self.device)
        self.preprocessor = ImagePreprocessor()

        # Map model outputs to damage types
        self.id2label = {
            0: DamageType.SCRATCH,
            1: DamageType.DENT,
            2: DamageType.BROKEN_BUMPER,
            3: DamageType.GLASS_SHATTER,
            4: DamageType.PAINT_DAMAGE
        }

    @torch.no_grad()
    def classify(self, image: Image.Image) -> DamageAnalysis:
        """
        Classify car damage from an image.
        
        Args:
            image: PIL Image to classify

        Returns:
            DamageAnalysis containing predictions and primary damage type
        """
        try:
            # Preprocess image
            inputs = self.preprocessor.preprocess(image)
            inputs = inputs.to(self.device)

            # Get model predictions
            outputs = self.model(imputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Convert to dictionary of predictions
            predictions = {
                self.id2label[i].value: float(prob)
                for i, prob in enumerate(probs[0])
                if float(prob > self.confidence_threshold)
            }

            # Get primary damage type
            primary_idx = torch.argmax(probs[0]).item()
            primary_damage = self.id2label[primary_idx]

            return DamageAnalysis(
                predictions=predictions,
                primary_damage=primary_damage
            )

        except Exception as e:
            logger.error(f"Error during damage classification: {str(e)}")
            raise RuntimeError(f"Failed to classify image: {str(e)}")