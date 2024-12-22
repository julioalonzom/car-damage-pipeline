# src/vision/model.py
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from typing import Dict, List, Optional

from src.vision.preprocessing import ImagePreprocessor
from src.vision.schemas import DamageAnalysis, CarPart
from src.vision.inference import TTAInference
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DamageClassifier:
    """
    Classifies car damage from images using a fine-tuned ViT model.
    Supports test-time augmentation for improved robustness.
    """

    def __init__(
        self,
        model_name: str,
        confidence_threshold: float = 0.3,
        use_tta: bool = True,
        device: Optional[str] = None,
        checkpoint_path: Optional[str] = None
    ):
        """
        Initialize the damage classifier.

        Args:
            model_name: Base model name or path
            confidence_threshold: Minimum confidence for predictions
            use_tta: Whether to use test-time augmentation
            device: Device to run model on ('cuda' or 'cpu')
            checkpoint_path: Path to fine-tuned model checkpoint
        """
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_tta = use_tta

        logger.info(f"Initializing DamageClassifier with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Test-time augmentation: {'enabled' if use_tta else 'disabled'}")

        # Initialize preprocessing components
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.preprocessor = ImagePreprocessor()

        # Initialize model
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=5,  # Number of car parts we can detect
            problem_type="multi_label_classification"
        )

        # Load fine-tuned weights if provided
        if checkpoint_path:
            logger.info(f"Loading checkpoint from: {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded checkpoint with validation loss: {checkpoint['loss']:.4f}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {str(e)}")
                raise RuntimeError(f"Checkpoint loading failed: {str(e)}")

        self.model.to(self.device)
        self.model.eval()

        # Map model outputs to car parts
        self.id2label = {
            0: CarPart.HEADLAMP,
            1: CarPart.REAR_BUMPER,
            2: CarPart.DOOR,
            3: CarPart.HOOD,
            4: CarPart.FRONT_BUMPER
        }

    @torch.no_grad()
    def classify(self, image: Image.Image) -> DamageAnalysis:
        """
        Classify car damage from an image.
        
        Args:
            image: PIL Image to classify

        Returns:
            DamageAnalysis containing predictions and confidence scores
        """
        try:
            # Preprocess image
            inputs = self.preprocessor.preprocess(image)
            inputs = inputs.to(self.device)
            
            if self.use_tta:
                # Get predictions with test-time augmentation
                tta_inputs = TTAInference.get_tta_transforms(inputs)
                predictions = []
                
                for tta_input in tta_inputs:
                    outputs = self.model(tta_input)
                    probs = F.softmax(outputs.logits, dim=-1)
                    predictions.append(probs)
                
                # Aggregate predictions from all augmentations
                final_probs = TTAInference.aggregate_predictions(predictions)
            else:
                # Single forward pass without TTA
                outputs = self.model(inputs)
                final_probs = F.softmax(outputs.logits, dim=-1)

            # Convert to dictionary of predictions with confidence scores
            part_damages = {}
            for idx, prob in enumerate(final_probs[0]):
                confidence = float(prob)
                if confidence > self.confidence_threshold:
                    part = self.id2label[idx]
                    part_damages[part] = confidence

            # Find most damaged part
            most_damaged_idx = torch.argmax(final_probs[0]).item()
            most_damaged_part = self.id2label[most_damaged_idx]
            max_confidence = float(final_probs[0][most_damaged_idx])

            return DamageAnalysis(
                part_damages=part_damages,
                most_damaged_part=most_damaged_part,
                max_confidence=max_confidence
            )

        except Exception as e:
            logger.error(f"Error during damage classification: {str(e)}")
            raise RuntimeError(f"Failed to classify image: {str(e)}")

    def __repr__(self):
        return (f"DamageClassifier(threshold={self.confidence_threshold}, "
                f"device={self.device}, tta={'enabled' if self.use_tta else 'disabled'})")