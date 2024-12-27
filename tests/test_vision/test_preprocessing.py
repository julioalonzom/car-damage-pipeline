# tests/test_vision/test_preprocessing.py
import pytest
import torch
from PIL import Image
import numpy as np
from src.vision.preprocessing import ImagePreprocessor

def test_preprocessor_initialization():
    preprocessor = ImagePreprocessor()
    assert preprocessor.target_size == (224, 224)

def test_image_preprocessing(test_image):
    preprocessor = ImagePreprocessor()
    result = preprocessor.preprocess(test_image)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] == 1  # batch dimension
    assert result.shape[1] == 3  # RGB channels