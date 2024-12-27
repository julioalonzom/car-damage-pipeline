# tests/conftest.py
import pytest
from pathlib import Path
from PIL import Image
import torch
import os
import tempfile
import numpy as np
from unittest.mock import MagicMock, patch

from src.vision.model import DamageClassifier
from src.cost.estimator import CostEstimator
from src.llm.generator import ReportGenerator
from src.vision.schemas import CarPart
from src.config import Settings
from decimal import Decimal

# Constants
TEST_DATA_DIR = Path(__file__).parent / "data"
MOCK_RESULTS = {
    "damage_analysis": {
        "part_damages": {CarPart.HEADLAMP: 0.8},
        "most_damaged_part": CarPart.HEADLAMP,
        "max_confidence": 0.8
    },
    "repair_costs": {
        "total_cost": Decimal("661.91"),
        "breakdown": {CarPart.HEADLAMP: Decimal("661.91")},
        "currency": "USD",
        "confidence_score": 0.8
    }
}

@pytest.fixture(scope="session")
def test_data_dir():
    """Set up and return test data directory."""
    test_dir = TEST_DATA_DIR / "test_images"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir

@pytest.fixture
def test_image(test_data_dir):
    """Provide a test image. Creates a dummy one if no real image exists."""
    image_path = test_data_dir / "1.jpg"
    if image_path.exists():
        return Image.open(image_path)
    
    # Create dummy image if no test image exists
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(img_array)
    return image

@pytest.fixture
def mock_checkpoint():
    """Create a temporary mock checkpoint file."""
    state_dict = {
        'model_state_dict': {},
        'optimizer_state_dict': {},
        'loss': 0.5
    }
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(state_dict, f.name)
        yield f.name
    Path(f.name).unlink()  # cleanup

@pytest.fixture(autouse=True)
def mock_env_vars():
    """Mock environment variables for testing."""
    os.environ["OPENAI_API_KEY"] = "test-key"
    yield

@pytest.fixture
def test_settings():
    """Override settings for testing."""
    return Settings(
        OPENAI_API_KEY="test-key",
        MODEL_CHECKPOINT_PATH="tests/data/mock_checkpoint.pt",
        USE_TTA=False,  # Disable TTA for faster testing
        VISION_MODEL_NAME="google/vit-base-patch16-224",
        CONFIDENCE_THRESHOLD=0.3
    )

@pytest.fixture
def mock_damage_classifier(mock_checkpoint):
    """Provide mocked damage classifier for testing."""
    with patch('src.vision.model.DamageClassifier') as MockClass:
        instance = MockClass.return_value
        instance.classify.return_value = MOCK_RESULTS["damage_analysis"]
        yield instance

@pytest.fixture
def mock_cost_estimator():
    """Provide mocked cost estimator for testing."""
    with patch('src.cost.estimator.CostEstimator') as MockClass:
        instance = MockClass.return_value
        instance.estimate.return_value = MOCK_RESULTS["repair_costs"]
        yield instance

@pytest.fixture
def mock_report_generator():
    """Provide mocked report generator for testing."""
    with patch('src.llm.generator.ReportGenerator') as MockClass:
        instance = MockClass.return_value
        instance.generate.return_value = {
            "summary": "Test summary",
            "details": "Test details",
            "repair_recommendations": "Test recommendations"
        }
        yield instance

@pytest.fixture
def real_damage_classifier(mock_checkpoint):
    """Provide real (non-mocked) damage classifier for integration testing."""
    return DamageClassifier(
        model_name="google/vit-base-patch16-224",
        checkpoint_path=mock_checkpoint,
        use_tta=False  # Disable for faster testing
    )

@pytest.fixture
def real_cost_estimator():
    """Provide real (non-mocked) cost estimator for integration testing."""
    return CostEstimator()

@pytest.fixture
def real_report_generator():
    """Provide real (non-mocked) report generator for integration testing."""
    return ReportGenerator()

@pytest.fixture
def app_client():
    """Provide FastAPI test client."""
    from fastapi.testclient import TestClient
    from src.api.app import create_app
    
    app = create_app()
    return TestClient(app)

# For API testing
@pytest.fixture
def mock_dependencies(
    mock_damage_classifier,
    mock_cost_estimator,
    mock_report_generator
):
    """Group all mocked dependencies together."""
    return {
        "damage_classifier": mock_damage_classifier,
        "cost_estimator": mock_cost_estimator,
        "report_generator": mock_report_generator
    }