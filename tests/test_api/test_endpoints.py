# tests/test_api/test_endpoints.py
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
from unittest.mock import MagicMock
from decimal import Decimal

from src.api.app import create_app
from src.vision.schemas import CarPart, DamageAnalysis
from src.cost.schemas import RepairCost
from src.llm.schemas import DamageReport

# Test Data
MOCK_DAMAGE_ANALYSIS = DamageAnalysis(
    part_damages={CarPart.HEADLAMP: 0.8},
    most_damaged_part=CarPart.HEADLAMP,
    max_confidence=0.8
)

MOCK_REPAIR_COST = RepairCost(
    total_cost=Decimal("661.91"),
    breakdown={CarPart.HEADLAMP: Decimal("661.91")},
    currency="USD",
    confidence_score=0.8
)

MOCK_REPORT = DamageReport(
    summary="Test summary",
    details="Test details",
    repair_recommendations="Test recommendations"
)

@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)

@pytest.fixture
def mock_openai():
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(
            message=MagicMock(
                content='{"summary": "Test summary", "details": "Test details", "repair_recommendations": "Test recommendations"}'
            )
        )
    ]
    mock_client.chat.completions.create.return_value = mock_completion
    return mock_client

@pytest.fixture
def mock_dependencies(monkeypatch):
    """Mock all dependencies for API testing."""
    from src.api.dependencies import set_test_instances, reset_dependencies
    
    # Create mock instances
    mock_classifier = MagicMock()
    mock_classifier.classify.return_value = MOCK_DAMAGE_ANALYSIS
    
    mock_estimator = MagicMock()
    mock_estimator.estimate.return_value = MOCK_REPAIR_COST
    
    mock_generator = MagicMock()
    mock_generator.generate.return_value = MOCK_REPORT

    # Set test instances
    set_test_instances(
        classifier=mock_classifier,
        estimator=mock_estimator,
        generator=mock_generator
    )
    
    yield {
        "classifier": mock_classifier,
        "estimator": mock_estimator,
        "generator": mock_generator
    }
    
    # Cleanup
    reset_dependencies()

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_analyze_damage(client, test_image, mock_dependencies):
    # Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    test_image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    response = client.post(
        "/api/v1/damage/analyze",
        files={"file": ("test.jpg", img_byte_arr, "image/jpeg")}
    )

    print("Response status:", response.status_code)
    print("Response content:", response.json())  # Add this debug line
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert "damage_analysis" in data
    assert "total_cost" in data
    assert "report" in data
    
    # Verify mock calls
    mock_deps = mock_dependencies
    assert mock_deps["classifier"].classify.called
    assert mock_deps["estimator"].estimate.called
    assert mock_deps["generator"].generate.called

def test_invalid_file_type(client, mock_dependencies):
    response = client.post(
        "/api/v1/damage/analyze",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400
    assert "must be an image" in response.json()["detail"]

def test_missing_file(client, mock_dependencies):
    response = client.post("/api/v1/damage/analyze")
    assert response.status_code == 422  # FastAPI validation error

def test_internal_error(client, mock_dependencies):
    # Make classifier raise an error
    mock_dependencies["classifier"].classify.side_effect = Exception("Test error")
    
    img_byte_arr = io.BytesIO()
    Image.new('RGB', (100, 100)).save(img_byte_arr, format='JPEG')
    
    response = client.post(
        "/api/v1/damage/analyze",
        files={"file": ("test.jpg", img_byte_arr.getvalue(), "image/jpeg")}
    )
    assert response.status_code == 500
    assert "error" in response.json()["detail"].lower()