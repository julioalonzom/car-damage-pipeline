# tests/test_llm/test_generator.py
import pytest
from unittest.mock import MagicMock, patch
from decimal import Decimal
from src.llm.generator import ReportGenerator
from src.llm.schemas import ReportRequest, DamageReport
from src.vision.schemas import CarPart

@pytest.fixture
def mock_openai():
    with patch('src.llm.generator.OpenAI') as mock:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"summary": "Test", "details": "Test", "repair_recommendations": "Test"}'
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response
        mock.return_value = mock_client
        yield mock

def test_report_generation(mock_openai):
    generator = ReportGenerator()
    request = ReportRequest(
        part_damages={CarPart.HEADLAMP: 0.8},
        cost_estimate=Decimal("100.00")
    )
    result = generator.generate(request)
    assert isinstance(result, DamageReport)
    assert result.summary == "Test"