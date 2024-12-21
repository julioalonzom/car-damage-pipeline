import pytest
from decimal import Decimal
from src.llm.schemas import ReportRequest, DamageReport
from src.llm.generator import ReportGenerator

@pytest.fixture
def sample_request():
    return ReportRequest(
        damage_predictions={
            "scratch": 0.9,
            "dent": 0.7
        },
        cost_estimate=Decimal("750.00"),
    )

def test_report_generation(generator, sample_request):
    report = generator.generate(sample_request)
    assert isinstance(report, DamageReport)
    assert report.summary
    assert report.details
    assert report.repair_recommendation
    
def test_damage_formatting(generator):
    damages = {"scratch": 0.9, "dent": 0.7}
    formatted = generator._format_damages(damages)
    assert "Scratch: 90.0% confidence" in formatted
    assert "Dent: 70.0% confidence" in formatted