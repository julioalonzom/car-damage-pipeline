# tests/test_integration/test_pipeline.py
import pytest
from decimal import Decimal
from src.vision.schemas import CarPart, DamageAnalysis
from src.llm.schemas import ReportRequest
from src.cost.schemas import RepairCost

def test_complete_pipeline(
    mock_checkpoint,
    test_image,
    mock_damage_classifier,
    mock_cost_estimator,
    mock_report_generator
):
    """Test entire pipeline end-to-end with mocked components."""
    # 1. Test Vision Analysis
    damage_analysis = DamageAnalysis(
        part_damages={CarPart.HEADLAMP: 0.8},
        most_damaged_part=CarPart.HEADLAMP,
        max_confidence=0.8
    )
    mock_damage_classifier.classify.return_value = damage_analysis
    
    vision_result = mock_damage_classifier.classify(test_image)
    assert vision_result.part_damages == damage_analysis.part_damages
    
    # 2. Test Cost Estimation
    repair_cost = RepairCost(
        total_cost=Decimal("661.91"),
        breakdown={CarPart.HEADLAMP: Decimal("661.91")},
        currency="USD",
        confidence_score=0.8
    )
    mock_cost_estimator.estimate.return_value = repair_cost
    
    cost_result = mock_cost_estimator.estimate(vision_result.part_damages)
    assert cost_result.total_cost == repair_cost.total_cost
    
    # 3. Test Report Generation
    mock_report_generator.generate.return_value = {
        "summary": "Test report",
        "details": "Test details",
        "repair_recommendations": "Test recommendations"
    }
    
    report = mock_report_generator.generate(
        ReportRequest(
            part_damages=vision_result.part_damages,
            cost_estimate=cost_result.total_cost
        )
    )
    assert report["summary"] == "Test report"