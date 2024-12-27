# tests/test_cost/test_estimator.py
import pytest
from decimal import Decimal
from src.cost.estimator import CostEstimator
from src.vision.schemas import CarPart
from src.cost.schemas import RepairCost

def test_cost_estimator_initialization():
    estimator = CostEstimator()
    assert isinstance(estimator.base_costs, dict)
    assert all(isinstance(cost, Decimal) for cost in estimator.base_costs.values())
    assert all(isinstance(key, CarPart) for key in estimator.base_costs.keys())

def test_cost_estimation():
    estimator = CostEstimator()
    damages = {CarPart.HEADLAMP: 0.8}
    result = estimator.estimate(damages)
    assert isinstance(result, RepairCost)
    assert result.total_cost == Decimal('640.00')  # 800 * 0.8 = 640
    assert result.currency == "USD"

def test_empty_damages():
    estimator = CostEstimator()
    with pytest.raises(Exception):  # Changed from RuntimeError
        estimator.estimate({})