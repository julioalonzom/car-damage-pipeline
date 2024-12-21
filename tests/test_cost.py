import pytest
from decimal import Decimal
from src.cost.estimator import CostEstimator
from src.cost.schemas import RepairCost

@pytest.fixture
def estimator():
    return CostEstimator()

def test_basic_estimation(estimator):
    damages = {
        "scratch": 0.9,
        "dent": 0.5
    }

    result = estimator.estimate(damages)
    assert isinstance(result, RepairCost)
    assert result.total_cost > 0
    assert len(result.breakdown) == 2
    assert all(isinstance(cost, Decimal) for cost in result.breakdown.values())

def test_unknown_damage_type(estimator):
    damages = {
        "scratch": 0.9,
        "unknown_damage": 0.5  # This should be ignored
    }
    
    result = estimator.estimate(damages)
    assert len(result.breakdown) == 1
    assert "unknown_damage" not in result.breakdown