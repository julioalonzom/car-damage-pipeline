from pydantic import BaseModel
from typing import Dict
from decimal import Decimal
from src.vision.schemas import CarPart

class DamageAnalysisResponse(BaseModel):
    """Response model for damage analysis endpoint."""
    damage_analysis: Dict[CarPart, float]
    most_damaged_part: CarPart
    repair_costs: Dict[CarPart, Decimal]
    total_cost: Decimal
    report: str
    confidence_score: float

class ErrorResponse(BaseModel):
    """Standard error response model."""
    detail: str