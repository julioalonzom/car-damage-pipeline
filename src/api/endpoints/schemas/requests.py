from pydantic import BaseModel
from typing import Dict
from src.vision.schemas import CarPart
from decimal import Decimal

class DamageAnalysisResponse(BaseModel):
    """Response model for damage analysis endpoint."""
    damage_analysis: Dict[CarPart, float]
    most_damaged_part: CarPart
    repair_costs: Dict[str, Decimal]
    total_cost: Decimal
    report: str
    confidence_score: float

class ErrorResponse(BaseModel):
    """Standard error response model."""
    detail: str
    error_code: str