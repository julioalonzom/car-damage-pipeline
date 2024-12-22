# src/cost/schemas.py
from pydantic import BaseModel, Field
from typing import Dict, Optional
from decimal import Decimal
from src.vision.schemas import CarPart

class RepairCost(BaseModel):
    """
    Represents the estimated cost of repairs.
    Using Decimal for monetary values
    """
    total_cost: Decimal = Field(..., decimal_places=2)
    breakdown: Dict[CarPart, Decimal]
    currency: str = "USD"
    confidence_score: Optional[float] = Field(None, ge=0, le=1)

    class Config:
        """Force all decimals to have 2 decimal places"""
        json_encoders = {
            Decimal: lambda v: str(v.quantize(Decimal("0.01")))
        }