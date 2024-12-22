from pydantic import BaseModel
from typing import Dict, Optional
from decimal import Decimal
from src.vision.schemas import CarPart

class DamageReport(BaseModel):
    """Structured damage report from LLM."""
    summary: str
    details: str
    repair_recommendation: str
    estimated_time: Optional[str] = None
    severity_assessment: Optional[str] = None

class ReportRequest(BaseModel):
    part_damages: Dict[CarPart, float]  # Updated to use part damages
    cost_estimate: Decimal
    image_location: Optional[str] = None