from pydantic import BaseModel
from typing import Dict, Optional
from decimal import Decimal

class DamageReport(BaseModel):
    """Structured damage report from LLM."""
    summary: str
    details: str
    repair_recommendation: str
    estimated_time: Optional[str] = None
    severity_assessment: Optional[str] = None

class ReportRequest(BaseModel):
    """Input for report generation."""
    damage_predictions: Dict[str, float]
    cost_estimate: Decimal
    image_location: Optional[str] = None