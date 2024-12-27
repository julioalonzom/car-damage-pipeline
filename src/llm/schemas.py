# src/llm/schemas.py
from pydantic import BaseModel
from typing import Dict, Optional
from decimal import Decimal
from src.vision.schemas import CarPart

class DamageReport(BaseModel):
    """Structured damage report from LLM."""
    summary: str
    details: str
    repair_recommendations: str  # Changed from repair_recommendation to match prompt
    severity_assessment: Optional[str] = None
    estimated_time: Optional[str] = None

class ReportRequest(BaseModel):
    """Request model for report generation."""
    part_damages: Dict[CarPart, float]  # Matches our DamageAnalysis output
    cost_estimate: Decimal
    image_location: Optional[str] = None