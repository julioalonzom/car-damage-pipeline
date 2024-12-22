# src/vision/schemas.py
from enum import Enum
from typing import Dict, Optional
from pydantic import BaseModel

class CarPart(str, Enum):
    HEADLAMP = "headlamp"
    REAR_BUMPER = "rear_bumper"
    DOOR = "door"
    HOOD = "hood"
    FRONT_BUMPER = "front_bumper"

class DamageAnalysis(BaseModel):
    """Multi-label damage predictions per part"""
    part_damages: Dict[CarPart, float]
    most_damaged_part: CarPart
    max_confidence: float