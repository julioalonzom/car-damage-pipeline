from pydantic import BaseModel
from typing import Dict, Optional
from enum import Enum

class DamageType(str, Enum):
    SCRATCH = "scratch"
    DENT = "dent"
    BROKEN_BUMPER = "broken_bumper"
    GLASS_SHATTER = "glass_shatter"
    PAINT_DAMAGE = "paint_damage"

class DamagePrediction(BaseModel):
    damage_type: DamageType
    confidence: float
    location: Optional[str] = None

class DamageAnalysis(BaseModel):
    predictions: Dict[str, float]
    primary_damage: DamageType