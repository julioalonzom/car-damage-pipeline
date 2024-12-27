# src/cost/estimator.py
from decimal import Decimal
from typing import Dict, Optional
import json
from pathlib import Path

from src.utils.metrics import log_performance
from src.vision.schemas import CarPart
from src.cost.schemas import RepairCost
from src.utils.logger import get_logger
from src.config import get_settings

logger = get_logger(__name__)

class CostEstimator:
    def __init__(self, config_path: Optional[Path] = None):
        self.settings = get_settings()
        
        # Updated base costs for car parts
        self.base_costs = {
            CarPart.HEADLAMP: Decimal("800.00"),
            CarPart.REAR_BUMPER: Decimal("1200.00"),
            CarPart.DOOR: Decimal("1500.00"),
            CarPart.HOOD: Decimal("1000.00"),
            CarPart.FRONT_BUMPER: Decimal("1200.00")
        }
        
        # If config provided, override defaults
        if config_path and config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                self.base_costs.update({
                    CarPart[k.upper()]: Decimal(str(v)) 
                    for k, v in config["base_costs"].items()
                })

    @log_performance("Cost Estimator")
    def estimate(self, part_damages: Dict[CarPart, float]) -> RepairCost:
        """
        Estimate repair costs based on damaged parts and confidence.
        
        Args:
            part_damages: Dictionary mapping car parts to damage confidence
        """
        try:
            cost_breakdown = {}
            total_cost = Decimal("0.00")
            TWOPLACES = Decimal('0.01')
            
            # Calculate confidence score first
            confidence_score = sum(part_damages.values()) / len(part_damages)
            
            for part, confidence in part_damages.items():
                if confidence > 0.3:
                    base_cost = self.base_costs[part]
                    adjusted_cost = (base_cost * Decimal(str(confidence))).quantize(TWOPLACES)
                    cost_breakdown[part] = adjusted_cost
                    total_cost += adjusted_cost
            
            total_cost = total_cost.quantize(TWOPLACES)
            
            logger.info(
                f"Cost Metrics | "
                f"Total Cost: ${total_cost:.2f} | "
                f"Parts Assessed: {len(cost_breakdown)} | "
                f"Average Confidence: {confidence_score:.2f}"
            )
            
            return RepairCost(
                total_cost=total_cost,
                breakdown=cost_breakdown,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Error estimating costs: {str(e)}")
            raise RuntimeError(f"Failed to estimate repair costs: {str(e)}")