from decimal import Decimal
from typing import Dict, Optional
import json
from pathlib import Path

from src.cost.schemas import RepairCost
from src.utils.logger import get_logger
from src.config import get_settings

logger = get_logger(__name__)

class CostEstimator:
    """
    Estimates repair costs based on damage predictions.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.settings = get_settings()

        # Load cost matrix from config file or use defaults
        if config_path and config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                self.base_costs = {k: Decimal(str(v)) for k, v in config["base_costs"].items()}
        else:
            logger.warning("No cost file found, using default costs.")    
            self.base_costs = {
                "scratch": Decimal("200.00"),
                "dent": Decimal("500.00"),
                "broken_bumper": Decimal("1200.00"),
                "glass_shatter": Decimal("800.00"),
                "paint_damage": Decimal("400.00")
            }

        # Severity multipliers
        self.severity_multipliers = {
            "low": Decimal("0.7"),
            "medium": Decimal("1.0"),
            "high": Decimal("1.5")
        }

    def _calculate_severity(self, confidence: float) -> str:
        """
        Determine damage severity based on model confidence.
        """
        if confidence < 0.4:
            return "low"
        elif confidence < 0.7:
            return "medium"
        else:
            return "high"

    def estimate(self, damage_predictions: Dict[str, float]) -> RepairCost:
        """
        Estimate repair costs based on damage predictions.

        Args:
            damage_predictions: Dict of damage types and their confidence scores

        Returns:
            RepairCost object with total cost and breakdown
        """
        try:
            cost_breakdown = {}
            total_cost = Decimal("0.00")

            for damage_type, confidence in damage_predictions.items():
                if damage_type not in self.base_costs:
                    logger.warning(f"Unknown damage type: {damage_type}, skipping")
                    continue

                severity = self._calculate_severity(confidence)
                multiplier = self.severity_multipliers[severity]

                base_cost = self.base_costs[damage]
                adjusted_cost = base_cost * multiplier

                cost_breakdown[damage_type] = adjusted_cost
                total_cost += adjusted_cost

            # Calculate overall confidence score
            confidence_score = sum(damage_predictions.values()) / len(damage_predictions)

            return RepairCost(
                total_cost=total_cost,
                breakdown=cost_breakdown,
                currency=self.settings.currency,
                confidence_score=confidence_score
            )

        except Exception as e:
            logger.error(f"Error during cost estimation: {str(e)}")
            raise RuntimeError(f"Failed to estimate repair costs: {str(e)}")