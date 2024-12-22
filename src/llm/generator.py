# src/llm/generator.py
from typing import Dict
from decimal import Decimal
import json
from openai import OpenAI

from src.llm.schemas import DamageReport, ReportRequest
from src.utils.logger import get_logger
from src.config import get_settings

logger = get_logger(__name__)

class ReportGenerator:
    """Generates detailed damage reports using LLM."""

    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI()

        # Base prompt template - keeping it here for MVP
        # In production, this would be in a template file or database
        self.base_prompt = """
        You are an expert car damage assessor. Based on the following damage detection results 
        and cost estimates, provide a professional damage assessment report.
        
        Detected Damages and Confidences:
        {damages}
        
        Estimated Repair Cost: ${cost}
        
        Generate a professional report including:
        1. Brief summary
        2. Detailed damage assessment
        3. Repair recommendations
        
        Format the response as JSON with keys: 'summary', 'details', 'repair_recommendations'
        """

    def _format_damages(self, damages: Dict[str, float]) -> str:
        """Format damage predictions for prompt."""
        return "\n".join(
            f"- {damage.replace("_", ' ').title()}: {confidence*100:.1f}% confidence"
            for damage, confidence in damages.items()
        )

    def generate(self, request: ReportRequest) -> DamageReport:
        """
        Generate a damage report using LLM.
        
        Args:
            request: ReportRequest containing damage predictions and cost estimate
            
        Returns:
            DamageReport containing structured report
        """
        try:
            # Format prompt
            prompt = self.base_prompt.format(
                damages=self._format_damages(request.damage_predictions),
                cost=request.cost_estimate
            )

            # Get LLM response
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "developer", "content": "You are a professional car damage assessor."},
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=500
            )

            # Parse response
            try:
                report_dict = json.loads(reponse.choices[0].message.content)
                return DamageReport(**report_dict)
            except Exception as e:
                logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
                # Fallback: Try to extract sections manually
                content = response.choices[0].message.content
                return DamageReport(
                    summary=content[:200],
                    details=content[200:400],
                    repair_recommendation=content[400:]
                )

        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise RuntimeError(f"Failed to generate report: {str(e)}")