# src/llm/generator.py
from typing import Dict, Optional
from decimal import Decimal
import json
from openai import OpenAI
import time

from src.vision.schemas import CarPart
from src.llm.schemas import DamageReport, ReportRequest
from src.utils.logger import get_logger
from src.utils.metrics import log_performance
from src.config import get_settings

logger = get_logger(__name__)

class ReportGenerator:
    """Generates detailed damage reports using LLM."""

    def __init__(self, client: Optional[OpenAI] = None):
        self.settings = get_settings()
        self.client = client or self._initialize_client()
        self.base_prompt = """
        You are an expert car damage assessor. Based on the following damage detection results 
        and cost estimates, provide a professional damage assessment report.
        
        Detected Car Part Damages and Confidences:
        {damages}
        
        Estimated Total Repair Cost: ${cost:.2f}
        
        Generate a detailed damage assessment report including:
        1. Brief summary of the overall damage situation
        2. Detailed assessment of each damaged part
        3. Specific repair recommendations for each damaged part
        
        Return the response as a JSON object with exactly these keys:
        {{
            "summary": "Brief overview of damage",
            "details": "Detailed part-by-part assessment",
            "repair_recommendations": "Specific repair steps needed",
            "severity_assessment": "Rate severity as LOW, MEDIUM, or HIGH",
            "estimated_time": "Estimated repair time in days"
        }}
        """

    def _initialize_client(self) -> OpenAI:
        """Initialize OpenAI client."""
        try:
            return OpenAI()
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise RuntimeError(f"OpenAI client initialization failed: {str(e)}")

    def _format_damages(self, part_damages: Dict[CarPart, float]) -> str:
        """Format part damages for prompt."""
        return "\n".join(
            f"- {part.value.replace('_', ' ').title()}: {confidence*100:.1f}% damage confidence"
            for part, confidence in part_damages.items()
        )

    def _create_report_from_response(self, content: str) -> DamageReport:
        """Create report from LLM response."""
        try:
            report_dict = json.loads(content)
            return DamageReport(**report_dict)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
            # Fallback: Create structured report from raw text
            sections = content.split("\n\n")
            return DamageReport(
                summary=sections[0] if sections else "Error parsing response",
                details=sections[1] if len(sections) > 1 else "",
                repair_recommendations=sections[2] if len(sections) > 2 else "",
                severity_assessment="MEDIUM",  # Default value
                estimated_time="3-5 days"  # Default value
            )

    @log_performance("LLM Report")
    def generate(self, request: ReportRequest) -> DamageReport:
        """
        Generate a damage report using LLM.
        
        Args:
            request: ReportRequest containing part damages and cost estimate
            
        Returns:
            DamageReport containing structured report
        """
        try:
            llm_start = time.perf_counter()
            
            # Format prompt
            prompt = self.base_prompt.format(
                damages=self._format_damages(request.part_damages),
                cost=float(request.cost_estimate)
            )

            # Get LLM response
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional car damage assessor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )

            llm_time = time.perf_counter() - llm_start
            logger.info(
                f"LLM Metrics | "
                f"Response Time: {llm_time:.3f}s | "
                f"Token Count: {len(response.choices[0].message.content.split())} | "
                f"Parts Analyzed: {len(request.part_damages)}"
            )

            return self._create_report_from_response(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise RuntimeError(f"Failed to generate report: {str(e)}")