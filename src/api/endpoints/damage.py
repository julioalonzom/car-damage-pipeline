from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from PIL import Image
import io
from typing import Dict

from src.api.dependencies import (
    get_damage_classifier,
    get_cost_estimator,
    get_report_generator
)
from src.api.schemas.requests import DamageAnalysisResponse, ErrorResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

@router.post(
    "/analyze",
    response_model=DamageAnalysisResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def analyze_damage(
    file: UploadFile = File(...),
    damage_classifier: DamageClassifier = Depends(get_damage_classifier),
    cost_estimator: CostEstimator = Depends(get_cost_estimator),
    report_generator: ReportGenerator = Depends(get_report_generator)
):
    """
    Analyze car damage from an uploaded image.
    Returns damage analysis, cost estimates, and a detailed report.
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )

        # Read and validate image
        image_data = await file.read()
        try:
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file"
            )

        # Get damage analysis
        damage_analysis = damage_classifier.classify(image)

        # Get cost estimate
        repair_costs = cost_estimator.estimate(damage_analysis.part_damages)

        # Generate report
        report = report_generator.generate(
            damage_predictions=damage_analysis.part_damages,
            cost_estimate=repair_costs.total_cost
        )

        return DamageAnalysisResponse(
            damage_analysis=damage_analysis.part_damages,
            most_damaged_part=damage_analysis.most_damaged_part,
            repair_costs=repair_costs.breakdown,
            total_cost=repair_costs.total_cost,
            report=report.summary,
            confidence_score=damage_analysis.max_confidence
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )