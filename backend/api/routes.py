from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from datetime import datetime
import structlog

from backend.api.schemas import (
    ClassificationRequest,
    ClassificationResponse,
    HealthResponse,
    ErrorResponse,
    PredictionScore,
    WellnessRecommendation,
    CrisisAssessment,
    ExplanationToken
)
from backend.agents.orchestrator import AgentOrchestrator
from backend.explainability.explainer import explainer
from backend.monitoring.metrics import record_request
from backend.models.classifier import emotion_classifier

logger = structlog.get_logger(__name__)
router = APIRouter()

# Orchestrator instance
orchestrator = AgentOrchestrator()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"]
)
async def health_check():
    """
    Health check endpoint.
    Verifies application is running and models are loaded.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=emotion_classifier.is_loaded
    )


@router.post(
    "/classify",
    response_model=ClassificationResponse,
    tags=["Analysis"],
    summary="Analyze emotion and provide mental health insights",
    description=(
        "Accepts user text and returns emotion classification, "
        "wellness recommendations, and crisis assessment. "
        "Powered by multi-agent AI orchestration."
    )
)
async def classify_text(request: ClassificationRequest):
    """
    Main classification endpoint.

    Pipeline:
    1. Validate input (handled by Pydantic schema)
    2. Run multi-agent orchestration
    3. Optionally generate explanations
    4. Record monitoring metrics
    5. Return structured response
    """
    logger.info(
        "Classification request received",
        text_length=len(request.text),
        session_id=request.session_id
    )

    try:
        # ── Run orchestrator ───────────────────────────────────────────
        result = await orchestrator.process(
            text=request.text,
            session_id=request.session_id
        )

        # ── Optional explanations ──────────────────────────────────────
        explanations = None
        if request.include_explanations:
            raw_explanations = explainer.explain(
                text=request.text,
                emotion=result["emotion"]
            )
            explanations = [
                ExplanationToken(**e) for e in raw_explanations
            ]

        # ── Record metrics ─────────────────────────────────────────────
        record_request(
            emotion=result["emotion"],
            risk_level=result["crisis_assessment"]["risk_level"],
            latency_ms=result["processing_time_ms"],
            confidence=result["confidence"],
            is_crisis=result["crisis_assessment"]["is_crisis"]
        )

        # ── Build response ─────────────────────────────────────────────
        return ClassificationResponse(
            emotion=result["emotion"],
            confidence=result["confidence"],
            all_predictions=[
                PredictionScore(**p) for p in result["all_predictions"]
            ],
            recommendations=[
                WellnessRecommendation(**r)
                for r in result["recommendations"]
            ],
            crisis_assessment=CrisisAssessment(
                **result["crisis_assessment"]
            ),
            explanations=explanations,
            session_id=result["session_id"],
            processing_time_ms=result["processing_time_ms"]
        )

    except ValueError as e:
        logger.warning("Validation error in classification", error=str(e))
        raise HTTPException(status_code=422, detail=str(e))

    except RuntimeError as e:
        logger.error("Runtime error in classification", error=str(e))
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        logger.error("Unexpected error in classification", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Please try again."
        )