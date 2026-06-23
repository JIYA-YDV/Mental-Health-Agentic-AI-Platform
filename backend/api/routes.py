# backend/api/routes.py
"""
FastAPI route handlers for the Mental Health Agentic AI Platform.
"""
from typing import Any, Dict

import structlog
from fastapi import APIRouter, HTTPException, status

from backend.agents.orchestrator import AgentOrchestrator
from backend.api.schemas import (
    ClassificationRequest,
    ClassificationResponse,
    CrisisAssessment,
    ExplanationToken,
    HealthResponse,
    PredictionScore,
    WellnessRecommendation,
)
from backend.config.settings import settings
from backend.explainability.explainer import explainer
from backend.models.classifier import emotion_classifier
from backend.monitoring.metrics import record_request

logger = structlog.get_logger(__name__)
router = APIRouter()

# Singleton orchestrator (initialized once per process)
orchestrator = AgentOrchestrator()


# ──────────────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────────────
@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Liveness + model-readiness probe",
)
async def health_check() -> HealthResponse:
    """Return service status and confirm ML models are loaded."""
    models_ok = emotion_classifier.is_loaded
    return HealthResponse(
        status="healthy" if models_ok else "degraded",
        version=settings.APP_VERSION,
        models_loaded=models_ok,
    )


# ──────────────────────────────────────────────────────────────────────
# Classification (main endpoint)
# ──────────────────────────────────────────────────────────────────────
@router.post(
    "/classify",
    response_model=ClassificationResponse,
    tags=["Analysis"],
    summary="Analyze emotion and provide mental-health insights",
    description=(
        "Accepts user text and returns emotion classification, "
        "wellness recommendations, and crisis assessment via "
        "multi-agent orchestration."
    ),
)
async def classify_text(
    request: ClassificationRequest,
) -> ClassificationResponse:
    """Run the full multi-agent pipeline on a single text input."""
    logger.info(
        "Classification request received",
        text_length=len(request.text),
        session_id=request.session_id,
        include_explanations=request.include_explanations,
    )

    try:
        # ── Multi-agent orchestration ─────────────────────────────────
        result: Dict[str, Any] = await orchestrator.process(
            text=request.text,
            session_id=request.session_id,
        )

        # ── Optional token-level explanations ─────────────────────────
        explanations = None
        explanation_summary = None

        if request.include_explanations:
            explanation_result = explainer.explain(
                text=request.text,
                primary_emotion=result["emotion"],
            )
            explanations = [
                ExplanationToken(**tok)
                for tok in explanation_result.get("tokens", [])
            ]
            explanation_summary = explanation_result.get("summary")

        # ── Record Prometheus / structlog metrics ─────────────────────
        record_request(
            emotion=result["emotion"],
            risk_level=result["crisis_assessment"]["risk_level"],
            latency_ms=result["processing_time_ms"],
            confidence=result["confidence"],
            is_crisis=result["crisis_assessment"]["is_crisis"],
        )

        # ── Build and return response ─────────────────────────────────
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
            crisis_assessment=CrisisAssessment(**result["crisis_assessment"]),
            explanations=explanations,
            explanation_summary=explanation_summary,
            session_id=result.get("session_id"),
            processing_time_ms=result["processing_time_ms"],
            model_version=settings.APP_VERSION,
        )

    except ValueError as e:
        logger.warning("Validation error in /classify", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    except RuntimeError as e:
        logger.error("Service unavailable in /classify", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )

    except Exception as e:
        logger.exception("Unexpected error in /classify", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error. Please try again.",
        )