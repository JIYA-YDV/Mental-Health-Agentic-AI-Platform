# backend/api/schemas.py
"""
Pydantic v2 schemas for API request/response contracts.
Mirrors the data shapes produced by orchestrator + explainer modules.
"""
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Request ────────────────────────────────────────────────────────────
class ClassificationRequest(BaseModel):
    """Input schema for emotion classification requests."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="User input text for mental health analysis.",
    )
    include_explanations: bool = Field(
        default=False,
        description="Include token-level attribution in response.",
    )
    session_id: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Optional session ID for conversation tracking.",
    )

    @field_validator("text")
    @classmethod
    def clean_text(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Text cannot be empty or whitespace only.")
        return cleaned


# ── Sub-schemas ────────────────────────────────────────────────────────
class PredictionScore(BaseModel):
    """Single emotion prediction with confidence."""

    label: str
    score: float = Field(ge=0.0, le=1.0)


class ExplanationToken(BaseModel):
    """
    Token-level attribution from the explainer.

    Must match the dict shape returned by EmotionExplainer.explain():
        {"token": str, "weight": float, "influence": "positive"|"negative"}
    """

    token: str
    weight: float = Field(
        ge=-1.5, le=1.5,
        description="Signed attribution score. Positive = supports emotion.",
    )
    influence: Literal["positive", "negative"]


class WellnessRecommendation(BaseModel):
    """Wellness resource from the RAG pipeline."""

    title: str
    content: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    category: str
    source: str


class CrisisAssessment(BaseModel):
    """Crisis-risk evaluation."""

    is_crisis: bool
    risk_level: Literal["low", "medium", "high", "critical"]
    risk_score: float = Field(ge=0.0, le=1.0)
    crisis_indicators: List[str] = Field(default_factory=list)
    immediate_resources: List[str] = Field(default_factory=list)


# ── Top-level Response ─────────────────────────────────────────────────
class ClassificationResponse(BaseModel):
    """Full multi-agent analysis response."""

    # Allow `model_version` field without triggering Pydantic's
    # "model_" protected-namespace warning.
    model_config = ConfigDict(protected_namespaces=())

    emotion: str
    confidence: float = Field(ge=0.0, le=1.0)
    all_predictions: List[PredictionScore]

    recommendations: List[WellnessRecommendation]
    crisis_assessment: CrisisAssessment

    explanations: Optional[List[ExplanationToken]] = None
    explanation_summary: Optional[str] = Field(
        default=None,
        description="Human-readable explanation paragraph.",
    )

    session_id: Optional[str] = None
    processing_time_ms: float = Field(ge=0.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_version: str = "1.0.1"


# ── Health / Errors ────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    """Health-check response."""

    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    models_loaded: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Standardized error envelope."""

    error: str
    detail: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)