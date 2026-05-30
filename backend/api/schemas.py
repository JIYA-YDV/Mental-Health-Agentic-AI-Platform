# backend/api/schemas.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime


class ClassificationRequest(BaseModel):
    """
    Input schema for emotion classification requests.
    """
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="User input text for mental health analysis"
    )
    include_explanations: bool = Field(
        default=False,
        description="Whether to include token-level explanations"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for conversation tracking"
    )

    @validator("text")
    def clean_text(cls, v):
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Text cannot be empty or whitespace only")
        return cleaned


class PredictionScore(BaseModel):
    """Individual emotion prediction with confidence score."""
    label: str
    score: float = Field(ge=0.0, le=1.0)


class ExplanationToken(BaseModel):
    """Token-level explanation from explainer."""
    token: str
    importance: float
    sentiment_direction: str


class WellnessRecommendation(BaseModel):
    """Wellness resource recommendation from RAG pipeline."""
    title: str
    content: str
    relevance_score: float
    category: str
    source: str


class CrisisAssessment(BaseModel):
    """Crisis risk evaluation result."""
    is_crisis: bool
    risk_level: str
    risk_score: float = Field(ge=0.0, le=1.0)
    crisis_indicators: List[str]
    immediate_resources: List[str]


class ClassificationResponse(BaseModel):
    """
    Complete response schema containing all agent outputs.
    Fix: model_config set to allow model_ prefix fields.
    """

    # ── Tell Pydantic to allow 'model_' prefixed field names ──────────────
    model_config = {"protected_namespaces": ()}

    # Core classification
    emotion: str
    confidence: float = Field(ge=0.0, le=1.0)
    all_predictions: List[PredictionScore]

    # Wellness recommendations from RAG
    recommendations: List[WellnessRecommendation]

    # Crisis assessment
    crisis_assessment: CrisisAssessment

    # Explainability (optional)
    explanations: Optional[List[ExplanationToken]] = None

    # Metadata
    session_id: Optional[str] = None
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_version: str = "1.0.0"   # This field caused the warning — now fixed


class HealthResponse(BaseModel):
    """Health check endpoint response."""
    status: str
    version: str
    models_loaded: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Standardized error response."""
    error: str
    detail: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)