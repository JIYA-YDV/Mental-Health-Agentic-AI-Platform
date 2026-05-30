from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class ClassificationRequest(BaseModel):
    """
    Input schema for emotion classification requests.
    Validates and cleans incoming text data.
    """
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="User input text for mental health analysis"
    )
    include_explanations: bool = Field(
        default=False,
        description="Whether to include SHAP/LIME explanations"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for conversation tracking"
    )

    @validator("text")
    def clean_text(cls, v):
        """Strip whitespace and validate non-empty."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Text cannot be empty or whitespace only")
        return cleaned


class PredictionScore(BaseModel):
    """Individual emotion prediction with confidence score."""
    label: str
    score: float = Field(ge=0.0, le=1.0)


class ExplanationToken(BaseModel):
    """Token-level explanation from SHAP/LIME."""
    token: str
    importance: float
    sentiment_direction: str  # "positive" | "negative" | "neutral"


class WellnessRecommendation(BaseModel):
    """Wellness resource recommendation from RAG pipeline."""
    title: str
    content: str
    relevance_score: float
    category: str  # "breathing", "mindfulness", "crisis", "coping"
    source: str


class CrisisAssessment(BaseModel):
    """Crisis risk evaluation result."""
    is_crisis: bool
    risk_level: str  # "low" | "medium" | "high" | "critical"
    risk_score: float = Field(ge=0.0, le=1.0)
    crisis_indicators: List[str]
    immediate_resources: List[str]


class ClassificationResponse(BaseModel):
    """
    Complete response schema containing all agent outputs.
    """
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
    model_version: str = "1.0.0"


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