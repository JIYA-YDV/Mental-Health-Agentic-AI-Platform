# backend/config/settings.py
"""
Centralized configuration management.

All tunable parameters live here. Override any value via environment
variables or a .env file in the project root.

Usage:
    from backend.config.settings import settings
    threshold = settings.SIMILARITY_THRESHOLD
"""
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration loaded from env vars + .env file."""

    # ── Pydantic v2 config ─────────────────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # ignore unknown env vars instead of crashing
    )

    # ── Application ────────────────────────────────────────────────────
    APP_NAME: str = "Mental Health Agentic AI Platform"
    APP_VERSION: str = "1.0.1"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = Field(default=8000, ge=1, le=65535)

    # ── Model Configuration ────────────────────────────────────────────
    EMOTION_MODEL: str = "j-hartmann/emotion-english-distilroberta-base"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    MAX_SEQUENCE_LENGTH: int = Field(default=512, ge=64, le=2048)
    BATCH_SIZE: int = Field(default=16, ge=1, le=128)

    # ── RAG Configuration ──────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "wellness_knowledge"
    TOP_K_RETRIEVAL: int = Field(default=3, ge=1, le=20)
    SIMILARITY_THRESHOLD: float = Field(default=0.7, ge=0.0, le=1.0)

    # NEW: previously hardcoded in rag_pipeline.py — now configurable
    FALLBACK_THRESHOLD_DELTA: float = Field(
        default=0.15,
        ge=0.0,
        le=0.5,
        description=(
            "If no doc meets SIMILARITY_THRESHOLD, accept top-1 result "
            "when its score is within this delta of the threshold."
        ),
    )

    # ── Explainability ─────────────────────────────────────────────────
    # NEW: previously hardcoded in explainer.py — now configurable
    EXPLAINER_DISPLAY_THRESHOLD: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Minimum abs(weight) for a token to surface in UI.",
    )
    EXPLAINER_MIN_TOKENS: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Fallback: surface at least N tokens even if weak.",
    )
    EXPLAINER_MAX_TOKENS: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Hard cap on tokens returned to UI.",
    )

    # ── Crisis Detection ───────────────────────────────────────────────
    CRISIS_KEYWORDS: List[str] = [
        "suicide", "self-harm", "hopeless",
        "end my life", "can't go on", "worthless",
    ]
    CRISIS_CONFIDENCE_THRESHOLD: float = Field(
        default=0.75, ge=0.0, le=1.0
    )

    # ── Monitoring ─────────────────────────────────────────────────────
    ENABLE_METRICS: bool = True
    LOG_LEVEL: str = "INFO"
    METRICS_PORT: int = Field(default=8001, ge=1, le=65535)

    # ── External Services (optional) ───────────────────────────────────
    OPENAI_API_KEY: Optional[str] = None

    # ── Validators ─────────────────────────────────────────────────────
    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}, got {v}")
        return upper


# ── Singleton instance (NO trailing comma!) ───────────────────────────
settings = Settings()