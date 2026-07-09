# -*- coding: utf-8 -*-
"""
Centralized configuration using Pydantic v2 BaseSettings.
Loads from .env file automatically.
"""

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All application settings in one place."""

    # ============================================================
    # APPLICATION INFO
    # ============================================================
    APP_NAME: str = "Mental Health Agentic AI Platform"
    APP_VERSION: str = "1.0.0"

    # ============================================================
    # MODEL CONFIGURATION
    # ============================================================
    EMOTION_MODEL: str = "YDVJIYA/distilroberta-base-finetuned-emotion"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    MODEL_DEVICE: int = -1
    MAX_LENGTH: int = 512

    # ============================================================
    # CRISIS DETECTION
    # ============================================================
    CRISIS_CONFIDENCE_THRESHOLD: float = 0.7

    # ============================================================
    # RAG PIPELINE
    # ============================================================
    SIMILARITY_THRESHOLD: float = 0.7
    RAG_TOP_K: int = 3
    TOP_K_RETRIEVAL: int = 3
    FALLBACK_THRESHOLD_DELTA: float = 0.1
    CHROMA_COLLECTION: str = "mental_health_resources"

    # ============================================================
    # EXPLAINER SETTINGS
    # ============================================================
    EXPLAINER_DISPLAY_THRESHOLD: float = 0.5
    EXPLAINER_MIN_TOKENS: int = 10
    EXPLAINER_MAX_TOKENS: int = 100

    # ============================================================
    # API SETTINGS
    # ============================================================
    API_HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_PORT: int = 8000
    API_PREFIX: str = "/api"

    # ============================================================
    # LLM CONFIGURATION (Groq)
    # ============================================================
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    LLM_MAX_TOKENS: int = 200
    LLM_TEMPERATURE: float = 0.7
    LLM_ENABLED: bool = True

    # ============================================================
    # LOGGING
    # ============================================================
    LOG_LEVEL: str = "INFO"

    # ============================================================
    # VALIDATORS
    # ============================================================
    @field_validator("LOG_LEVEL", mode="before")
    @classmethod
    def validate_log_level(cls, v):
        """Normalize log level to uppercase and validate."""
        if isinstance(v, str):
            v = v.upper()
            valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            if v not in valid_levels:
                raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v

    @field_validator("PORT")
    @classmethod
    def validate_port(cls, v):
        """Validate port is in valid range."""
        if not (1 <= v <= 65535):
            raise ValueError("PORT must be between 1 and 65535")
        return v

    @field_validator("SIMILARITY_THRESHOLD")
    @classmethod
    def validate_similarity_threshold(cls, v):
        """Validate similarity threshold is between 0.0 and 1.0."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("SIMILARITY_THRESHOLD must be between 0.0 and 1.0")
        return v

    # ============================================================
    # PYDANTIC V2 CONFIG
    # ============================================================
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )


# Instantiate settings - other files import this
settings = Settings()