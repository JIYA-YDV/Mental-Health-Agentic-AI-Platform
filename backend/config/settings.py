# -*- coding: utf-8 -*-
"""
Centralized configuration using Pydantic v2 BaseSettings.
Loads from .env file automatically.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All application settings in one place."""

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
    CHROMA_COLLECTION: str = "mental_health_resources"

    # ============================================================
    # API SETTINGS
    # ============================================================
    API_HOST: str = "0.0.0.0"
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