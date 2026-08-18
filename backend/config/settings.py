# -*- coding: utf-8 -*-
"""
Application settings — canonical single source of truth.

All settings referenced anywhere in the codebase are defined here.
Values can be overridden via environment variables or .env file.

Audited against all backend/**/*.py on 2026-08-19.
"""

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root — used for locating .env, data dirs, etc.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Central configuration for the Mental Health Agentic AI Platform."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ═══════════════════════════════════════════════════════════════════
    # APP METADATA
    # ═══════════════════════════════════════════════════════════════════
    APP_NAME: str = "Mental Health Agentic AI Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"

    # ═══════════════════════════════════════════════════════════════════
    # SERVER
    # ═══════════════════════════════════════════════════════════════════
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_PREFIX: str = ""

    # ═══════════════════════════════════════════════════════════════════
    # GROQ LLM
    # ═══════════════════════════════════════════════════════════════════
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "groq/compound-mini"
    GROQ_TEMPERATURE: float = 0.7
    GROQ_MAX_TOKENS: int = 350
    GROQ_TOP_P: float = 0.95

    # ═══════════════════════════════════════════════════════════════════
    # ML MODELS
    # ═══════════════════════════════════════════════════════════════════
    EMOTION_MODEL: str = "YDVJIYA/distilroberta-base-finetuned-emotion"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Model inference behavior
    MAX_SEQUENCE_LENGTH: int = 512
    BATCH_SIZE: int = 16
    DEVICE: str = "cpu"
    USE_FAST_TOKENIZER: bool = False
    MODEL_CACHE_DIR: str = str(PROJECT_ROOT / ".model_cache")

    # ═══════════════════════════════════════════════════════════════════
    # RAG / VECTOR STORE
    # ═══════════════════════════════════════════════════════════════════
    CHROMA_PERSIST_DIR: str = str(PROJECT_ROOT / "data" / "chroma_db")
    CHROMA_COLLECTION_NAME: str = "mental_health_kb"
    KNOWLEDGE_BASE_PATH: str = str(PROJECT_ROOT / "datasets" / "knowledge_base")

    # Retrieval tuning
    TOP_K_RETRIEVAL: int = 3
    SIMILARITY_THRESHOLD: float = 0.4
    FALLBACK_SIMILARITY_FLOOR: float = 0.25

    # ═══════════════════════════════════════════════════════════════════
    # SAFETY / CRISIS DETECTION
    # ═══════════════════════════════════════════════════════════════════
    CRISIS_CONFIDENCE_THRESHOLD: float = 0.6
    SAFETY_OVERRIDE_ENABLED: bool = True

    CRISIS_KEYWORDS: List[str] = [
        "suicide",
        "suicidal",
        "kill myself",
        "end my life",
        "want to die",
        "wanna die",
        "don't want to be here",
        "dont want to be here",
        "no way out",
        "better off dead",
        "no reason to live",
        "can't go on",
        "cant go on",
        "end it all",
        "take my life",
        "self harm",
        "self-harm",
        "hurt myself",
    ]

    # ═══════════════════════════════════════════════════════════════════
    # EXPLAINABILITY (SHAP)
    # ═══════════════════════════════════════════════════════════════════
    EXPLAINER_ENABLED: bool = True
    EXPLAINER_MAX_TOKENS: int = 50           # max tokens SHAP will analyze
    EXPLAINER_MIN_TOKENS: int = 3            # min tokens required to run
    EXPLAINER_TOP_N: int = 10                # top N tokens to return in response
    EXPLAINER_DISPLAY_THRESHOLD: float = 0.05  # min weight to include in output
    SHAP_MAX_EVALS: int = 100                # SHAP sample count for approximation

    # ═══════════════════════════════════════════════════════════════════
    # MONITORING & METRICS
    # ═══════════════════════════════════════════════════════════════════
    METRICS_PORT: int = 8001
    ENABLE_METRICS: bool = True
    PROMETHEUS_ENABLED: bool = True

    # ═══════════════════════════════════════════════════════════════════
    # AGENTS
    # ═══════════════════════════════════════════════════════════════════
    MAX_AGENT_ITERATIONS: int = 5
    AGENT_TIMEOUT_SECONDS: int = 30

    # ═══════════════════════════════════════════════════════════════════
    # CORS & SECURITY
    # ═══════════════════════════════════════════════════════════════════
    CORS_ORIGINS: str = "http://localhost:8501,http://127.0.0.1:8501"
    RATE_LIMIT_PER_MINUTE: int = 60


# Singleton instance — import this everywhere
settings = Settings()