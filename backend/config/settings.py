# -*- coding: utf-8 -*-
"""
Application settings — comprehensive, covers all subsystems.
"""
from typing import List
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Central configuration for the Mental Health AI Platform."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── App Metadata ──────────────────────────────────────────────────────
    APP_NAME: str = "Mental Health Agentic AI Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"

    # ── Server ────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_PREFIX: str = ""

    # ── Groq LLM ──────────────────────────────────────────────────────────
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "groq/compound-mini"
    GROQ_TEMPERATURE: float = 0.7
    GROQ_MAX_TOKENS: int = 350
    GROQ_TOP_P: float = 0.95

    # ── ML Models (both name variants for safety) ─────────────────────────
    EMOTION_MODEL: str = "YDVJIYA/distilroberta-base-finetuned-emotion"
    EMOTION_MODEL_NAME: str = "YDVJIYA/distilroberta-base-finetuned-emotion"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Model behavior
    MAX_SEQUENCE_LENGTH: int = 512
    MAX_LENGTH: int = 512
    BATCH_SIZE: int = 16
    DEVICE: str = "cpu"
    USE_FAST_TOKENIZER: bool = False
    MODEL_CACHE_DIR: str = str(PROJECT_ROOT / ".model_cache")

    # ── RAG / Vector Store ────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = str(PROJECT_ROOT / "data" / "chroma_db")
    CHROMA_COLLECTION_NAME: str = "mental_health_kb"
    RAG_TOP_K: int = 3
    RAG_SIMILARITY_THRESHOLD: float = 0.5
    KNOWLEDGE_BASE_PATH: str = str(PROJECT_ROOT / "datasets" / "knowledge_base")

    # ── Monitoring & Metrics ──────────────────────────────────────────────
    METRICS_PORT: int = 9090
    ENABLE_METRICS: bool = True
    PROMETHEUS_ENABLED: bool = True

    # ── Safety ────────────────────────────────────────────────────────────
       # ── Crisis Detection Keywords ─────────────────────────────────────────
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
    
    SAFETY_OVERRIDE_ENABLED: bool = True
    CRISIS_CONFIDENCE_THRESHOLD: float = 0.5

    # ── Agents ────────────────────────────────────────────────────────────
    MAX_AGENT_ITERATIONS: int = 5
    AGENT_TIMEOUT_SECONDS: int = 30

    # ── CORS ──────────────────────────────────────────────────────────────
    CORS_ORIGINS: str = "http://localhost:8501,http://127.0.0.1:8501"

    # ── Rate Limiting ─────────────────────────────────────────────────────
    RATE_LIMIT_PER_MINUTE: int = 60


# Singleton instance
settings = Settings()