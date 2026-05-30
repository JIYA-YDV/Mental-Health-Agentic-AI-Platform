from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Central configuration management using pydantic-settings.
    Reads from environment variables and .env file.
    """

    # Application
    APP_NAME: str = "Mental Health Agentic AI Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Model Configuration
    EMOTION_MODEL: str = "j-hartmann/emotion-english-distilroberta-base"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    MAX_SEQUENCE_LENGTH: int = 512
    BATCH_SIZE: int = 16

    # RAG Configuration
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    TOP_K_RETRIEVAL: int = 3
    SIMILARITY_THRESHOLD: float = 0.7

    # Crisis Detection
    CRISIS_KEYWORDS: list = [
        "suicide", "self-harm", "hopeless",
        "end my life", "can't go on", "worthless"
    ]
    CRISIS_CONFIDENCE_THRESHOLD: float = 0.75

    # Monitoring
    ENABLE_METRICS: bool = True
    LOG_LEVEL: str = "INFO"
    METRICS_PORT: int = 8001

    # API Keys (optional external services)
    OPENAI_API_KEY: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = True


# Singleton instance
settings = Settings()