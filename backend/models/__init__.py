# backend/models/__init__.py
"""
Models Package

Lazy imports - models are only imported when explicitly requested.
This prevents circular import chains during startup.

Usage:
    from backend.models.classifier import emotion_classifier
    from backend.models.embeddings import embedding_model
    from backend.models.rag_pipeline import rag_pipeline
"""

__all__ = [
    "emotion_classifier",
    "embedding_model",
    "rag_pipeline",
]