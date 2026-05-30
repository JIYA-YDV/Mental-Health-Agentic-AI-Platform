"""
Models Package

ML model layer containing:
- EmotionClassifier  : DistilRoBERTa transformer for emotion detection
- EmbeddingModel     : Sentence-transformers for semantic embeddings
- RAGPipeline        : ChromaDB vector store + retrieval pipeline

All models follow lazy-loading pattern.
Call load() / initialize() explicitly during app startup.

Usage:
    from backend.models.classifier import emotion_classifier
    from backend.models.embeddings import embedding_model
    from backend.models.rag_pipeline import rag_pipeline

    emotion_classifier.load()
    embedding_model.load()
    rag_pipeline.initialize()
"""

from backend.models.classifier import emotion_classifier
from backend.models.embeddings import embedding_model
from backend.models.rag_pipeline import rag_pipeline

__all__ = [
    "emotion_classifier",
    "embedding_model",
    "rag_pipeline",
]