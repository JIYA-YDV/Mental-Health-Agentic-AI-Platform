# tests/conftest.py
"""
Shared pytest fixtures.

Heavy ML models (DistilRoBERTa, MiniLM) are MOCKED so the entire
test suite runs in < 5 seconds without downloading 500MB of weights.

Integration tests can opt into real models via the `requires_models`
marker if needed.
"""
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ──────────────────────────────────────────────────────────────────────
# Mock data factories
# ──────────────────────────────────────────────────────────────────────

def _fake_predictions() -> List[Dict[str, float]]:
    """Standard fake classifier output used across tests."""
    return [
        {"label": "sadness", "score": 0.85},
        {"label": "neutral", "score": 0.08},
        {"label": "fear", "score": 0.04},
        {"label": "joy", "score": 0.02},
        {"label": "anger", "score": 0.005},
        {"label": "surprise", "score": 0.003},
        {"label": "disgust", "score": 0.002},
    ]


def _fake_recommendations() -> List[Dict[str, Any]]:
    return [
        {
            "title": "Mock Wellness Tip",
            "content": "Take a deep breath and practice mindfulness.",
            "relevance_score": 0.82,
            "category": "mindfulness",
            "source": "Mental Health Knowledge Base",
        }
    ]


def _fake_crisis_assessment(is_crisis: bool = False) -> Dict[str, Any]:
    return {
        "is_crisis": is_crisis,
        "risk_level": "high" if is_crisis else "low",
        "risk_score": 0.85 if is_crisis else 0.05,
        "crisis_indicators": ["end my life"] if is_crisis else [],
        "immediate_resources": ["988 Suicide & Crisis Lifeline"],
    }


# ──────────────────────────────────────────────────────────────────────
# Component-level fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_classifier():
    """Mock EmotionClassifier — returns sadness prediction by default."""
    mock = MagicMock()
    mock.is_loaded = True
    mock.predict.return_value = {
        "emotion": "sadness",
        "confidence": 0.85,
        "all_predictions": _fake_predictions(),
    }
    mock.load.return_value = None
    return mock


@pytest.fixture
def mock_embedding_model():
    """Mock sentence-transformers embedding model."""
    mock = MagicMock()
    mock.is_loaded = True
    # Encode returns a fake 384-dim vector (MiniLM size)
    mock.encode.return_value = [[0.1] * 384]
    mock.encode_single.return_value = [0.1] * 384
    return mock


@pytest.fixture
def mock_rag_pipeline():
    """Mock RAG pipeline — returns one recommendation by default."""
    mock = MagicMock()
    mock.is_initialized = True
    mock.retrieve.return_value = _fake_recommendations()
    return mock


@pytest.fixture
def mock_crisis_agent():
    """Mock CrisisAgent — non-crisis by default."""
    mock = MagicMock()

    async def _run(text, classification_result):
        is_crisis = any(
            kw in text.lower()
            for kw in ["end my life", "kill myself", "suicide"]
        )
        return _fake_crisis_assessment(is_crisis=is_crisis)

    mock.run = _run
    return mock


# ──────────────────────────────────────────────────────────────────────
# API client fixture (full stack, with mocks)
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def api_client(
    mock_classifier,
    mock_embedding_model,
    mock_rag_pipeline,
):
    """
    FastAPI TestClient with heavy dependencies replaced.

    Use this for fast integration tests on routes without needing
    real models loaded.
    """
    # Patch BEFORE importing main app so lifespan uses mocks
    with patch(
        "backend.models.classifier.emotion_classifier", mock_classifier
    ), patch(
        "backend.models.embeddings.embedding_model", mock_embedding_model
    ), patch(
        "backend.models.rag_pipeline.rag_pipeline", mock_rag_pipeline
    ):
        from backend.main import app
        # Don't actually run lifespan (no model loading) for unit tests
        with TestClient(app) as client:
            yield client


# ──────────────────────────────────────────────────────────────────────
# Sample inputs
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_text_sad() -> str:
    return "I feel hopeless, tired, and not earning anything."


@pytest.fixture
def sample_text_crisis() -> str:
    return "I want to end my life, I can't do this anymore."


@pytest.fixture
def sample_text_neutral() -> str:
    return "Today is Tuesday and I had coffee."


@pytest.fixture
def sample_text_mixed() -> str:
    return "I am too good but I am not earning and that is making my head spin."