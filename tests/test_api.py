"""
API Integration Tests

Tests for FastAPI endpoints using httpx AsyncClient.
Models are mocked to avoid loading actual ML weights.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock


# ── Shared mock result ─────────────────────────────────────────────────────

MOCK_ORCHESTRATOR_RESULT = {
    "emotion": "Joy / Happiness",
    "confidence": 0.92,
    "all_predictions": [
        {"label": "Joy / Happiness", "score": 0.92},
        {"label": "Neutral", "score": 0.05},
        {"label": "Sadness / Depression", "score": 0.03}
    ],
    "recommendations": [
        {
            "title": "Gratitude Practice",
            "content": "Write 3 things you are grateful for.",
            "relevance_score": 0.88,
            "category": "mindfulness",
            "source": "Knowledge Base"
        }
    ],
    "crisis_assessment": {
        "is_crisis": False,
        "risk_level": "low",
        "risk_score": 0.05,
        "crisis_indicators": [],
        "immediate_resources": [
            "988 Suicide & Crisis Lifeline"
        ]
    },
    "session_id": None,
    "processing_time_ms": 245.5
}

MOCK_CRISIS_RESULT = {
    "emotion": "Sadness / Depression",
    "confidence": 0.89,
    "all_predictions": [
        {"label": "Sadness / Depression", "score": 0.89}
    ],
    "recommendations": [],
    "crisis_assessment": {
        "is_crisis": True,
        "risk_level": "critical",
        "risk_score": 0.9,
        "crisis_indicators": ["hopeless", "end my life"],
        "immediate_resources": [
            "988 Suicide & Crisis Lifeline",
            "Crisis Text Line: Text HOME to 741741",
            "Emergency Services: Call 911"
        ]
    },
    "session_id": None,
    "processing_time_ms": 180.2
}


# ── Health Check Tests ─────────────────────────────────────────────────────

class TestHealthEndpoint:

    @pytest.mark.asyncio
    async def test_health_returns_200(self):
        """GET /health returns 200 OK."""
        from httpx import AsyncClient
        from backend.main import app

        with patch(
            "backend.models.classifier.emotion_classifier"
        ) as mock_clf:
            mock_clf.is_loaded = True

            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get("/health")

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_response_schema(self):
        """GET /health response contains required fields."""
        from httpx import AsyncClient
        from backend.main import app

        with patch(
            "backend.models.classifier.emotion_classifier"
        ) as mock_clf:
            mock_clf.is_loaded = True

            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get("/health")

        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "models_loaded" in data
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_health_status_is_healthy(self):
        """GET /health returns status: healthy."""
        from httpx import AsyncClient
        from backend.main import app

        with patch(
            "backend.models.classifier.emotion_classifier"
        ) as mock_clf:
            mock_clf.is_loaded = True

            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.get("/health")

        assert response.json()["status"] == "healthy"


# ── Classification Endpoint Tests ──────────────────────────────────────────

class TestClassifyEndpoint:

    @pytest.mark.asyncio
    async def test_classify_valid_input_returns_200(self):
        """POST /classify with valid input returns 200."""
        from httpx import AsyncClient
        from backend.main import app

        with patch(
            "backend.api.routes.orchestrator.process",
            new_callable=AsyncMock,
            return_value=MOCK_ORCHESTRATOR_RESULT
        ):
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/classify",
                    json={"text": "I feel happy today!"}
                )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_classify_response_has_all_keys(self):
        """POST /classify response contains all required keys."""
        from httpx import AsyncClient
        from backend.main import app

        required_keys = {
            "emotion",
            "confidence",
            "all_predictions",
            "recommendations",
            "crisis_assessment",
            "processing_time_ms",
            "timestamp",
            "model_version"
        }

        with patch(
            "backend.api.routes.orchestrator.process",
            new_callable=AsyncMock,
            return_value=MOCK_ORCHESTRATOR_RESULT
        ):
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/classify",
                    json={"text": "I feel happy today!"}
                )

        data = response.json()
        assert required_keys.issubset(data.keys())

    @pytest.mark.asyncio
    async def test_classify_empty_text_returns_422(self):
        """POST /classify with empty text returns 422."""
        from httpx import AsyncClient
        from backend.main import app

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/classify",
                json={"text": "   "}
            )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_classify_missing_text_field_returns_422(self):
        """POST /classify with missing text field returns 422."""
        from httpx import AsyncClient
        from backend.main import app

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/classify",
                json={}
            )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_classify_crisis_input_flags_correctly(self):
        """POST /classify with crisis text returns is_crisis=True."""
        from httpx import AsyncClient
        from backend.main import app

        with patch(
            "backend.api.routes.orchestrator.process",
            new_callable=AsyncMock,
            return_value=MOCK_CRISIS_RESULT
        ):
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/classify",
                    json={"text": "I feel hopeless and want to end my life"}
                )

        data = response.json()
        assert data["crisis_assessment"]["is_crisis"] is True
        assert data["crisis_assessment"]["risk_level"] in ["high", "critical"]

    @pytest.mark.asyncio
    async def test_classify_with_session_id(self):
        """POST /classify passes session_id through to response."""
        from httpx import AsyncClient
        from backend.main import app

        result_with_session = {
            **MOCK_ORCHESTRATOR_RESULT,
            "session_id": "test_session_99"
        }

        with patch(
            "backend.api.routes.orchestrator.process",
            new_callable=AsyncMock,
            return_value=result_with_session
        ):
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/classify",
                    json={
                        "text": "I feel okay",
                        "session_id": "test_session_99"
                    }
                )

        data = response.json()
        assert data["session_id"] == "test_session_99"

    @pytest.mark.asyncio
    async def test_classify_confidence_between_0_and_1(self):
        """Confidence score in response is between 0.0 and 1.0."""
        from httpx import AsyncClient
        from backend.main import app

        with patch(
            "backend.api.routes.orchestrator.process",
            new_callable=AsyncMock,
            return_value=MOCK_ORCHESTRATOR_RESULT
        ):
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/classify",
                    json={"text": "I feel great"}
                )

        data = response.json()
        assert 0.0 <= data["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_classify_text_too_long_returns_422(self):
        """POST /classify with text over 5000 chars returns 422."""
        from httpx import AsyncClient
        from backend.main import app

        long_text = "a" * 5001

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/classify",
                json={"text": long_text}
            )

        assert response.status_code == 422