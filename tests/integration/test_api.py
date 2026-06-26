# tests/integration/test_api.py
"""Integration tests for FastAPI endpoints with mocked ML models."""
import pytest


pytestmark = pytest.mark.integration


class TestHealthEndpoint:

    def test_health_returns_200(self, api_client):
        response = api_client.get("/health")
        assert response.status_code == 200

    def test_health_payload_shape(self, api_client):
        response = api_client.get("/health")
        body = response.json()
        assert "status" in body
        assert "version" in body
        assert "models_loaded" in body
        assert body["status"] in ("healthy", "degraded", "unhealthy")


class TestClassifyEndpoint:

    def test_empty_text_returns_422(self, api_client):
        response = api_client.post("/classify", json={"text": ""})
        assert response.status_code == 422

    def test_whitespace_text_returns_422(self, api_client):
        response = api_client.post("/classify", json={"text": "   "})
        assert response.status_code == 422

    def test_missing_text_returns_422(self, api_client):
        response = api_client.post("/classify", json={})
        assert response.status_code == 422

    def test_too_long_text_returns_422(self, api_client):
        response = api_client.post(
            "/classify", json={"text": "x" * 5001}
        )
        assert response.status_code == 422

    def test_valid_request_returns_200(self, api_client, sample_text_sad):
        response = api_client.post(
            "/classify",
            json={"text": sample_text_sad},
        )
        assert response.status_code == 200

    def test_response_has_required_fields(self, api_client, sample_text_sad):
        response = api_client.post(
            "/classify",
            json={"text": sample_text_sad},
        )
        body = response.json()
        for field in (
            "emotion", "confidence", "all_predictions",
            "recommendations", "crisis_assessment",
            "processing_time_ms", "model_version",
        ):
            assert field in body, f"Missing field: {field}"

    def test_explanations_returned_when_requested(
        self, api_client, sample_text_sad
    ):
        response = api_client.post(
            "/classify",
            json={"text": sample_text_sad, "include_explanations": True},
        )
        body = response.json()
        # Explanations may be empty list, but field must exist
        assert "explanations" in body
        assert body["explanations"] is not None

    def test_explanations_omitted_when_not_requested(
        self, api_client, sample_text_sad
    ):
        response = api_client.post(
            "/classify",
            json={"text": sample_text_sad, "include_explanations": False},
        )
        body = response.json()
        assert body.get("explanations") is None

    def test_session_id_echoed_back(self, api_client, sample_text_sad):
        response = api_client.post(
            "/classify",
            json={"text": sample_text_sad, "session_id": "test-session-123"},
        )
        body = response.json()
        assert body["session_id"] == "test-session-123"

    def test_confidence_in_valid_range(self, api_client, sample_text_sad):
        response = api_client.post(
            "/classify", json={"text": sample_text_sad}
        )
        body = response.json()
        assert 0.0 <= body["confidence"] <= 1.0