import pytest
from httpx import AsyncClient
from backend.main import app


@pytest.mark.asyncio
async def test_health_check():
    """Test health endpoint returns 200."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


@pytest.mark.asyncio
async def test_classify_valid_input():
    """Test classification with valid text input."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/classify",
            json={"text": "I feel really happy and excited today!"}
        )
    assert response.status_code == 200
    data = response.json()
    assert "emotion" in data
    assert "confidence" in data
    assert "all_predictions" in data
    assert "crisis_assessment" in data
    assert "recommendations" in data
    assert 0.0 <= data["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_classify_empty_input():
    """Test classification rejects empty input."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/classify",
            json={"text": "   "}
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_classify_crisis_input():
    """Test crisis detection triggers for concerning text."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/classify",
            json={"text": "I feel hopeless and want to end my life"}
        )
    assert response.status_code == 200
    data = response.json()
    crisis = data["crisis_assessment"]
    assert crisis["is_crisis"] is True
    assert crisis["risk_level"] in ["high", "critical"]
    assert len(crisis["immediate_resources"]) > 0


@pytest.mark.asyncio
async def test_classify_with_explanations():
    """Test explanation generation when requested."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/classify",
            json={
                "text": "I am feeling very sad and lonely",
                "include_explanations": True
            }
        )
    assert response.status_code == 200
    data = response.json()
    assert data["explanations"] is not None
    assert isinstance(data["explanations"], list)