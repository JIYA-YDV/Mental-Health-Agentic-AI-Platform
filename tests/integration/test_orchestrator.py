# tests/integration/test_orchestrator.py
"""Test multi-agent orchestration flow."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


pytestmark = pytest.mark.integration


@pytest.fixture
def orchestrator_with_mocks():
    """Build an orchestrator where each agent is mocked."""
    with patch(
        "backend.agents.orchestrator.ClassificationAgent"
    ) as MockClf, patch(
        "backend.agents.orchestrator.CrisisAgent"
    ) as MockCrisis, patch(
        "backend.agents.orchestrator.RAGAgent"
    ) as MockRAG:

        # Set up async mocks
        MockClf.return_value.run = AsyncMock(return_value={
            "emotion": "sadness",
            "confidence": 0.85,
            "all_predictions": [
                {"label": "sadness", "score": 0.85},
                {"label": "joy", "score": 0.15},
            ],
        })
        MockCrisis.return_value.run = AsyncMock(return_value={
            "is_crisis": False,
            "risk_level": "low",
            "risk_score": 0.05,
            "crisis_indicators": [],
            "immediate_resources": ["988"],
        })
        MockRAG.return_value.run = AsyncMock(return_value=[
            {
                "title": "Test Tip",
                "content": "Breathe deeply",
                "relevance_score": 0.8,
                "category": "breathing",
                "source": "KB",
            }
        ])

        from backend.agents.orchestrator import AgentOrchestrator
        yield AgentOrchestrator()


@pytest.mark.asyncio
async def test_orchestrator_returns_all_required_fields(orchestrator_with_mocks):
    result = await orchestrator_with_mocks.process(
        text="I feel sad", session_id="abc"
    )
    for field in (
        "emotion", "confidence", "all_predictions",
        "recommendations", "crisis_assessment",
        "session_id", "processing_time_ms",
    ):
        assert field in result


@pytest.mark.asyncio
async def test_orchestrator_echoes_session_id(orchestrator_with_mocks):
    result = await orchestrator_with_mocks.process(
        text="hi", session_id="session-xyz"
    )
    assert result["session_id"] == "session-xyz"


@pytest.mark.asyncio
async def test_orchestrator_measures_latency(orchestrator_with_mocks):
    result = await orchestrator_with_mocks.process(text="hi")
    assert result["processing_time_ms"] > 0
    assert result["processing_time_ms"] < 10_000  # < 10 sec sanity check