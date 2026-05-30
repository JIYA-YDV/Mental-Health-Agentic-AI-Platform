"""
Agent Unit Tests

Tests for ClassificationAgent, CrisisAgent,
RAGAgent, WellnessAgent, and AgentOrchestrator.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock


# ── Classification Agent Tests ─────────────────────────────────────────────

class TestClassificationAgent:

    @pytest.mark.asyncio
    async def test_run_returns_required_keys(self):
        """ClassificationAgent.run() returns all required keys."""
        from backend.agents.classification_agent import ClassificationAgent

        agent = ClassificationAgent()

        mock_result = {
            "emotion": "joy",
            "confidence": 0.92,
            "all_predictions": [
                {"label": "joy", "score": 0.92},
                {"label": "neutral", "score": 0.05}
            ]
        }

        with patch(
            "backend.agents.classification_agent.emotion_classifier"
        ) as mock_clf:
            mock_clf.predict.return_value = mock_result
            result = await agent.run("I feel great today!")

        assert "emotion" in result
        assert "confidence" in result
        assert "all_predictions" in result
        assert "low_confidence" in result

    @pytest.mark.asyncio
    async def test_emotion_label_mapped_to_display_name(self):
        """Raw model labels are mapped to human-friendly display names."""
        from backend.agents.classification_agent import ClassificationAgent

        agent = ClassificationAgent()

        mock_result = {
            "emotion": "fear",
            "confidence": 0.85,
            "all_predictions": [{"label": "fear", "score": 0.85}]
        }

        with patch(
            "backend.agents.classification_agent.emotion_classifier"
        ) as mock_clf:
            mock_clf.predict.return_value = mock_result
            result = await agent.run("I am terrified")

        assert result["emotion"] == "Fear / Anxiety"

    @pytest.mark.asyncio
    async def test_low_confidence_flag_set(self):
        """Low confidence predictions are flagged."""
        from backend.agents.classification_agent import ClassificationAgent

        agent = ClassificationAgent()

        mock_result = {
            "emotion": "neutral",
            "confidence": 0.20,  # Below MIN_CONFIDENCE_THRESHOLD
            "all_predictions": [{"label": "neutral", "score": 0.20}]
        }

        with patch(
            "backend.agents.classification_agent.emotion_classifier"
        ) as mock_clf:
            mock_clf.predict.return_value = mock_result
            result = await agent.run("maybe")

        assert result["low_confidence"] is True

    @pytest.mark.asyncio
    async def test_high_confidence_flag_not_set(self):
        """High confidence predictions are not flagged."""
        from backend.agents.classification_agent import ClassificationAgent

        agent = ClassificationAgent()

        mock_result = {
            "emotion": "joy",
            "confidence": 0.95,
            "all_predictions": [{"label": "joy", "score": 0.95}]
        }

        with patch(
            "backend.agents.classification_agent.emotion_classifier"
        ) as mock_clf:
            mock_clf.predict.return_value = mock_result
            result = await agent.run("I am so happy!")

        assert result["low_confidence"] is False


# ── Crisis Agent Tests ─────────────────────────────────────────────────────

class TestCrisisAgent:

    @pytest.mark.asyncio
    async def test_clean_text_returns_low_risk(self):
        """Normal text produces low risk assessment."""
        from backend.agents.crisis_agent import CrisisAgent

        agent = CrisisAgent()
        classification = {"emotion": "Joy / Happiness", "confidence": 0.9}

        result = await agent.run(
            "I had a wonderful day at the park",
            classification
        )

        assert result["risk_level"] == "low"
        assert result["is_crisis"] is False
        assert result["risk_score"] < 0.3

    @pytest.mark.asyncio
    async def test_crisis_keyword_triggers_alert(self):
        """Crisis keywords trigger high or critical risk."""
        from backend.agents.crisis_agent import CrisisAgent

        agent = CrisisAgent()
        classification = {"emotion": "Sadness / Depression", "confidence": 0.88}

        result = await agent.run(
            "I feel hopeless and want to end my life",
            classification
        )

        assert result["is_crisis"] is True
        assert result["risk_level"] in ["high", "critical"]
        assert len(result["crisis_indicators"]) > 0

    @pytest.mark.asyncio
    async def test_crisis_resources_always_present(self):
        """Immediate resources are always included in response."""
        from backend.agents.crisis_agent import CrisisAgent

        agent = CrisisAgent()
        classification = {"emotion": "Neutral", "confidence": 0.6}

        result = await agent.run("I am fine", classification)

        assert "immediate_resources" in result
        assert isinstance(result["immediate_resources"], list)
        assert len(result["immediate_resources"]) > 0

    @pytest.mark.asyncio
    async def test_risk_score_is_float_between_0_and_1(self):
        """Risk score is always between 0.0 and 1.0."""
        from backend.agents.crisis_agent import CrisisAgent

        agent = CrisisAgent()
        classification = {"emotion": "Anger", "confidence": 0.75}

        result = await agent.run("I am so angry right now", classification)

        assert isinstance(result["risk_score"], float)
        assert 0.0 <= result["risk_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_multiple_keywords_raise_risk_level(self):
        """Multiple crisis keywords increase risk level."""
        from backend.agents.crisis_agent import CrisisAgent

        agent = CrisisAgent()
        classification = {"emotion": "Sadness / Depression", "confidence": 0.9}

        result = await agent.run(
            "I feel hopeless and want to harm myself",
            classification
        )

        assert result["risk_level"] in ["high", "critical"]
        assert result["risk_score"] >= 0.4


# ── RAG Agent Tests ────────────────────────────────────────────────────────

class TestRAGAgent:

    @pytest.mark.asyncio
    async def test_run_returns_list(self):
        """RAGAgent.run() always returns a list."""
        from backend.agents.rag_agent import RAGAgent

        agent = RAGAgent()

        with patch(
            "backend.agents.rag_agent.rag_pipeline"
        ) as mock_rag:
            mock_rag.retrieve.return_value = [
                {
                    "title": "Breathing Exercise",
                    "content": "Try deep breathing.",
                    "relevance_score": 0.85,
                    "category": "breathing",
                    "source": "Knowledge Base"
                }
            ]
            result = await agent.run(
                "I feel anxious",
                "Fear / Anxiety"
            )

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_run_returns_empty_list_on_failure(self):
        """RAGAgent returns empty list on pipeline failure, not exception."""
        from backend.agents.rag_agent import RAGAgent

        agent = RAGAgent()

        with patch(
            "backend.agents.rag_agent.rag_pipeline"
        ) as mock_rag:
            mock_rag.retrieve.side_effect = RuntimeError("DB error")
            result = await agent.run("I feel bad", "Sadness")

        assert result == []

    @pytest.mark.asyncio
    async def test_recommendation_has_required_keys(self):
        """Each recommendation contains all required fields."""
        from backend.agents.rag_agent import RAGAgent

        agent = RAGAgent()
        required_keys = {"title", "content", "relevance_score", "category", "source"}

        with patch(
            "backend.agents.rag_agent.rag_pipeline"
        ) as mock_rag:
            mock_rag.retrieve.return_value = [
                {
                    "title": "Test",
                    "content": "Test content",
                    "relevance_score": 0.9,
                    "category": "coping",
                    "source": "Knowledge Base"
                }
            ]
            result = await agent.run("test", "Neutral")

        assert len(result) > 0
        for rec in result:
            assert required_keys.issubset(rec.keys())


# ── Wellness Agent Tests ───────────────────────────────────────────────────

class TestWellnessAgent:

    @pytest.mark.asyncio
    async def test_run_returns_tips(self):
        """WellnessAgent returns wellness tips."""
        from backend.agents.wellness_agent import WellnessAgent

        agent = WellnessAgent()
        result = await agent.run("Joy / Happiness", 0.9)

        assert "wellness_tips" in result
        assert isinstance(result["wellness_tips"], list)
        assert len(result["wellness_tips"]) > 0

    @pytest.mark.asyncio
    async def test_high_confidence_returns_more_tips(self):
        """High confidence produces more targeted tips."""
        from backend.agents.wellness_agent import WellnessAgent

        agent = WellnessAgent()
        result = await agent.run("Sadness / Depression", 0.85)

        assert result["tip_quality"] == "high_confidence"
        assert len(result["wellness_tips"]) == 3

    @pytest.mark.asyncio
    async def test_low_confidence_returns_fewer_tips(self):
        """Low confidence produces fewer, more generic tips."""
        from backend.agents.wellness_agent import WellnessAgent

        agent = WellnessAgent()
        result = await agent.run("Neutral", 0.25)

        assert result["tip_quality"] == "low_confidence"
        assert len(result["wellness_tips"]) == 2

    @pytest.mark.asyncio
    async def test_unknown_emotion_returns_default_tips(self):
        """Unknown emotion falls back to default tips."""
        from backend.agents.wellness_agent import WellnessAgent

        agent = WellnessAgent()
        result = await agent.run("unknown_emotion_xyz", 0.8)

        assert "wellness_tips" in result
        assert len(result["wellness_tips"]) > 0

    @pytest.mark.asyncio
    async def test_self_care_reminder_present(self):
        """Self-care reminder is always included."""
        from backend.agents.wellness_agent import WellnessAgent

        agent = WellnessAgent()
        result = await agent.run("Anger", 0.7)

        assert "self_care_reminder" in result
        assert isinstance(result["self_care_reminder"], str)


# ── Orchestrator Tests ─────────────────────────────────────────────────────

class TestAgentOrchestrator:

    @pytest.mark.asyncio
    async def test_process_returns_all_required_keys(self):
        """Orchestrator output contains all required response keys."""
        from backend.agents.orchestrator import AgentOrchestrator

        orchestrator = AgentOrchestrator()

        required_keys = {
            "emotion",
            "confidence",
            "all_predictions",
            "recommendations",
            "crisis_assessment",
            "processing_time_ms"
        }

        mock_classification = {
            "emotion": "Joy / Happiness",
            "confidence": 0.92,
            "all_predictions": [{"label": "joy", "score": 0.92}],
            "low_confidence": False
        }
        mock_crisis = {
            "is_crisis": False,
            "risk_level": "low",
            "risk_score": 0.05,
            "crisis_indicators": [],
            "immediate_resources": ["988 Lifeline"]
        }
        mock_rag = [
            {
                "title": "Breathing",
                "content": "Breathe deeply.",
                "relevance_score": 0.8,
                "category": "breathing",
                "source": "KB"
            }
        ]

        with patch.object(
            orchestrator.classification_agent, "run",
            return_value=mock_classification
        ), patch.object(
            orchestrator.crisis_agent, "run",
            return_value=mock_crisis
        ), patch.object(
            orchestrator.rag_agent, "run",
            return_value=mock_rag
        ):
            result = await orchestrator.process("I feel great!")

        assert required_keys.issubset(result.keys())

    @pytest.mark.asyncio
    async def test_processing_time_is_positive(self):
        """Processing time is always a positive number."""
        from backend.agents.orchestrator import AgentOrchestrator

        orchestrator = AgentOrchestrator()

        mock_classification = {
            "emotion": "Neutral",
            "confidence": 0.6,
            "all_predictions": [],
            "low_confidence": False
        }
        mock_crisis = {
            "is_crisis": False,
            "risk_level": "low",
            "risk_score": 0.0,
            "crisis_indicators": [],
            "immediate_resources": []
        }

        with patch.object(
            orchestrator.classification_agent, "run",
            return_value=mock_classification
        ), patch.object(
            orchestrator.crisis_agent, "run",
            return_value=mock_crisis
        ), patch.object(
            orchestrator.rag_agent, "run",
            return_value=[]
        ):
            result = await orchestrator.process("test input")

        assert result["processing_time_ms"] > 0

    @pytest.mark.asyncio
    async def test_session_id_passed_through(self):
        """Session ID is preserved in orchestrator output."""
        from backend.agents.orchestrator import AgentOrchestrator

        orchestrator = AgentOrchestrator()

        mock_classification = {
            "emotion": "Neutral",
            "confidence": 0.6,
            "all_predictions": [],
            "low_confidence": False
        }
        mock_crisis = {
            "is_crisis": False,
            "risk_level": "low",
            "risk_score": 0.0,
            "crisis_indicators": [],
            "immediate_resources": []
        }

        with patch.object(
            orchestrator.classification_agent, "run",
            return_value=mock_classification
        ), patch.object(
            orchestrator.crisis_agent, "run",
            return_value=mock_crisis
        ), patch.object(
            orchestrator.rag_agent, "run",
            return_value=[]
        ):
            result = await orchestrator.process(
                "test",
                session_id="test_session_42"
            )

        assert result["session_id"] == "test_session_42"