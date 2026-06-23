import asyncio
import time
from typing import Dict, Any
import structlog
from backend.agents.classification_agent import ClassificationAgent
from backend.agents.crisis_agent import CrisisAgent
from backend.agents.rag_agent import RAGAgent
from backend.agents.semantic_interpreter import SemanticInterpreter

logger = structlog.get_logger(__name__)


class AgentOrchestrator:
    """
    Multi-agent orchestration layer.

    Execution flow:
    1. ClassificationAgent → detect emotion (required, blocks downstream)
    2. CrisisAgent + RAGAgent → run in parallel (non-blocking to each other)
    3. Aggregate all results into unified response

    This design minimizes latency by parallelizing independent agents.
    """

    def __init__(self):
        self.classification_agent = ClassificationAgent()
        self.crisis_agent = CrisisAgent()
        self.rag_agent = RAGAgent()

    async def process(
        self,
        text: str,
        session_id: str = None
    ) -> Dict[str, Any]:
        """
        Orchestrate all agents and return aggregated result.

        Args:
            text: Validated user input
            session_id: Optional session identifier

        Returns:
            Complete analysis result dict
        """
        start_time = time.time()
        logger.info(
            "Orchestrator starting",
            text_length=len(text),
            session_id=session_id
        )

        # ── Step 1: Classification (required first) ──────────────────────
        classification_result = await self.classification_agent.run(text)

        primary_emotion = classification_result["emotion"]

        # ── Step 2: Crisis + RAG in parallel ─────────────────────────────
        crisis_task = self.crisis_agent.run(text, classification_result)
        rag_task = self.rag_agent.run(text, primary_emotion)

        crisis_result, recommendations = await asyncio.gather(
            crisis_task,
            rag_task,
            return_exceptions=False
        )

        # ── Step 3: Aggregate results ─────────────────────────────────────
        processing_time_ms = round((time.time() - start_time) * 1000, 2)

        aggregated = {
            # Classification outputs
            "emotion": classification_result["emotion"],
            "confidence": classification_result["confidence"],
            "all_predictions": classification_result["all_predictions"],

            # RAG outputs
            "recommendations": recommendations,

            # Crisis outputs
            "crisis_assessment": crisis_result,

            # Metadata
            "session_id": session_id,
            "processing_time_ms": processing_time_ms
        }

        logger.info(
            "Orchestrator complete",
            emotion=primary_emotion,
            processing_time_ms=processing_time_ms,
            is_crisis=crisis_result["is_crisis"]
        )

        return aggregated