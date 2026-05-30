from typing import Dict, Any, List
import structlog
from backend.models.rag_pipeline import rag_pipeline

logger = structlog.get_logger(__name__)


class RAGAgent:
    """
    Agent responsible for retrieving wellness recommendations.
    Uses the RAG pipeline to find contextually relevant resources
    based on user text and detected emotion.
    """

    async def run(
        self,
        text: str,
        emotion: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant wellness recommendations.

        Args:
            text: User input text
            emotion: Detected primary emotion

        Returns:
            List of wellness recommendation dicts
        """
        logger.info("RAGAgent running", emotion=emotion)

        try:
            recommendations = rag_pipeline.retrieve(
                query_text=text,
                emotion=emotion
            )

            logger.info(
                "RAGAgent complete",
                recommendations_count=len(recommendations)
            )
            return recommendations

        except Exception as e:
            logger.error("RAGAgent failed", error=str(e))
            # Return empty list rather than crashing pipeline
            return []