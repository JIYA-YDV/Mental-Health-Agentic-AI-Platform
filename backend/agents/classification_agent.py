from typing import Dict, Any
import structlog
from backend.models.classifier import emotion_classifier

logger = structlog.get_logger(__name__)


class ClassificationAgent:
    """
    Agent responsible for emotion classification.
    Wraps the transformer model and adds agent-level logic
    including confidence thresholding and emotion mapping.
    """

    # Map model labels to human-friendly names
    EMOTION_DISPLAY_MAP = {
        "anger": "Anger",
        "disgust": "Disgust",
        "fear": "Fear / Anxiety",
        "joy": "Joy / Happiness",
        "neutral": "Neutral",
        "sadness": "Sadness / Depression",
        "surprise": "Surprise"
    }

    # Minimum confidence to trust primary prediction
    MIN_CONFIDENCE_THRESHOLD = 0.35

    async def run(self, text: str) -> Dict[str, Any]:
        """
        Execute classification on input text.

        Args:
            text: Cleaned user input

        Returns:
            Classification result with emotion, confidence, all scores
        """
        logger.info("ClassificationAgent running", text_length=len(text))

        try:
            result = emotion_classifier.predict(text)

            # Map to display names
            result["emotion"] = self.EMOTION_DISPLAY_MAP.get(
                result["emotion"].lower(),
                result["emotion"]
            )

            # Flag low confidence predictions
            if result["confidence"] < self.MIN_CONFIDENCE_THRESHOLD:
                logger.warning(
                    "Low confidence prediction",
                    confidence=result["confidence"],
                    emotion=result["emotion"]
                )
                result["low_confidence"] = True
            else:
                result["low_confidence"] = False

            logger.info(
                "ClassificationAgent complete",
                emotion=result["emotion"],
                confidence=result["confidence"]
            )
            return result

        except Exception as e:
            logger.error("ClassificationAgent failed", error=str(e))
            raise