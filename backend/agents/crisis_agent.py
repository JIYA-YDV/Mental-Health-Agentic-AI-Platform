from typing import Dict, Any, List
import structlog
from backend.config.settings import settings

logger = structlog.get_logger(__name__)


class CrisisAgent:
    """
    Specialized agent for crisis detection and risk assessment.
    Analyzes both keyword signals and emotion confidence scores
    to determine risk level and appropriate intervention resources.

    IMPORTANT: This is an AI tool and NOT a substitute for
    professional mental health care or emergency services.
    """

    CRISIS_RESOURCES = [
        "🆘 National Suicide Prevention Lifeline: Call or text 988",
        "💬 Crisis Text Line: Text HOME to 741741",
        "🏥 Emergency Services: Call 911",
        "🌍 International Crisis Resources: https://www.iasp.info/resources/Crisis_Centres/"
    ]

    HIGH_RISK_EMOTIONS = {"sadness", "fear"}
    MEDIUM_RISK_EMOTIONS = {"anger", "disgust"}

    async def run(
        self,
        text: str,
        classification_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess crisis risk from text and classification.

        Args:
            text: Original user input
            classification_result: Output from ClassificationAgent

        Returns:
            Crisis assessment with risk level and resources
        """
        logger.info("CrisisAgent running")

        text_lower = text.lower()
        crisis_indicators = []

        # 1. Keyword detection
        detected_keywords = [
            kw for kw in settings.CRISIS_KEYWORDS
            if kw in text_lower
        ]
        crisis_indicators.extend(detected_keywords)

        # 2. Emotion-based risk factors
        primary_emotion = classification_result.get("emotion", "").lower()
        confidence = classification_result.get("confidence", 0.0)

        emotion_risk = False
        if (primary_emotion in self.HIGH_RISK_EMOTIONS
                and confidence > settings.CRISIS_CONFIDENCE_THRESHOLD):
            emotion_risk = True
            crisis_indicators.append(
                f"High confidence {primary_emotion} detected"
            )

        # 3. Determine risk level
        risk_level, risk_score, is_crisis = self._calculate_risk(
            keyword_count=len(detected_keywords),
            emotion_risk=emotion_risk,
            confidence=confidence
        )

        # 4. Select appropriate resources
        immediate_resources = (
            self.CRISIS_RESOURCES if is_crisis
            else self.CRISIS_RESOURCES[:2]  # Always show helpline info
        )

        result = {
            "is_crisis": is_crisis,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "crisis_indicators": crisis_indicators,
            "immediate_resources": immediate_resources
        }

        logger.info(
            "CrisisAgent complete",
            risk_level=risk_level,
            is_crisis=is_crisis
        )
        return result

    def _calculate_risk(
        self,
        keyword_count: int,
        emotion_risk: bool,
        confidence: float
    ) -> tuple:
        """
        Calculate risk level from multiple signals.

        Returns:
            Tuple of (risk_level, risk_score, is_crisis)
        """
        # Base score from keywords
        keyword_score = min(keyword_count * 0.3, 0.9)

        # Emotion contribution
        emotion_score = 0.3 if emotion_risk else 0.0

        # Combined risk score
        risk_score = min(keyword_score + emotion_score, 1.0)

        # Determine level
        if risk_score >= 0.7 or keyword_count >= 2:
            return "critical", risk_score, True
        elif risk_score >= 0.4 or keyword_count >= 1:
            return "high", risk_score, True
        elif risk_score >= 0.2 or emotion_risk:
            return "medium", risk_score, False
        else:
            return "low", risk_score, False