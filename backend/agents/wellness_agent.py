"""
Wellness Agent

Provides personalized wellness tips and coping strategies
based on detected emotion and user context.
"""

from typing import Dict, Any, List
import structlog

logger = structlog.get_logger(__name__)


class WellnessAgent:
    """
    Agent that generates personalized wellness recommendations
    based on the detected emotion and confidence level.
    """

    WELLNESS_TIPS = {
        "joy / happiness": [
            "Savour this positive moment by journaling about it.",
            "Share your happiness with someone you care about.",
            "Use this energy to tackle a goal you have been postponing.",
            "Practice gratitude to amplify positive emotions."
        ],
        "sadness / depression": [
            "Allow yourself to feel without judgment.",
            "Reach out to one trusted person today.",
            "Take a short walk outside — sunlight helps mood.",
            "Write down three small things that went okay today."
        ],
        "fear / anxiety": [
            "Try box breathing: inhale 4s, hold 4s, exhale 4s, hold 4s.",
            "Ground yourself with the 5-4-3-2-1 sensory technique.",
            "Challenge anxious thoughts by asking: is this fact or fear?",
            "Limit caffeine and focus on slow, steady breathing."
        ],
        "anger": [
            "Step away from the situation for 10 minutes.",
            "Physical exercise helps discharge anger safely.",
            "Write your feelings in a journal before responding.",
            "Use the STOP technique: Stop, Take a breath, Observe, Proceed."
        ],
        "disgust": [
            "Identify the specific trigger and examine your values.",
            "Practice compassionate perspective-taking.",
            "Redirect energy into something constructive.",
            "Speak to someone you trust about your feelings."
        ],
        "surprise": [
            "Give yourself time to process before reacting.",
            "Write down your initial thoughts and feelings.",
            "Seek more information before drawing conclusions.",
            "Use mindful breathing to settle your nervous system."
        ],
        "neutral": [
            "This is a great time to set an intention for your day.",
            "Practice mindfulness to stay present.",
            "Check in with your physical needs: water, food, rest.",
            "Connect with someone you care about."
        ]
    }

    DEFAULT_TIPS = [
        "Take three slow deep breaths right now.",
        "Drink a glass of water and notice how you feel.",
        "Step outside for five minutes of fresh air.",
        "Write one sentence about how you are feeling."
    ]

    async def run(
        self,
        emotion: str,
        confidence: float
    ) -> Dict[str, Any]:
        """
        Generate wellness tips based on detected emotion.

        Args:
            emotion: Detected primary emotion label
            confidence: Model confidence score

        Returns:
            Dict with wellness tips and self-care actions
        """
        logger.info(
            "WellnessAgent running",
            emotion=emotion,
            confidence=round(confidence, 3)
        )

        emotion_key = emotion.lower()

        # Match emotion to tip category
        tips = self.WELLNESS_TIPS.get(emotion_key, self.DEFAULT_TIPS)

        # Select tips based on confidence level
        # Higher confidence → more targeted recommendations
        if confidence >= 0.7:
            selected_tips = tips[:3]
            tip_quality = "high_confidence"
        elif confidence >= 0.4:
            selected_tips = tips[:2]
            tip_quality = "medium_confidence"
        else:
            selected_tips = [tips[0]] + [self.DEFAULT_TIPS[0]]
            tip_quality = "low_confidence"

        result = {
            "wellness_tips": selected_tips,
            "tip_quality": tip_quality,
            "emotion_addressed": emotion,
            "self_care_reminder": (
                "Remember: small consistent actions "
                "build lasting mental wellbeing."
            )
        }

        logger.info(
            "WellnessAgent complete",
            tips_count=len(selected_tips),
            quality=tip_quality
        )

        return result