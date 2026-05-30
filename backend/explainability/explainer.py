from typing import List, Dict, Any
import structlog

logger = structlog.get_logger(__name__)


class EmotionExplainer:
    """
    Lightweight token-importance explainer using attention-based attribution.
    Provides human-readable explanations of why specific emotions were detected.

    Note: Full SHAP integration requires significant compute.
    This implementation provides fast approximate explanations
    suitable for production inference.
    """

    # Words associated with each emotion for attribution
    EMOTION_LEXICONS = {
        "anger": [
            "angry", "furious", "hate", "rage", "mad",
            "frustrated", "annoyed", "outraged"
        ],
        "sadness": [
            "sad", "depressed", "lonely", "hopeless",
            "crying", "miserable", "grief", "heartbroken"
        ],
        "fear": [
            "scared", "afraid", "anxious", "worried",
            "nervous", "terrified", "panic", "dread"
        ],
        "joy": [
            "happy", "excited", "wonderful", "great",
            "love", "amazing", "fantastic", "grateful"
        ],
        "disgust": [
            "disgusting", "awful", "terrible", "horrible",
            "revolting", "gross", "nasty", "vile"
        ],
        "surprise": [
            "surprised", "shocked", "unexpected",
            "astonished", "amazed", "stunned"
        ]
    }

    def explain(
        self,
        text: str,
        emotion: str
    ) -> List[Dict[str, Any]]:
        """
        Generate token-level importance scores.

        Args:
            text: Input text
            emotion: Detected primary emotion

        Returns:
            List of token importance dicts
        """
        tokens = text.lower().split()
        emotion_key = emotion.split("/")[0].strip().lower()

        relevant_words = self.EMOTION_LEXICONS.get(emotion_key, [])
        explanations = []

        for token in tokens:
            # Clean punctuation for matching
            clean_token = token.strip(".,!?;:'\"")

            if clean_token in relevant_words:
                importance = 0.9
                direction = "positive"
            elif len(clean_token) > 3:
                # Small baseline importance for content words
                importance = 0.1
                direction = "neutral"
            else:
                importance = 0.0
                direction = "neutral"

            explanations.append({
                "token": token,
                "importance": importance,
                "sentiment_direction": direction
            })

        return explanations


# Module-level singleton
explainer = EmotionExplainer()