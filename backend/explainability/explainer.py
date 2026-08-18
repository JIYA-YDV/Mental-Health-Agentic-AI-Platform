# backend/explainability/explainer.py
"""
Lightweight token-level emotion attribution module.

Uses curated lexicon scoring as a fast alternative to SHAP.
Suitable for real-time inference on consumer hardware.

Public exports:
    - explainer         : singleton instance
    - emotion_explainer : backward-compatible alias
    - EmotionExplainer  : class for testing
    - TokenAttribution  : dataclass
"""
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import structlog

from backend.config.settings import settings

logger = structlog.get_logger(__name__)


@dataclass
class TokenAttribution:
    """Represents influence score for a single token."""
    token: str
    weight: float
    emotion_category: str

    def __post_init__(self):
        self.token = self.token.lower().strip()


class EmotionExplainer:
    """
    Token-level emotion attribution via curated lexicon scoring.
    Weights: 0.1 (weak) → 1.0 (strong diagnostic indicator).
    """

    EMOTION_LEXICONS: Dict[str, Dict[str, float]] = {
        # ═══════════════════════════════════════════════════════════════
        # SADNESS
        # ═══════════════════════════════════════════════════════════════
        "sadness": {
            "sad": 1.0, "depressed": 1.0, "depression": 1.0,
            "lonely": 0.9, "loneliness": 0.9, "crying": 0.95, "tears": 0.85,
            "hopeless": 1.0, "worthless": 1.0, "empty": 0.8, "numb": 0.75,
            "down": 0.6, "grief": 0.9, "grieving": 0.9, "loss": 0.75,
            "miss": 0.6, "missing": 0.6, "alone": 0.7, "isolated": 0.8,
            "exhausted": 0.8, "tired": 0.7, "fatigue": 0.7, "drained": 0.65,
            "heavy": 0.4, "burden": 0.65, "dark": 0.55,
            "meaningless": 0.9, "pointless": 0.85, "useless": 0.8,
            "failure": 0.9, "failing": 0.8, "failed": 0.75,
            "stuck": 0.75, "trapped": 0.85, "behind": 0.55,
            "nothing": 0.5, "cannot": 0.4, "never": 0.35,
        },

        # ═══════════════════════════════════════════════════════════════
        # FEAR / ANXIETY (unified — classifier calls both "fear")
        # ═══════════════════════════════════════════════════════════════
        "fear": {
            # Direct anxiety terms
            "anxious": 1.0, "anxiety": 1.0, "worried": 0.9, "worrying": 0.9,
            "worry": 0.85, "nervous": 0.8, "nervousness": 0.8,
            "overwhelm": 0.9, "overwhelmed": 0.9, "overwhelming": 0.9,
            "panic": 1.0, "panicked": 1.0, "panicking": 1.0,

            # Direct fear terms
            "afraid": 1.0, "scared": 1.0, "terrified": 1.0, "fear": 1.0,
            "fearful": 0.95, "phobia": 0.9, "dread": 0.9, "horror": 0.8,
            "frightened": 0.95, "freaked": 0.75,

            # Physical anxiety markers
            "heart": 0.5, "pounding": 0.75, "racing": 0.8, "breath": 0.6,
            "breathless": 0.75, "chest": 0.55, "shaking": 0.7,
            "trembling": 0.7, "sweating": 0.65, "dizzy": 0.6,
            "restless": 0.7, "insomnia": 0.75,

            # Cognitive anxiety
            "future": 0.5, "uncertain": 0.75, "uncertainty": 0.75,
            "unknown": 0.6, "deadline": 0.7, "pressure": 0.75,
            "stress": 0.7, "stressful": 0.75, "stressed": 0.75,
            "interview": 0.65, "exam": 0.6, "test": 0.4,
            "tomorrow": 0.4, "presentation": 0.55,

            # Safety threats
            "dangerous": 0.85, "unsafe": 0.8, "threat": 0.75,
            "threatened": 0.8, "attack": 0.7, "hurt": 0.55,

            # Loss-related
            "abandonment": 0.9, "die": 0.7, "death": 0.65,
        },

        # ═══════════════════════════════════════════════════════════════
        # ANGER
        # ═══════════════════════════════════════════════════════════════
        "anger": {
            "angry": 1.0, "anger": 1.0, "rage": 0.95, "raging": 0.95,
            "furious": 0.95, "fury": 0.9, "irritated": 0.75,
            "irritating": 0.7, "frustrated": 0.85, "frustration": 0.85,
            "mad": 0.7, "annoyed": 0.7, "annoying": 0.65, "upset": 0.55,
            "hate": 0.85, "hatred": 0.85, "resent": 0.75, "resentment": 0.75,
            "bitter": 0.7, "bitterness": 0.7,
            "unfair": 0.65, "unjust": 0.65, "wrong": 0.4,
            "ignored": 0.65, "dismissed": 0.6, "disrespected": 0.75,
            "damn": 0.4, "scream": 0.7, "yelling": 0.7, "shouting": 0.7,
            "tired": 0.35,
        },

        # ═══════════════════════════════════════════════════════════════
        # JOY (new!)
        # ═══════════════════════════════════════════════════════════════
        "joy": {
            "happy": 1.0, "happiness": 1.0, "joy": 1.0, "joyful": 0.95,
            "excited": 0.9, "excitement": 0.9, "thrilled": 0.95, "thrilling": 0.9,
            "delighted": 0.9, "delight": 0.9, "cheerful": 0.85, "elated": 0.95,
            "elation": 0.95, "wonderful": 0.85, "amazing": 0.8, "awesome": 0.75,
            "great": 0.6, "fantastic": 0.85, "brilliant": 0.75,
            "grateful": 0.9, "gratitude": 0.9, "thankful": 0.9, "appreciate": 0.7,
            "blessed": 0.85, "lucky": 0.75, "fortunate": 0.75,
            "proud": 0.9, "pride": 0.85, "accomplished": 0.9,
            "achievement": 0.85, "success": 0.85, "successful": 0.85,
            "promotion": 0.9, "promoted": 0.9, "won": 0.85, "winning": 0.85,
            "celebrate": 0.8, "celebrating": 0.8, "celebration": 0.8,
            "smile": 0.75, "smiling": 0.75, "laugh": 0.75, "laughing": 0.75,
            "enjoy": 0.7, "enjoyed": 0.7, "enjoying": 0.7,
            "good": 0.5, "wonderful": 0.8, "positive": 0.65,
            "paid": 0.5, "finally": 0.55, "hard work": 0.6,
        },

        # ═══════════════════════════════════════════════════════════════
        # LOVE (new!)
        # ═══════════════════════════════════════════════════════════════
        "love": {
            "love": 1.0, "loved": 1.0, "loving": 1.0, "lovely": 0.85,
            "adore": 0.95, "adoring": 0.95, "cherish": 0.9, "cherished": 0.9,
            "care": 0.7, "caring": 0.75, "cared": 0.7,
            "affection": 0.85, "affectionate": 0.85, "warm": 0.6, "warmth": 0.65,
            "connection": 0.8, "connected": 0.8, "bond": 0.75, "bonded": 0.75,
            "close": 0.6, "closeness": 0.7, "intimate": 0.85, "intimacy": 0.85,
            "family": 0.75, "friend": 0.6, "friendship": 0.7, "partner": 0.7,
            "together": 0.65, "sharing": 0.6, "shared": 0.6,
            "grateful": 0.7, "appreciate": 0.65, "reminded": 0.55,
            "heart": 0.55, "soul": 0.6, "belong": 0.75, "belonging": 0.75,
            "support": 0.6, "supported": 0.65, "kindness": 0.7, "kind": 0.6,
        },

        # ═══════════════════════════════════════════════════════════════
        # SURPRISE (new!)
        # ═══════════════════════════════════════════════════════════════
        "surprise": {
            "surprised": 1.0, "surprise": 1.0, "surprising": 0.95,
            "shocked": 0.95, "shock": 0.9, "shocking": 0.9,
            "astonished": 0.9, "astonishing": 0.9, "amazed": 0.85, "amazing": 0.7,
            "unexpected": 0.85, "unexpectedly": 0.85, "suddenly": 0.75, "sudden": 0.7,
            "wow": 0.75, "whoa": 0.75, "unbelievable": 0.85,
            "cannot believe": 0.85, "cant believe": 0.85, "didnt expect": 0.85,
            "surprised": 0.9, "startled": 0.85, "stunned": 0.9,
            "speechless": 0.85, "surprised": 0.9,
            "out of nowhere": 0.8, "surprise": 1.0,
        },

        # ═══════════════════════════════════════════════════════════════
        # DISGUST (new!)
        # ═══════════════════════════════════════════════════════════════
        "disgust": {
            "disgusted": 1.0, "disgust": 1.0, "disgusting": 0.95,
            "revolted": 0.9, "revolting": 0.9, "gross": 0.85,
            "nauseated": 0.85, "nauseous": 0.85, "sick": 0.7, "sickened": 0.85,
            "repulsed": 0.9, "repulsive": 0.9, "vile": 0.85,
            "awful": 0.75, "terrible": 0.65, "horrible": 0.75,
            "hate": 0.7, "hated": 0.7, "cannot stand": 0.8,
            "yuck": 0.85, "eww": 0.85,
        },
    }

    def __init__(self):
        self._compiled_patterns: Dict[str, re.Pattern] = {}
        self._compile_lexicon_patterns()

    def _compile_lexicon_patterns(self) -> None:
        """Pre-compile regex patterns for efficient multi-word matching."""
        for category, tokens in self.EMOTION_LEXICONS.items():
            escaped = [
                re.escape(t) for t in sorted(tokens.keys(), key=len, reverse=True)
            ]
            self._compiled_patterns[category] = re.compile(
                r"\b(" + "|".join(escaped) + r")\b",
                re.IGNORECASE,
            )

    def explain(
        self,
        text: str,
        primary_emotion: str,
        secondary_emotions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate token-level attributions for the predicted emotion.

        Args:
            text: Raw user input.
            primary_emotion: Main predicted label (e.g., "fear", "joy").
            secondary_emotions: Other candidate labels for cross-emotion analysis.
        """
        text_lower = text.lower()

        normalized_primary = self._normalize_emotion(primary_emotion)
        emotion_contexts = [normalized_primary]

        if secondary_emotions:
            for em in secondary_emotions:
                norm = self._normalize_emotion(em)
                if norm not in emotion_contexts:
                    emotion_contexts.append(norm)

        all_attributions: List[TokenAttribution] = []
        for emotion_key in emotion_contexts:
            if emotion_key not in self.EMOTION_LEXICONS:
                logger.debug("No lexicon for emotion", key=emotion_key)
                continue
            all_attributions.extend(
                self._score_text_against_lexicon(text_lower, emotion_key)
            )

        # Deduplicate + rank
        unique_attrs = self._deduplicate(all_attributions)
        ranked = sorted(unique_attrs, key=lambda x: (-abs(x.weight), x.token))

        # Apply display threshold, but always show at least MIN tokens if any matched
        visible = [
            a for a in ranked
            if abs(a.weight) >= settings.EXPLAINER_DISPLAY_THRESHOLD
        ]
        if not visible and ranked:
            visible = ranked[: max(settings.EXPLAINER_MIN_TOKENS, 3)]

        summary = self._generate_summary(visible, normalized_primary)

        logger.info(
            "Explainer complete",
            emotion=normalized_primary,
            tokens_matched=len(all_attributions),
            tokens_returned=len(visible[: settings.EXPLAINER_MAX_TOKENS]),
        )

        return {
            "tokens": [
                {
                    "token": t.token,
                    "weight": round(t.weight, 3),
                    "influence": "positive" if t.weight > 0 else "negative",
                }
                for t in visible[: settings.EXPLAINER_MAX_TOKENS]
            ],
            "summary": summary,
            "method": "lexicon_attribution",
            "confidence_indicators": [t.token for t in visible[:3]],
        }

    def _score_text_against_lexicon(
        self, text: str, emotion_key: str
    ) -> List[TokenAttribution]:
        """Match tokens against emotion lexicon and collect weighted scores."""
        lexicon = self.EMOTION_LEXICONS.get(emotion_key, {})
        pattern = self._compiled_patterns.get(emotion_key)
        if not pattern or not lexicon:
            return []

        attributions: List[TokenAttribution] = []
        for match in pattern.finditer(text):
            token = match.group(1)
            base_weight = lexicon[token.lower()]
            multiplier = 1.15 if len(token.split()) > 1 else 1.0
            attributions.append(
                TokenAttribution(
                    token=token,
                    weight=base_weight * multiplier,
                    emotion_category=emotion_key,
                )
            )
        return attributions

    @staticmethod
    def _normalize_emotion(emotion_label: str) -> str:
        """
        Convert classifier output to lexicon key.
        Now returns exact matches for all 8 emotion types.
        """
        if not emotion_label:
            return "sadness"

        lower = emotion_label.lower().strip()
        primary = lower.split("/")[0].split("-")[0].split("(")[0].strip()

        # All classifier labels map to a lexicon bucket
        mappings = {
            "sadness": "sadness",
            "depression": "sadness",
            "sad": "sadness",
            "grief": "sadness",

            "fear": "fear",
            "anxiety": "fear",       # ← Merged into fear bucket
            "anxious": "fear",
            "scared": "fear",
            "worried": "fear",

            "anger": "anger",
            "angry": "anger",
            "rage": "anger",

            "joy": "joy",
            "happy": "joy",
            "happiness": "joy",

            "love": "love",
            "loved": "love",

            "surprise": "surprise",
            "surprised": "surprise",

            "disgust": "disgust",
            "disgusted": "disgust",

            "neutral": "sadness",    # Fallback (rare)
        }
        return mappings.get(primary, "sadness")

    @staticmethod
    def _deduplicate(attrs: List[TokenAttribution]) -> List[TokenAttribution]:
        """Keep entry with highest absolute weight per unique token."""
        seen: Dict[str, TokenAttribution] = {}
        for attr in attrs:
            key = attr.token.lower()
            if key not in seen or abs(attr.weight) > abs(seen[key].weight):
                seen[key] = attr
        return list(seen.values())

    def _generate_summary(
        self, attrs: List[TokenAttribution], emotion: str
    ) -> str:
        """Create a human-readable explanation string."""
        if not attrs:
            return (
                f"No strongly influential individual tokens detected for {emotion}. "
                "The model's decision came from broader semantic patterns rather "
                "than individual keywords."
            )

        positive = [a for a in attrs if a.weight > 0][:3]
        negative = [a for a in attrs if a.weight < 0][:2]

        parts = []
        if positive:
            tokens_str = ", ".join([f'"{t.token}"' for t in positive])
            parts.append(f"Key indicators: {tokens_str}")
        if negative:
            tokens_str = ", ".join([f'"{t.token}"' for t in negative])
            parts.append(f"Diluted by: {tokens_str}")

        main_tokens = ", ".join([a.token for a in attrs[:5]])
        return (
            f"The {emotion} classification is primarily driven by: "
            f"{main_tokens}. {' '.join(parts)}"
        )


# ─────────────────────────────────────────────────────────────────────
# Singletons
# ─────────────────────────────────────────────────────────────────────
explainer = EmotionExplainer()
emotion_explainer = explainer

__all__ = [
    "EmotionExplainer",
    "TokenAttribution",
    "explainer",
    "emotion_explainer",
]