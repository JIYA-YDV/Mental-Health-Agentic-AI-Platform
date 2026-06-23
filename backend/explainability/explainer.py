# backend/explainability/explainer.py
"""
Lightweight token-level emotion attribution module.

Public exports:
    - explainer          : singleton instance (primary import)
    - emotion_explainer  : backward-compatible alias
    - EmotionExplainer   : class for testing / custom instantiation
    - TokenAttribution   : dataclass for individual token scores

Used by:
    - backend/api/routes.py  → `from backend.explainability.explainer import explainer`
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
    Lightweight token-level emotion attribution system.

    Uses curated lexicon scoring as a fast alternative to compute-heavy
    SHAP values, suitable for local LLM deployment on consumer hardware.

    Handles sub-clinical signals (circumstantial stressors) alongside
    traditional clinical symptom vocabulary.

    Thresholds (display threshold, min/max tokens) are sourced from
    backend.config.settings so they can be tuned via .env without
    code changes.
    """

    # ──────────────────────────────────────────────────────────────
    # COMBINED EMOTION LEXICON MAPPINGS
    # Weights range: 0.1 (weak signal) → 1.0 (strong diagnostic indicator)
    # ──────────────────────────────────────────────────────────────

    EMOTION_LEXICONS: Dict[str, Dict[str, float]] = {
        "sadness": {
            # Clinical symptoms
            "sad": 1.0,
            "depressed": 1.0,
            "depression": 1.0,
            "lonely": 0.9,
            "loneliness": 0.9,
            "crying": 0.95,
            "tears": 0.85,
            "hopeless": 1.0,
            "worthless": 1.0,
            "empty": 0.8,
            "numb": 0.75,
            "down": 0.6,

            # Physical manifestations & sensations
            "exhausted": 0.8,
            "tired": 0.7,
            "fatigue": 0.7,
            "drained": 0.65,
            "spin": 0.75,
            "spinning": 0.75,
            "head": 0.5,
            "dizzy": 0.6,
            "heavy": 0.4,

            # Circumstantial / Financial stressors
            "earning": 0.80,
            "earnings": 0.80,
            "money": 0.70,
            "financial": 0.70,
            "unemployed": 0.90,
            "job": 0.55,
            "career": 0.60,
            "work": 0.40,
            "salary": 0.65,
            "income": 0.65,
            "debt": 0.75,
            "broke": 0.85,
            "poor": 0.50,
            "afford": 0.60,

            # Existential / Stuck states
            "stuck": 0.75,
            "trapped": 0.85,
            "behind": 0.55,
            "failing": 0.80,
            "failure": 0.90,
            "cannot": 0.40,
            "never": 0.35,
            "enough": 0.45,

            # Mixed-signal dissonance indicators
            "good": -0.10,
            "fine": -0.15,
            "okay": -0.10,
            "but": 0.30,
            "however": 0.25,
        },

        "anxiety": {
            "anxious": 1.0,
            "anxiety": 1.0,
            "panic": 1.0,
            "worried": 0.85,
            "worrying": 0.85,
            "nervous": 0.75,
            "overwhelm": 0.9,
            "overwhelmed": 0.9,

            # Physical anxiety markers
            "heart": 0.5,
            "pounding": 0.75,
            "racing": 0.75,
            "breath": 0.6,
            "chest": 0.55,
            "shaking": 0.7,
            "trembling": 0.7,

            # Future uncertainty
            "future": 0.45,
            "uncertain": 0.7,
            "uncertainty": 0.7,
            "what if": 0.8,
            "deadline": 0.65,
            "pressure": 0.75,
            "stress": 0.6,
            "stressful": 0.7,

            # Financial / career anxiety overlap
            "earning": 0.65,
            "lose job": 0.85,
            "recession": 0.7,
        },

        "fear": {
            "afraid": 1.0,
            "scared": 1.0,
            "terrified": 1.0,
            "phobia": 0.9,
            "dread": 0.9,
            "horror": 0.8,

            # Safety threats
            "dangerous": 0.85,
            "unsafe": 0.8,
            "threat": 0.75,
            "attack": 0.7,
            "hurt": 0.6,

            # Loss-related fear
            "abandonment": 0.9,
            "alone": 0.6,
            "die": 0.8,
            "death": 0.75,
            "end": 0.5,
        },

        "anger": {
            "angry": 1.0,
            "anger": 1.0,
            "rage": 0.95,
            "furious": 0.95,
            "irritated": 0.75,
            "frustrated": 0.8,
            "mad": 0.7,
            "annoyed": 0.65,
            "upset": 0.5,

            # Physical anger cues
            "hate": 0.85,
            "kill": 0.9,
            "damn": 0.4,
            "scream": 0.7,
            "yelling": 0.7,
        },
    }

    def __init__(self):
        self._compiled_patterns: Dict[str, re.Pattern] = {}
        self._compile_lexicon_patterns()

    def _compile_lexicon_patterns(self) -> None:
        """Pre-compile regex patterns for efficient multi-word matching."""
        for category, tokens in self.EMOTION_LEXICONS.items():
            # Escape regex special chars; sort by length (longest first)
            # so multi-word phrases match before their sub-tokens.
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
        Generate token-level attributions for predicted emotions.

        Args:
            text: Raw input text from user.
            primary_emotion: Main predicted label (e.g., "Sadness / Depression").
            secondary_emotions: Other high-probability candidate labels.

        Returns:
            Dictionary with 'tokens', 'summary', 'method', and
            'confidence_indicators'.
        """
        text_lower = text.lower()

        # Normalize emotion labels (same logic as RAG pipeline)
        normalized_primary = self._normalize_emotion(primary_emotion)
        emotion_contexts = [normalized_primary]

        if secondary_emotions:
            for em in secondary_emotions:
                norm = self._normalize_emotion(em)
                if norm not in emotion_contexts:
                    emotion_contexts.append(norm)

        # Score each relevant emotion bucket
        all_attributions: List[TokenAttribution] = []
        for emotion_key in emotion_contexts:
            if emotion_key not in self.EMOTION_LEXICONS:
                logger.debug("No lexicon for emotion key", key=emotion_key)
                continue
            all_attributions.extend(
                self._score_text_against_lexicon(text_lower, emotion_key)
            )

        # Deduplicate (keep highest absolute weight per token)
        unique_attrs = self._deduplicate(all_attributions)

        # Sort by absolute influence (desc), then alphabetically
        ranked = sorted(unique_attrs, key=lambda x: (-abs(x.weight), x.token))

        # Filter by display threshold OR top-N fallback
        visible = [
            a for a in ranked
            if abs(a.weight) >= settings.EXPLAINER_DISPLAY_THRESHOLD
        ]

        # Safety net: surface at least MIN tokens if anything matched
        if not visible and ranked:
            logger.debug(
                "Explanation relaxed below display threshold",
                threshold=settings.EXPLAINER_DISPLAY_THRESHOLD,
                top_score=abs(ranked[0].weight),
            )
            visible = ranked[: settings.EXPLAINER_MIN_TOKENS]

        summary = self._generate_summary(visible, normalized_primary)

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
        """Match tokens against specific emotion lexicon and collect scores."""
        lexicon = self.EMOTION_LEXICONS.get(emotion_key, {})
        pattern = self._compiled_patterns.get(emotion_key)
        if not pattern or not lexicon:
            return []

        attributions: List[TokenAttribution] = []
        for match in pattern.finditer(text):
            token = match.group(1)
            base_weight = lexicon[token.lower()]
            # Boost multi-word phrases — they signal stronger intent
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
        Convert classifier output to canonical lexicon key.

        Examples:
            "Sadness / Depression" → "sadness"
            "Fear / Anxiety"       → "fear"
        """
        if not emotion_label:
            return "neutral"

        lower = emotion_label.lower().strip()
        primary = lower.split("/")[0].split("-")[0].split("(")[0].strip()

        mappings = {
            "sadness": "sadness",
            "depression": "sadness",
            "sad": "sadness",
            "anxiety": "anxiety",
            "anxious": "anxiety",
            "fear": "fear",
            "scared": "fear",
            "anger": "anger",
            "angry": "anger",
            "joy": "joy",
            "happy": "joy",
            "surprise": "surprise",
            "disgust": "disgust",
            "neutral": "neutral",
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
            return f"No strongly influential individual tokens detected for {emotion}."

        positive = [a for a in attrs if a.weight > 0][:3]
        negative = [a for a in attrs if a.weight < 0][:2]

        parts = []
        if positive:
            tokens_str = ", ".join([f'"{t.token}"' for t in positive])
            parts.append(f"Key stressors: {tokens_str}")
        if negative:
            tokens_str = ", ".join([f'"{t.token}"' for t in negative])
            parts.append(f"Diluted by: {tokens_str}")

        main_tokens = ", ".join([a.token for a in attrs[:5]])
        return (
            f"The {emotion} classification is primarily driven by mentions of "
            f"{main_tokens}. {' '.join(parts)}"
        )


# ──────────────────────────────────────────────────────────────────────
# Singleton instances for app-wide use
# ──────────────────────────────────────────────────────────────────────
# Primary export — matches existing import in backend/api/routes.py
explainer = EmotionExplainer()

# Backward-compatible alias
emotion_explainer = explainer

__all__ = [
    "EmotionExplainer",
    "TokenAttribution",
    "explainer",
    "emotion_explainer",
]