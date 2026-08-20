# backend/explainability/shap_explainer.py
"""
Real SHAP-based token attribution for emotion classification.

Uses shap.Explainer with the DistilRoBERTa pipeline to compute
gradient-based attributions for each token in the input text.

Slower than lexicon (~2s vs 5ms) but shows what the ACTUAL model
attended to, not what a hardcoded dictionary matched.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import structlog

from backend.config.settings import settings

logger = structlog.get_logger(__name__)


class SHAPExplainer:
    """
    Wraps SHAP's text explainer around the emotion classifier pipeline.
    Lazy-loads SHAP to avoid startup delay if never used.
    """

    def __init__(self):
        self._explainer = None
        self._available = None  # Lazy check

    @property
    def available(self) -> bool:
        """Check if SHAP is installed and pipeline is loaded."""
        if self._available is None:
            try:
                import shap  # noqa: F401
                from backend.models.classifier import emotion_classifier

                if not emotion_classifier.is_loaded:
                    self._available = False
                    logger.warning("SHAP unavailable: classifier not loaded")
                else:
                    self._available = True
                    logger.info("SHAP explainer available")
            except ImportError:
                self._available = False
                logger.warning("SHAP not installed (pip install shap)")
        return self._available

    def _get_explainer(self):
        """Lazy-initialize the SHAP text explainer."""
        if self._explainer is not None:
            return self._explainer

        import shap
        from backend.models.classifier import emotion_classifier

        pipeline = emotion_classifier.classifier
        # shap.Explainer with a transformers pipeline uses Partition explainer
        # + text masker automatically
        self._explainer = shap.Explainer(pipeline)
        logger.info("SHAP explainer initialized")
        return self._explainer

    def explain(
        self,
        text: str,
        primary_emotion: str,
        secondary_emotions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute SHAP values for the input text.

        Returns:
            Dict compatible with lexicon explainer format:
            {
                "tokens": [{"token", "weight", "influence"}, ...],
                "summary": str,
                "method": "shap_gradient",
                "confidence_indicators": [top 3 tokens]
            }
        """
        if not self.available:
            raise RuntimeError("SHAP explainer not available")

        try:
            explainer = self._get_explainer()

            # Compute SHAP values for a single input
            shap_values = explainer([text])

            # shap_values.values shape: (1, num_tokens, num_classes)
            # shap_values.data shape:   (1, num_tokens)
            # shap_values.output_names: list of emotion labels

            output_names = list(shap_values.output_names)
            emotion_lower = primary_emotion.lower()

            # Find the column index for the primary emotion
            try:
                emotion_idx = [n.lower() for n in output_names].index(emotion_lower)
            except ValueError:
                logger.warning(
                    "Primary emotion not in SHAP output_names",
                    primary=emotion_lower,
                    output_names=output_names,
                )
                # Fallback: use column 0
                emotion_idx = 0

            # Extract per-token attributions for the primary emotion
            tokens_raw = shap_values.data[0]
            values_raw = shap_values.values[0][:, emotion_idx]

            # Normalize weights to [-1, 1] range for consistency with lexicon
            max_abs = max(abs(v) for v in values_raw) if len(values_raw) > 0 else 1.0
            if max_abs == 0:
                max_abs = 1.0

            # Build token attributions, filtering out whitespace/punctuation
            token_attrs = []
            for tok, val in zip(tokens_raw, values_raw):
                tok_clean = str(tok).strip()
                if not tok_clean or len(tok_clean) < 2:
                    continue
                # Skip common subword prefixes
                if tok_clean.startswith("##"):
                    tok_clean = tok_clean[2:]
                # Skip stop-word-like tokens
                if tok_clean.lower() in {"the", "a", "an", "is", "are", "was", "were",
                                         "to", "of", "in", "on", "at", "for", "and",
                                         "or", "but", "i", "my", "me", "you", "your"}:
                    continue

                normalized_weight = float(val) / max_abs
                token_attrs.append({
                    "token": tok_clean,
                    "weight": round(normalized_weight, 3),
                    "raw_value": float(val),
                })

            # Sort by absolute weight (most influential first)
            token_attrs.sort(key=lambda x: abs(x["weight"]), reverse=True)

            # Apply display threshold
            threshold = settings.EXPLAINER_DISPLAY_THRESHOLD
            visible = [t for t in token_attrs if abs(t["weight"]) >= threshold]

            # Safety net: show at least MIN tokens if any exist
            if not visible and token_attrs:
                visible = token_attrs[: settings.EXPLAINER_MIN_TOKENS]

            # Cap at MAX
            visible = visible[: settings.EXPLAINER_MAX_TOKENS]

            # Build final format
            tokens_out = [
                {
                    "token": t["token"],
                    "weight": t["weight"],
                    "influence": "positive" if t["weight"] > 0 else "negative",
                }
                for t in visible
            ]

            # Generate summary
            if tokens_out:
                top_positive = [t["token"] for t in tokens_out if t["weight"] > 0][:3]
                top_negative = [t["token"] for t in tokens_out if t["weight"] < 0][:2]

                parts = []
                if top_positive:
                    parts.append(
                        f"Model attention concentrated on: "
                        f"{', '.join(top_positive)}"
                    )
                if top_negative:
                    parts.append(f"Diminished by: {', '.join(top_negative)}")

                summary = (
                    f"SHAP attribution for {emotion_lower}: "
                    f"{' | '.join(parts)}. "
                )
            else:
                summary = (
                    f"SHAP found no strongly-attributed tokens for {emotion_lower}. "
                    "Model relied on distributed patterns rather than individual words."
                )

            logger.info(
                "SHAP explainer complete",
                emotion=emotion_lower,
                tokens_returned=len(tokens_out),
                max_weight=max_abs,
            )

            return {
                "tokens": tokens_out,
                "summary": summary,
                "method": "shap_gradient",
                "confidence_indicators": [t["token"] for t in tokens_out[:3]],
            }

        except Exception as e:
            logger.error("SHAP explanation failed", error=str(e), exc_info=True)
            raise


# Singleton
shap_explainer = SHAPExplainer()