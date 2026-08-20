# backend/explainability/hybrid_explainer.py
"""
Hybrid explainer: routes to SHAP or lexicon based on request preference.

Method precedence:
1. If explicitly requested method is available → use it
2. If SHAP requested but unavailable → fallback to lexicon (log warning)
3. Default (no method specified) → lexicon (fast)
"""

from typing import Any, Dict, List, Optional
import structlog

from backend.explainability.explainer import explainer as lexicon_explainer
from backend.explainability.shap_explainer import shap_explainer

logger = structlog.get_logger(__name__)


class HybridExplainer:
    """
    Facade that routes explanation requests to the appropriate backend.
    """

    def explain(
        self,
        text: str,
        primary_emotion: str,
        secondary_emotions: Optional[List[str]] = None,
        method: str = "lexicon",  # "lexicon" or "shap"
    ) -> Dict[str, Any]:
        """
        Route the explanation request based on method preference.

        Args:
            text: Raw user input
            primary_emotion: Detected emotion
            secondary_emotions: Other candidates (used by lexicon)
            method: "lexicon" (fast) or "shap" (slow but authentic)
        """
        method = (method or "lexicon").lower()

        if method == "shap":
            if shap_explainer.available:
                try:
                    result = shap_explainer.explain(
                        text=text,
                        primary_emotion=primary_emotion,
                        secondary_emotions=secondary_emotions,
                    )
                    return result
                except Exception as e:
                    logger.warning(
                        "SHAP failed, falling back to lexicon",
                        error=str(e),
                    )
                    # Fall through to lexicon
            else:
                logger.info("SHAP requested but unavailable, using lexicon")

        # Default / fallback: lexicon
        result = lexicon_explainer.explain(
            text=text,
            primary_emotion=primary_emotion,
            secondary_emotions=secondary_emotions,
        )
        # Ensure method field is present
        result["method"] = result.get("method", "lexicon_attribution")
        return result


# Singleton
hybrid_explainer = HybridExplainer()