"""
Explainability Package

Token-level explanation generation for emotion classification.
Provides human-readable attribution scores showing which
words influenced the model's emotion prediction.

Components:
- explainer.py : EmotionExplainer with lexicon-based attribution

Output per token:
    token               : original word from input
    importance          : float 0.0 - 1.0
    sentiment_direction : positive | neutral | negative

Usage:
    from backend.explainability.explainer import explainer

    explanations = explainer.explain(
        text="I feel really sad and lonely",
        emotion="Sadness / Depression"
    )

    for token in explanations:
        print(token["token"], token["importance"])

Note:
    Explanations are only generated when
    include_explanations=True is set in the API request.
    This avoids unnecessary compute on every request.
"""

from backend.explainability.explainer import explainer, EmotionExplainer

__all__ = [
    "explainer",
    "EmotionExplainer",
]