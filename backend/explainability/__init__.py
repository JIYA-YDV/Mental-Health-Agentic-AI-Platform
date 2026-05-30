# backend/explainability/__init__.py
"""
Explainability Package

Token-level explanation generation for emotion classification.

Usage:
    from backend.explainability.explainer import explainer
    explanations = explainer.explain(text="I feel sad", emotion="Sadness")
"""

__all__ = [
    "explainer",
    "EmotionExplainer",
]