# evaluation/label_mapper.py
"""
Bidirectional label mapping between dataset labels and model labels.

The j-hartmann/emotion-english-distilroberta-base model predicts 7 emotions:
    anger, disgust, fear, joy, neutral, sadness, surprise

The dair-ai/emotion dataset uses 6 emotions:
    sadness, joy, love, anger, fear, surprise

Mapping strategy:
  - 'love' (dataset) → 'joy' (model)   [no direct match; positive affect]
  - 'disgust', 'neutral' (model)       [not in dataset; excluded from eval]
"""
from typing import Dict, List, Optional


# Dataset label index → label name (from dair-ai/emotion)
DATASET_LABELS: Dict[int, str] = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

# Map dataset labels → canonical model labels
DATASET_TO_MODEL: Dict[str, str] = {
    "sadness":  "sadness",
    "joy":      "joy",
    "love":     "joy",       # collapse 'love' into 'joy' for evaluation
    "anger":    "anger",
    "fear":     "fear",
    "surprise": "surprise",
}

# Reverse map (for confusion matrix display)
MODEL_LABELS_USED: List[str] = sorted({v for v in DATASET_TO_MODEL.values()})
# → ['anger', 'fear', 'joy', 'sadness', 'surprise']

# Model emits these but they're not in the eval dataset
EXCLUDED_MODEL_LABELS = {"disgust", "neutral"}


def dataset_label_to_model_label(dataset_label_id: int) -> str:
    """Convert dataset integer label → canonical model emotion string."""
    if dataset_label_id not in DATASET_LABELS:
        raise ValueError(f"Unknown dataset label ID: {dataset_label_id}")
    raw = DATASET_LABELS[dataset_label_id]
    return DATASET_TO_MODEL[raw]


def normalize_model_prediction(prediction: str) -> Optional[str]:
    """
    Normalize a model prediction string for comparison.

    Returns None if prediction is one of the excluded labels
    (disgust, neutral) — caller should drop these samples or
    count them as misclassifications depending on policy.
    """
    pred_lower = prediction.lower().strip()
    if pred_lower in EXCLUDED_MODEL_LABELS:
        return None
    return pred_lower


def get_eval_label_set() -> List[str]:
    """Return the canonical list of labels used in evaluation (sorted)."""
    return MODEL_LABELS_USED.copy()