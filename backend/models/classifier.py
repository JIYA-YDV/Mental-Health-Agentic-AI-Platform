import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any
import structlog
from backend.config.settings import settings

logger = structlog.get_logger(__name__)


class EmotionClassifier:
    """
    Transformer-based emotion classification using DistilRoBERTa.
    Handles model loading, inference, and result formatting.

    Emotions detected: anger, disgust, fear, joy, neutral, sadness, surprise
    """

    def __init__(self):
        self.model_name = settings.EMOTION_MODEL
        self.classifier = None
        self.tokenizer = None
        self._is_loaded = False

    def load(self) -> None:
        """
        Load model and tokenizer from HuggingFace.
        Called once at application startup.
        """
        try:
            logger.info("Loading emotion classifier", model=self.model_name)

            # Determine device
            device = 0 if torch.cuda.is_available() else -1
            device_name = "CUDA" if device == 0 else "CPU"
            logger.info(f"Using device: {device_name}")

            # Load HuggingFace pipeline
            self.classifier = pipeline(
                task="text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                top_k=None,           # Return all emotion scores
                device=device,
                truncation=True,
                max_length=settings.MAX_SEQUENCE_LENGTH
            )

            self._is_loaded = True
            logger.info("Emotion classifier loaded successfully")

        except Exception as e:
            logger.error("Failed to load emotion classifier", error=str(e))
            raise RuntimeError(f"Model loading failed: {e}")

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Run inference on input text.

        Args:
            text: Cleaned user input string

        Returns:
            Dict with primary emotion, confidence, and all scores
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        try:
            # Run inference
            raw_results = self.classifier(text)

            # raw_results is List[List[Dict]] when top_k=None
            predictions = raw_results[0]

            # Sort by score descending
            predictions_sorted = sorted(
                predictions,
                key=lambda x: x["score"],
                reverse=True
            )

            # Extract primary prediction
            primary = predictions_sorted[0]

            return {
                "emotion": primary["label"],
                "confidence": primary["score"],
                "all_predictions": [
                    {"label": p["label"], "score": p["score"]}
                    for p in predictions_sorted
                ]
            }

        except Exception as e:
            logger.error("Inference failed", error=str(e), text_length=len(text))
            raise RuntimeError(f"Classification inference failed: {e}")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded


# Module-level singleton
emotion_classifier = EmotionClassifier()