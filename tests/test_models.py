"""
Model Layer Unit Tests

Tests for EmotionClassifier, EmbeddingModel, and RAGPipeline.
Uses mocking to avoid loading actual ML models during testing.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock


# ── Emotion Classifier Tests ───────────────────────────────────────────────

class TestEmotionClassifier:

    def test_predict_raises_if_not_loaded(self):
        """predict() raises RuntimeError if model not loaded."""
        from backend.models.classifier import EmotionClassifier

        clf = EmotionClassifier()
        # Do NOT call clf.load()

        with pytest.raises(RuntimeError, match="Model not loaded"):
            clf.predict("test text")

    def test_is_loaded_false_before_load(self):
        """is_loaded returns False before load() is called."""
        from backend.models.classifier import EmotionClassifier

        clf = EmotionClassifier()
        assert clf.is_loaded is False

    def test_predict_returns_required_keys(self):
        """predict() returns emotion, confidence, all_predictions."""
        from backend.models.classifier import EmotionClassifier

        clf = EmotionClassifier()
        clf._is_loaded = True

        mock_pipeline_output = [[
            {"label": "joy", "score": 0.90},
            {"label": "neutral", "score": 0.06},
            {"label": "sadness", "score": 0.04}
        ]]

        clf.classifier = MagicMock(return_value=mock_pipeline_output)

        result = clf.predict("I feel happy!")

        assert "emotion" in result
        assert "confidence" in result
        assert "all_predictions" in result

    def test_predict_returns_highest_confidence_emotion(self):
        """predict() returns the emotion with highest score."""
        from backend.models.classifier import EmotionClassifier

        clf = EmotionClassifier()
        clf._is_loaded = True

        mock_pipeline_output = [[
            {"label": "sadness", "score": 0.75},
            {"label": "fear", "score": 0.15},
            {"label": "neutral", "score": 0.10}
        ]]

        clf.classifier = MagicMock(return_value=mock_pipeline_output)

        result = clf.predict("I feel so sad")

        assert result["emotion"] == "sadness"
        assert result["confidence"] == 0.75

    def test_predict_all_predictions_sorted_descending(self):
        """all_predictions are sorted by score descending."""
        from backend.models.classifier import EmotionClassifier

        clf = EmotionClassifier()
        clf._is_loaded = True

        mock_pipeline_output = [[
            {"label": "neutral", "score": 0.10},
            {"label": "joy", "score": 0.80},
            {"label": "fear", "score": 0.10}
        ]]

        clf.classifier = MagicMock(return_value=mock_pipeline_output)

        result = clf.predict("okay I guess")
        scores = [p["score"] for p in result["all_predictions"]]

        assert scores == sorted(scores, reverse=True)

    def test_predict_confidence_between_0_and_1(self):
        """Confidence score is always between 0.0 and 1.0."""
        from backend.models.classifier import EmotionClassifier

        clf = EmotionClassifier()
        clf._is_loaded = True

        mock_pipeline_output = [[
            {"label": "anger", "score": 0.65},
            {"label": "disgust", "score": 0.35}
        ]]

        clf.classifier = MagicMock(return_value=mock_pipeline_output)

        result = clf.predict("I am frustrated")

        assert 0.0 <= result["confidence"] <= 1.0


# ── Embedding Model Tests ──────────────────────────────────────────────────

class TestEmbeddingModel:

    def test_encode_raises_if_not_loaded(self):
        """encode() raises RuntimeError if model not loaded."""
        from backend.models.embeddings import EmbeddingModel

        em = EmbeddingModel()

        with pytest.raises(RuntimeError, match="not loaded"):
            em.encode(["test"])

    def test_is_loaded_false_before_load(self):
        """is_loaded returns False before load() is called."""
        from backend.models.embeddings import EmbeddingModel

        em = EmbeddingModel()
        assert em.is_loaded is False

    def test_encode_returns_numpy_array(self):
        """encode() returns a numpy ndarray."""
        from backend.models.embeddings import EmbeddingModel

        em = EmbeddingModel()
        em._is_loaded = True
        em.model = MagicMock()
        em.model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        result = em.encode(["test text"])

        assert isinstance(result, np.ndarray)

    def test_encode_single_returns_1d_array(self):
        """encode_single() returns a 1D array for single text."""
        from backend.models.embeddings import EmbeddingModel

        em = EmbeddingModel()
        em._is_loaded = True
        em.model = MagicMock()
        em.model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        result = em.encode_single("test text")

        assert result.ndim == 1

    def test_encode_batch_returns_correct_shape(self):
        """encode() output shape matches (n_texts, embedding_dim)."""
        from backend.models.embeddings import EmbeddingModel

        em = EmbeddingModel()
        em._is_loaded = True
        em.model = MagicMock()

        fake_embeddings = np.random.rand(3, 384)
        em.model.encode.return_value = fake_embeddings

        result = em.encode(["text1", "text2", "text3"])

        assert result.shape == (3, 384)


# ── RAG Pipeline Tests ─────────────────────────────────────────────────────

class TestRAGPipeline:

    def test_retrieve_raises_if_not_initialized(self):
        """retrieve() raises RuntimeError if pipeline not initialized."""
        from backend.models.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline()

        with pytest.raises(RuntimeError, match="not initialized"):
            pipeline.retrieve("test", "neutral")

    def test_is_initialized_false_before_init(self):
        """is_initialized returns False before initialize() is called."""
        from backend.models.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline()
        assert pipeline.is_initialized is False

    def test_retrieve_returns_list(self):
        """retrieve() returns a list."""
        from backend.models.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline()
        pipeline._is_initialized = True

        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["doc1 content", "doc2 content"]],
            "metadatas": [
                [
                    {
                        "title": "Title 1",
                        "category": "breathing",
                        "emotions": "anxiety,fear"
                    },
                    {
                        "title": "Title 2",
                        "category": "coping",
                        "emotions": "sadness"
                    }
                ]
            ],
            "distances": [[0.1, 0.2]]
        }
        pipeline.collection = mock_collection

        with patch(
            "backend.models.rag_pipeline.embedding_model"
        ) as mock_em:
            mock_em.encode_single.return_value = np.random.rand(384)
            result = pipeline.retrieve("I feel anxious", "fear")

        assert isinstance(result, list)

    def test_retrieve_filters_low_similarity(self):
        """retrieve() excludes results below similarity threshold."""
        from backend.models.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline()
        pipeline._is_initialized = True

        mock_collection = MagicMock()
        mock_collection.count.return_value = 2
        mock_collection.query.return_value = {
            "ids": [["id1"]],
            "documents": [["Low relevance document"]],
            "metadatas": [[
                {
                    "title": "Low Relevance",
                    "category": "wellness",
                    "emotions": "neutral"
                }
            ]],
            # Distance of 0.95 = similarity of 0.05 (below threshold)
            "distances": [[0.95]]
        }
        pipeline.collection = mock_collection

        with patch(
            "backend.models.rag_pipeline.embedding_model"
        ) as mock_em:
            mock_em.encode_single.return_value = np.random.rand(384)
            result = pipeline.retrieve("random text", "neutral")

        # Low similarity results filtered out
        assert result == []