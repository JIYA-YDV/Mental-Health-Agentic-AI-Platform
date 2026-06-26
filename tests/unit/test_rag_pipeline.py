# tests/unit/test_rag_pipeline.py
"""Test RAG pipeline emotion normalization and fallback logic."""
from unittest.mock import MagicMock, patch

import pytest


pytestmark = pytest.mark.unit


class TestEmotionNormalization:
    """RAG pipeline should clean classifier suffixes before querying."""

    def test_normalize_strips_suffix(self):
        from backend.models.rag_pipeline import RAGPipeline
        assert RAGPipeline._normalize_emotion("Sadness / Depression") == "sadness"

    def test_normalize_lowercases(self):
        from backend.models.rag_pipeline import RAGPipeline
        assert RAGPipeline._normalize_emotion("ANGER") == "anger"

    def test_normalize_handles_empty(self):
        from backend.models.rag_pipeline import RAGPipeline
        assert RAGPipeline._normalize_emotion("") == ""


class TestRAGSingleton:

    def test_singleton_not_a_tuple(self):
        """Regression test for the trailing-comma bug."""
        from backend.models.rag_pipeline import rag_pipeline
        assert not isinstance(rag_pipeline, tuple)

    def test_singleton_has_retrieve_method(self):
        from backend.models.rag_pipeline import rag_pipeline
        assert hasattr(rag_pipeline, "retrieve")
        assert callable(rag_pipeline.retrieve)


class TestKnowledgeBaseContent:
    """Ensure the financial/career docs we added are present."""

    def test_financial_doc_present(self):
        from backend.models.rag_pipeline import WELLNESS_KNOWLEDGE_BASE
        ids = {doc["id"] for doc in WELLNESS_KNOWLEDGE_BASE}
        assert "financial_001" in ids

    def test_mixed_signals_doc_present(self):
        from backend.models.rag_pipeline import WELLNESS_KNOWLEDGE_BASE
        ids = {doc["id"] for doc in WELLNESS_KNOWLEDGE_BASE}
        assert "mixed_signals_001" in ids

    def test_every_doc_has_required_fields(self):
        from backend.models.rag_pipeline import WELLNESS_KNOWLEDGE_BASE
        required = {"id", "title", "content", "category", "emotions"}
        for doc in WELLNESS_KNOWLEDGE_BASE:
            assert required.issubset(doc.keys()), \
                f"Doc {doc.get('id')} missing fields"

    def test_all_ids_are_unique(self):
        from backend.models.rag_pipeline import WELLNESS_KNOWLEDGE_BASE
        ids = [doc["id"] for doc in WELLNESS_KNOWLEDGE_BASE]
        assert len(ids) == len(set(ids)), "Duplicate doc IDs found"


class TestRetrievalLogic:
    """Test retrieve() with a mocked ChromaDB collection."""

    def _make_pipeline_with_mock_collection(self, mock_results):
        from backend.models.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline()
        pipeline._is_initialized = True
        pipeline.collection = MagicMock()
        pipeline.collection.count.return_value = 10
        pipeline.collection.query.return_value = mock_results
        return pipeline

    def test_fallback_returns_top_match_below_threshold(self):
        """If top match is just below threshold, return it via fallback."""
        import numpy as np
        from backend.config.settings import settings

        # Score below threshold but within fallback delta
        floor = settings.SIMILARITY_THRESHOLD - 0.05
        mock_distance = 1.0 - floor

        with patch(
            "backend.models.embeddings.embedding_model"
        ) as mock_emb:
            # Return numpy array so .tolist() works (matches real API)
            mock_emb.encode_single.return_value = np.array([0.1] * 384)

            pipeline = self._make_pipeline_with_mock_collection({
                "ids": [["doc1"]],
                "documents": [["Some wellness content"]],
                "metadatas": [[{
                    "title": "Fallback Match",
                    "category": "coping",
                    "emotions": "sadness",
                }]],
                "distances": [[mock_distance]],
            })

            results = pipeline.retrieve("test query", "Sadness / Depression")

        assert len(results) == 1, "Fallback should return top-1 match"
        assert results[0]["title"] == "Fallback Match"
        assert results[0]["category"] == "coping"

    def test_no_results_when_top_match_far_below_threshold(self):
        """If even top match is too far below threshold, return empty."""
        import numpy as np
        from backend.config.settings import settings

        # Way below the fallback floor
        very_low_score = max(
            0.01,
            settings.SIMILARITY_THRESHOLD - settings.FALLBACK_THRESHOLD_DELTA - 0.2,
        )
        mock_distance = 1.0 - very_low_score

        with patch(
            "backend.models.embeddings.embedding_model"
        ) as mock_emb:
            mock_emb.encode_single.return_value = np.array([0.1] * 384)

            pipeline = self._make_pipeline_with_mock_collection({
                "ids": [["doc1"]],
                "documents": [["Irrelevant content"]],
                "metadatas": [[{
                    "title": "Weak Match",
                    "category": "general",
                    "emotions": "neutral",
                }]],
                "distances": [[mock_distance]],
            })

            results = pipeline.retrieve("query", "sadness")

        assert results == [], "Should return empty when even fallback fails"