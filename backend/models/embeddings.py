from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import structlog
from backend.config.settings import settings

logger = structlog.get_logger(__name__)


class EmbeddingModel:
    """
    Sentence embedding model for RAG pipeline.
    Uses sentence-transformers for semantic similarity search.
    """

    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.model = None
        self._is_loaded = False

    def load(self) -> None:
        """Load sentence transformer model."""
        try:
            logger.info("Loading embedding model", model=self.model_name)
            self.model = SentenceTransformer(self.model_name)
            self._is_loaded = True
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error("Failed to load embedding model", error=str(e))
            raise RuntimeError(f"Embedding model loading failed: {e}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode list of texts to embedding vectors.

        Args:
            texts: List of strings to embed

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if not self._is_loaded:
            raise RuntimeError("Embedding model not loaded")

        embeddings = self.model.encode(
            texts,
            batch_size=settings.BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True   # L2 normalization for cosine similarity
        )
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string."""
        return self.encode([text])[0]

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded


# Module-level singleton
embedding_model = EmbeddingModel()