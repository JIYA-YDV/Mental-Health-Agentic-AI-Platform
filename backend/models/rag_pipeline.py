# backend/models/rag_pipeline.py
import os
import chromadb
from typing import List, Dict, Any
import structlog
from backend.config.settings import settings

logger = structlog.get_logger(__name__)


WELLNESS_KNOWLEDGE_BASE = [
    {
        "id": "breathing_001",
        "title": "4-7-8 Breathing Technique",
        "content": (
            "The 4-7-8 breathing technique helps reduce anxiety and stress. "
            "Inhale for 4 seconds, hold for 7 seconds, exhale for 8 seconds. "
            "Repeat 4 times. This activates the parasympathetic nervous system."
        ),
        "category": "breathing",
        "emotions": ["anxiety", "fear", "anger", "stress"]
    },
    {
        "id": "mindfulness_001",
        "title": "5-4-3-2-1 Grounding Exercise",
        "content": (
            "When feeling overwhelmed, use the 5-4-3-2-1 technique: "
            "Name 5 things you can see, 4 you can touch, 3 you can hear, "
            "2 you can smell, 1 you can taste. "
            "This brings you to the present moment."
        ),
        "category": "mindfulness",
        "emotions": ["anxiety", "fear", "sadness", "dissociation"]
    },
    {
        "id": "coping_001",
        "title": "Cognitive Reframing for Negative Thoughts",
        "content": (
            "When negative thoughts arise, challenge them: "
            "Is this thought based on facts? What would I tell a friend? "
            "What is the most realistic outcome? "
            "Write the thought, challenge it, create a balanced response."
        ),
        "category": "coping",
        "emotions": ["sadness", "anger", "disgust", "fear"]
    },
    {
        "id": "joy_001",
        "title": "Gratitude Practice for Emotional Wellbeing",
        "content": (
            "Daily gratitude journaling strengthens positive emotions. "
            "Write 3 specific things you are grateful for each morning. "
            "Research shows this reduces depression and increases life satisfaction."
        ),
        "category": "mindfulness",
        "emotions": ["joy", "neutral", "sadness"]
    },
    {
        "id": "crisis_001",
        "title": "Immediate Crisis Support Resources",
        "content": (
            "If you are in crisis, please reach out immediately. "
            "National Suicide Prevention Lifeline: 988 or 1-800-273-8255. "
            "Crisis Text Line: Text HOME to 741741."
        ),
        "category": "crisis",
        "emotions": ["crisis", "sadness", "fear", "hopeless"]
    },
    {
        "id": "anger_001",
        "title": "Anger Management: STOP Technique",
        "content": (
            "When anger arises, use STOP: Stop what you are doing, "
            "Take a breath, Observe your feelings without judgment, "
            "Proceed mindfully. Physical activity and counting to 10 "
            "help regulate intense anger responses."
        ),
        "category": "coping",
        "emotions": ["anger", "disgust"]
    },
    {
        "id": "sleep_001",
        "title": "Sleep Hygiene for Mental Health",
        "content": (
            "Quality sleep is foundational to mental health. "
            "Keep consistent sleep and wake times, avoid screens 1 hour before bed, "
            "keep your room cool and dark, avoid caffeine after 2pm."
        ),
        "category": "wellness",
        "emotions": ["neutral", "sadness", "anxiety"]
    },
    {
        "id": "social_001",
        "title": "Building Social Connection to Combat Loneliness",
        "content": (
            "Social isolation worsens mental health outcomes. "
            "Start small: send one text to a friend, join an online community, "
            "volunteer locally. Even brief positive interactions improve mood."
        ),
        "category": "wellness",
        "emotions": ["sadness", "neutral", "fear"]
    }
]


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.
    Compatible with chromadb==0.4.24 using PersistentClient.
    """

    def __init__(self):
        self.client = None
        self.collection = None
        self._is_initialized = False

    def initialize(self) -> None:
        """Initialize ChromaDB with PersistentClient."""
        try:
            logger.info("Initializing RAG pipeline")

            # Create persist directory if it does not exist
            os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)

            # PersistentClient is the correct API for chromadb 0.4.x
            self.client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR
            )

            self.collection = self.client.get_or_create_collection(
                name="wellness_knowledge",
                metadata={"hnsw:space": "cosine"}
            )

            if self.collection.count() == 0:
                self._populate_knowledge_base()

            self._is_initialized = True
            logger.info(
                "RAG pipeline initialized",
                document_count=self.collection.count()
            )

        except Exception as e:
            logger.error("RAG pipeline init failed", error=str(e))
            raise RuntimeError(f"RAG initialization failed: {e}")

    def _populate_knowledge_base(self) -> None:
        """Embed and store wellness documents."""
        # Import here to avoid circular imports at module level
        from backend.models.embeddings import embedding_model

        logger.info("Populating knowledge base")

        texts     = [doc["content"] for doc in WELLNESS_KNOWLEDGE_BASE]
        ids       = [doc["id"]      for doc in WELLNESS_KNOWLEDGE_BASE]
        metadatas = [
            {
                "title":    doc["title"],
                "category": doc["category"],
                "emotions": ",".join(doc["emotions"])
            }
            for doc in WELLNESS_KNOWLEDGE_BASE
        ]

        embeddings = embedding_model.encode(texts).tolist()

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        logger.info(f"Stored {len(texts)} documents in knowledge base")

    def retrieve(
        self,
        query_text: str,
        emotion: str,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant wellness recommendations."""
        from backend.models.embeddings import embedding_model

        if not self._is_initialized:
            raise RuntimeError("RAG pipeline not initialized")

        top_k = top_k or settings.TOP_K_RETRIEVAL
        enhanced_query = f"{emotion} {query_text}"
        query_embedding = embedding_model.encode_single(enhanced_query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        recommendations = []
        for i in range(len(results["ids"][0])):
            distance   = results["distances"][0][i]
            similarity = 1.0 - distance

            if similarity >= settings.SIMILARITY_THRESHOLD:
                metadata = results["metadatas"][0][i]
                recommendations.append({
                    "title":           metadata["title"],
                    "content":         results["documents"][0][i],
                    "relevance_score": round(similarity, 4),
                    "category":        metadata["category"],
                    "source":          "Mental Health Knowledge Base"
                })

        return recommendations

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized


rag_pipeline = RAGPipeline()