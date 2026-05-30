import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any
import structlog
from backend.models.embeddings import embedding_model
from backend.config.settings import settings

logger = structlog.get_logger(__name__)


# Curated mental health knowledge base
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
            "2 you can smell, 1 you can taste. This brings you to the present moment."
        ),
        "category": "mindfulness",
        "emotions": ["anxiety", "fear", "sadness", "dissociation"]
    },
    {
        "id": "coping_001",
        "title": "Cognitive Reframing for Negative Thoughts",
        "content": (
            "When negative thoughts arise, challenge them with questions: "
            "Is this thought based on facts? What would I tell a friend? "
            "What is the most realistic outcome? "
            "Write down the thought, challenge it, and create a balanced response."
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
            "Focus on why each matters to you personally. "
            "Research shows this reduces depression and increases life satisfaction."
        ),
        "category": "mindfulness",
        "emotions": ["joy", "neutral", "sadness"]
    },
    {
        "id": "crisis_001",
        "title": "Immediate Crisis Support Resources",
        "content": (
            "If you are in crisis, please reach out immediately: "
            "National Suicide Prevention Lifeline: 988 or 1-800-273-8255. "
            "Crisis Text Line: Text HOME to 741741. "
            "International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/"
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
            "Proceed mindfully. Physical activity, cold water on the face, "
            "and counting to 10 help regulate intense anger responses."
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
            "keep your room cool and dark, avoid caffeine after 2pm, "
            "and practice a wind-down routine like reading or gentle stretching."
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
            "volunteer locally, or attend a community event. "
            "Even brief positive interactions significantly improve mood and wellbeing."
        ),
        "category": "wellness",
        "emotions": ["sadness", "neutral", "fear"]
    }
]


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for wellness recommendations.
    Uses ChromaDB as vector store with sentence-transformer embeddings.
    """

    def __init__(self):
        self.client = None
        self.collection = None
        self._is_initialized = False

    def initialize(self) -> None:
        """
        Set up ChromaDB and populate knowledge base.
        Called once at application startup.
        """
        try:
            logger.info("Initializing RAG pipeline")

            # Initialize ChromaDB client
            self.client = chromadb.Client(
                ChromaSettings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=settings.CHROMA_PERSIST_DIR,
                    anonymized_telemetry=False
                )
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="wellness_knowledge",
                metadata={"hnsw:space": "cosine"}
            )

            # Populate if empty
            if self.collection.count() == 0:
                self._populate_knowledge_base()

            self._is_initialized = True
            logger.info(
                "RAG pipeline initialized",
                document_count=self.collection.count()
            )

        except Exception as e:
            logger.error("RAG pipeline initialization failed", error=str(e))
            raise RuntimeError(f"RAG initialization failed: {e}")

    def _populate_knowledge_base(self) -> None:
        """Embed and store all knowledge base documents."""
        logger.info("Populating knowledge base with wellness documents")

        texts = [doc["content"] for doc in WELLNESS_KNOWLEDGE_BASE]
        ids = [doc["id"] for doc in WELLNESS_KNOWLEDGE_BASE]
        metadatas = [
            {
                "title": doc["title"],
                "category": doc["category"],
                "emotions": ",".join(doc["emotions"])
            }
            for doc in WELLNESS_KNOWLEDGE_BASE
        ]

        # Generate embeddings
        embeddings = embedding_model.encode(texts).tolist()

        # Store in ChromaDB
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
        """
        Retrieve relevant wellness recommendations.

        Args:
            query_text: Original user input
            emotion: Detected emotion to enhance retrieval
            top_k: Number of results to return

        Returns:
            List of recommendation dicts with content and metadata
        """
        if not self._is_initialized:
            raise RuntimeError("RAG pipeline not initialized")

        top_k = top_k or settings.TOP_K_RETRIEVAL

        # Enhance query with emotion context
        enhanced_query = f"{emotion} {query_text}"

        # Generate query embedding
        query_embedding = embedding_model.encode_single(enhanced_query).tolist()

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        recommendations = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            # Convert cosine distance to similarity score
            similarity = 1.0 - distance

            if similarity >= settings.SIMILARITY_THRESHOLD:
                metadata = results["metadatas"][0][i]
                recommendations.append({
                    "title": metadata["title"],
                    "content": results["documents"][0][i],
                    "relevance_score": round(similarity, 4),
                    "category": metadata["category"],
                    "source": "Mental Health Knowledge Base"
                })

        return recommendations

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized


# Module-level singleton
rag_pipeline = RAGPipeline()