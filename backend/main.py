from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import structlog

from backend.config.settings import settings
from backend.api.routes import router
from backend.models.classifier import emotion_classifier
from backend.models.embeddings import embedding_model
from backend.models.rag_pipeline import rag_pipeline
from backend.monitoring.logger import setup_logging
from backend.monitoring.metrics import start_metrics_server

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup initialization and graceful shutdown.
    Runs once when the server starts.
    """
    # ── STARTUP ────────────────────────────────────────────────────────────
    logger.info("Starting Mental Health Agentic AI Platform")

    # Initialize logging
    setup_logging()

    # Load ML models (sequential to manage memory)
    logger.info("Loading ML models...")
    emotion_classifier.load()
    embedding_model.load()
    rag_pipeline.initialize()

    # Start monitoring
    start_metrics_server()

    logger.info("All systems initialized. Server ready.")

    yield  # Application runs here

    # ── SHUTDOWN ───────────────────────────────────────────────────────────
    logger.info("Shutting down Mental Health Agentic AI Platform")


# ── FastAPI Application ────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "End-to-end Agentic AI platform for mental health intelligence. "
        "Features transformer-based NLP, RAG pipelines, multi-agent "
        "orchestration, explainability, and real-time monitoring."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# ── CORS Middleware ────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)

# ── Routes ─────────────────────────────────────────────────────────────────

app.include_router(router, prefix="")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )