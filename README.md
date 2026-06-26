# 🧠 Mental Health Agentic AI Platform

[![CI](https://github.com/JIYA-YDV/Mental-Health-Agentic-AI-Platform/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/JIYA-YDV/Mental-Health-Agentic-AI-Platform/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/JIYA-YDV/Mental-Health-Agentic-AI-Platform/branch/main/graph/badge.svg)](https://codecov.io/gh/JIYA-YDV/Mental-Health-Agentic-AI-Platform)
[![Tests](https://img.shields.io/badge/tests-59%20passing-brightgreen)](https://github.com/JIYA-YDV/Mental-Health-Agentic-AI-Platform/actions)
[![Coverage](https://img.shields.io/badge/coverage-74.4%25-green)](https://github.com/JIYA-YDV/Mental-Health-Agentic-AI-Platform)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.1-FF4B4B)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.1-EE4C2C)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Production-grade multi-agent AI platform for mental health intelligence.** Built with Transformer-based NLP, retrieval-augmented generation, async multi-agent orchestration, explainability, structured monitoring, and full CI/CD.

> ⚠️ **DISCLAIMER**: AI research tool — NOT a substitute for professional mental health care. If you are in crisis, call **988** (US) or your local emergency number.

---

## ✨ Highlights

- 🎯 **59 automated tests** with 74.4% code coverage (sub-8s feedback loop)
- 🤖 **4-agent async pipeline** — Classification → (Crisis + RAG in parallel) → Aggregation
- 🔍 **DistilRoBERTa emotion classifier** detecting 7 emotions with calibrated confidence
- 📚 **Semantic RAG** over curated wellness KB with similarity-threshold fallback
- 🚨 **Crisis detection** layered with confidence + keyword triggers
- 🔮 **Lexicon-based token attribution** for explainable predictions
- 📊 **Prometheus metrics** + **structlog** structured JSON logging
- 🐳 **Docker-ready** with multi-service `docker-compose`
- ✅ **GitHub Actions CI** running tests + lint on every push

---

## 🏗️ Architecture
┌──────────────────────────────────────────────────────────────────┐
│ Streamlit Frontend (Port 8501) │
└─────────────────────────────┬────────────────────────────────────┘
│ HTTPS / JSON
▼
┌──────────────────────────────────────────────────────────────────┐
│ FastAPI Backend (Port 8000) │
│ /health /classify /docs (Swagger UI) │
└─────────────────────────────┬────────────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────────────────┐
│ Agent Orchestrator (asyncio.gather) │
│ │
│ 1. ClassificationAgent (blocking — feeds downstream) │
│ │ │
│ ▼ │
│ ┌─────────────────────┐ ┌─────────────────────┐ │
│ │ 2. CrisisAgent │ ║ │ 3. RAGAgent │ │
│ │ (risk scoring) │ ║ │ (vector retrieval) │ │
│ └─────────────────────┘ ║ └─────────────────────┘ │
│ │ ║ │ │
│ └────────────╨──────────────┘ │
│ ▼ │
│ 4. WellnessAgent (aggregation + tip generation) │
└─────────────────────────────┬────────────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────────────────┐
│ Model Layer │
│ DistilRoBERTa │ MiniLM Embeddings │ ChromaDB Vector Store │
└──────────────────────────────────────────────────────────────────┘


**Why this design?** Crisis assessment and KB retrieval are independent of each other but both depend on classification → ideal for `asyncio.gather()`. Cuts total latency by ~40% vs sequential execution.

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|-------|------------|-----|
| **Backend** | FastAPI 0.104 | Async-first, auto OpenAPI, Pydantic v2 |
| **Frontend** | Streamlit 1.28 | Fast dashboards, no JS required |
| **ML Framework** | PyTorch 2.1 + Transformers 4.35 | Industry-standard HF ecosystem |
| **Embeddings** | sentence-transformers 2.7 (MiniLM) | 384-dim, 5× faster than BERT |
| **Vector DB** | ChromaDB 0.4 | Embedded, no separate service |
| **Validation** | Pydantic v2 | Type-safe schemas, strict validation |
| **Monitoring** | Prometheus + structlog | Industry-standard observability |
| **Testing** | pytest 7.4 + pytest-asyncio | 59 tests in <8s |
| **CI/CD** | GitHub Actions + Codecov | Auto-test on every push |
| **Linting** | Ruff 0.4 | 10-100× faster than flake8/black |

---

## 🚀 Quick Start

### Prerequisites
- Python **3.10+** (tested on 3.10.11)
- 8GB+ RAM (transformer models)
- Git

### 1. Clone & Setup

git clone https://github.com/JIYA-YDV/Mental-Health-Agentic-AI-Platform.git
cd Mental-Health-Agentic-AI-Platform

python -m venv venv

# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

2. Install Dependencies
   python -m pip install --upgrade pip
python -m pip install -r requirements.txt
3. Configure Environment
  # Windows
copy .env.example .env
# Mac/Linux
cp .env.example .env
4. Start Backend (Terminal 1)
python -m uvicorn backend.main:app --reload --port 8000
5. Start Frontend (Terminal 2)
streamlit run frontend/app.py
Access:

🖥️ UI: http://localhost:8501
🔌 API: http://localhost:8000
📚 Docs: http://localhost:8000/docs
📊 Metrics: http://localhost:8001/metrics

🧪 Testing
The project ships with a professional test suite (59 tests, 74.4% coverage, <8s runtime) using mocked ML models for fast feedback.


# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=backend --cov-report=term-missing

# Just unit tests (fastest)
pytest tests/unit/ -v

# Just integration tests
pytest tests/integration/ -v

# Generate HTML coverage report
pytest tests/ --cov=backend --cov-report=html
# Then open htmlcov/index.html
Test Architecture
text

tests/
├── conftest.py              # Shared fixtures with mocked ML models
├── unit/                    # 41 fast unit tests (no I/O)
│   ├── test_settings.py     # Config validation
│   ├── test_schemas.py      # Pydantic v2 contracts
│   ├── test_explainer.py    # Lexicon scoring + normalization
│   └── test_rag_pipeline.py # Fallback logic + KB integrity
├── integration/             # 15 integration tests
│   ├── test_api.py          # FastAPI endpoint contracts
│   └── test_orchestrator.py # Multi-agent flow
└── _legacy/                 # Archived auto-generated tests
Coverage Highlights
Module	Coverage
schemas.py, settings.py, orchestrator.py, logger.py	100%
explainer.py	87.8%
crisis_agent.py	83.3%
metrics.py	77.8%
routes.py, rag_pipeline.py	69–76%
Total	74.4%
📡 API Reference
POST /classify
Analyze text and return emotion + crisis assessment + recommendations.

Request:

JSON

{
  "text": "I've been feeling really anxious about work lately",
  "include_explanations": true,
  "session_id": "optional-id-123"
}
Response:

JSON

{
  "emotion": "Fear / Anxiety",
  "confidence": 0.89,
  "all_predictions": [
    {"label": "Fear / Anxiety", "score": 0.89},
    {"label": "Neutral",        "score": 0.08}
  ],
  "recommendations": [
    {
      "title": "4-7-8 Breathing Technique",
      "content": "Inhale for 4 seconds, hold for 7...",
      "relevance_score": 0.82,
      "category": "breathing",
      "source": "Mental Health Knowledge Base"
    }
  ],
  "crisis_assessment": {
    "is_crisis": false,
    "risk_level": "low",
    "risk_score": 0.05,
    "crisis_indicators": [],
    "immediate_resources": ["988 Suicide & Crisis Lifeline"]
  },
  "explanations": [
    {"token": "anxious", "weight": 0.95, "influence": "positive"},
    {"token": "work",    "weight": 0.40, "influence": "positive"}
  ],
  "explanation_summary": "Driven by mentions of anxious, work...",
  "processing_time_ms": 245.5,
  "model_version": "1.0.1",
  "timestamp": "2026-06-26T18:40:12.430718Z"
}
GET /health
JSON

{
  "status": "healthy",
  "version": "1.0.1",
  "models_loaded": true,
  "timestamp": "2026-06-26T18:40:12Z"
}
Full interactive docs: http://localhost:8000/docs

⚙️ Configuration
All settings are environment-variable driven via pydantic-settings. Override any value in .env:

env

# Application
DEBUG=false
LOG_LEVEL=INFO
PORT=8000

# Models
EMOTION_MODEL=j-hartmann/emotion-english-distilroberta-base
EMBEDDING_MODEL=all-MiniLM-L6-v2
MAX_SEQUENCE_LENGTH=512

# RAG
CHROMA_PERSIST_DIR=./chroma_db
TOP_K_RETRIEVAL=5
SIMILARITY_THRESHOLD=0.4
FALLBACK_THRESHOLD_DELTA=0.15

# Explainer
EXPLAINER_DISPLAY_THRESHOLD=0.25
EXPLAINER_MIN_TOKENS=3
EXPLAINER_MAX_TOKENS=10

# Crisis Detection
CRISIS_CONFIDENCE_THRESHOLD=0.75

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=8001
📁 Project Structure
text

Mental-Health-Agentic-AI-Platform/
├── .github/workflows/
│   └── ci.yml                       # GitHub Actions pipeline
├── backend/
│   ├── agents/                      # Multi-agent orchestration
│   │   ├── orchestrator.py
│   │   ├── classification_agent.py
│   │   ├── crisis_agent.py
│   │   ├── rag_agent.py
│   │   └── wellness_agent.py
│   ├── api/                         # FastAPI layer
│   │   ├── routes.py
│   │   ├── schemas.py               # Pydantic v2 contracts
│   │   └── middleware.py
│   ├── config/
│   │   └── settings.py              # Centralized config
│   ├── explainability/
│   │   └── explainer.py             # Lexicon-based attribution
│   ├── models/
│   │   ├── classifier.py            # DistilRoBERTa wrapper
│   │   ├── embeddings.py            # MiniLM wrapper
│   │   └── rag_pipeline.py          # ChromaDB integration
│   ├── monitoring/
│   │   ├── logger.py                # structlog setup
│   │   └── metrics.py               # Prometheus
│   └── main.py                      # FastAPI app + lifespan
├── frontend/
│   └── app.py                       # Streamlit dashboard
├── tests/
│   ├── conftest.py                  # Shared fixtures
│   ├── unit/                        # 41 unit tests
│   └── integration/                 # 15 integration tests
├── docker/                          # Container configs
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
├── pyproject.toml                   # pytest + coverage + ruff config
├── requirements.txt
├── .env.example
└── README.md
🐳 Docker Deployment

# Build and start everything
docker-compose -f docker/docker-compose.yml up --build

# Background mode
docker-compose -f docker/docker-compose.yml up -d --build

# View logs
docker-compose logs -f backend

# Stop
docker-compose down
📊 Implementation Status
Honest snapshot of what's built vs planned. This project values transparency over marketing.

Component	Status	Notes
Emotion Classifier (DistilRoBERTa, 7-class)	✅ Production	
Multi-Agent Orchestrator (async)	✅ Production	100% test coverage
RAG Pipeline (ChromaDB + fallback)	✅ Production	
Crisis Detection (keyword + confidence)	✅ Production	
Token Explainability (lexicon-based)	✅ Production	SHAP planned for v1.2
Streamlit Dashboard	✅ Production	
FastAPI Backend + auto-docs	✅ Production	
Test Suite (59 tests, 74.4% coverage)	✅ Production	
GitHub Actions CI	✅ Production	
Prometheus Metrics	🟡 Beta	Endpoint live; dashboards pending
Docker Compose Deployment	🟡 Beta	Works locally; cloud TBD
Live Public Demo	🔴 Roadmap	v1.2 target
ML Evaluation Harness (F1 / confusion matrix)	🔴 Roadmap	v1.1 target
Fine-tuned Domain Model	🔴 Roadmap	v1.2 target
LangGraph Migration	🔴 Roadmap	v1.3 target
MLflow Experiment Tracking	🔴 Roadmap	v1.3 target
See ROADMAP.md for upcoming work.

🤝 Contributing
Fork the repo
Create a feature branch: git checkout -b feat/your-feature
Make sure tests pass: pytest tests/ -v
Maintain coverage ≥ 70%: pytest --cov=backend --cov-fail-under=70
Commit using conventional commits: feat:, fix:, test:, docs:, refactor:
Push & open a Pull Request
CI will automatically run tests on Python 3.10 + 3.11.

⚠️ Safety & Ethics
This platform cannot:

Diagnose mental health conditions
Replace licensed therapists or psychiatrists
Handle emergency crises
If you or someone you know is in crisis:

🇺🇸 988 Suicide & Crisis Lifeline (call or text)
🇺🇸 Text HOME to 741741 (Crisis Text Line)
🌍 International directory
📄 License
MIT — see LICENSE.

🙏 Acknowledgments
HuggingFace Transformers — model ecosystem
j-hartmann — base emotion model
ChromaDB — embedded vector store
FastAPI & Streamlit — the magic that makes this work
