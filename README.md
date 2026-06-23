# 🧠 Mental Health Agentic AI Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.1-FF4B4B)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.1-EE4C2C)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**End-to-end Agentic AI platform for mental health intelligence** featuring Transformer-based NLP classification, RAG pipelines, multi-agent orchestration, explainability, automated ML workflows, evaluation, monitoring, and deployment.

> ⚠️ **IMPORTANT DISCLAIMER**: This is an AI research tool and NOT a substitute for professional mental health care.

---

## 🎯 Features

- **🔍 Emotion Classification**: DistilRoBERTa-based detection of 7 emotions (anger, disgust, fear, joy, neutral, sadness, surprise)
- **🚨 Crisis Detection**: Real-time risk assessment with keyword and confidence-based triggers
- **📚 RAG Pipeline**: Semantic search over curated mental health wellness knowledge base (ChromaDB)
- **🤖 Multi-Agent Orchestration**: Parallel execution of Classification, Crisis, RAG, and Wellness agents
- **🔮 Explainability**: Token-level attribution showing which words influenced predictions
- **📊 Monitoring**: Prometheus metrics + structured logging with structlog
- **🖥️ Interactive UI**: Streamlit frontend with real-time analysis and crisis alerts
- **🐳 Docker Ready**: Full containerization with docker-compose

---

## 🏗️ Architecture
┌─────────────────────────────────────────────────────────────────┐
│ Streamlit Frontend │
│ (Port 8501 - User Interface) │
└──────────────────────┬──────────────────────────────────────────┘
│ HTTP Requests
▼
┌─────────────────────────────────────────────────────────────────┐
│ FastAPI Backend (Port 8000) │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐ │
│ │ /health │ │ /classify │ │ /docs │ │
│ │ (Health) │ │ (Analysis) │ │ (Swagger UI) │ │
│ └──────────────┘ └──────────────┘ └──────────────────────┘ │
└──────────────────────┬──────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ Agent Orchestrator (Async) │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ 1. Classification Agent (Blocking) │ │
│ │ └─> DistilRoBERTa Emotion Classifier │ │
│ └──────────────────────────────────────────────────────────┘ │
│ │ │
│ ┌───────────────┴───────────────┐ │
│ ▼ ▼ │
│ ┌──────────────────────────┐ ┌──────────────────────────┐ │
│ │ 2. Crisis Agent │ │ 3. RAG Agent │ │
│ │ Risk Assessment │ │ Semantic Retrieval │ │
│ └──────────────────────────┘ └──────────────────────────┘ │
│ │ │ │
│ └───────────────┬───────────────┘ │
│ ▼ │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ 4. Wellness Agent (Tip Generation) │ │
│ └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ Model Layer │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐ │
│ │ Emotion │ │ Sentence │ │ ChromaDB │ │
│ │ Classifier │ │ Embeddings │ │ Vector Store │ │
│ │ (Transformers)│ │ (MiniLM) │ │ (Wellness KB) │ │
│ └──────────────┘ └──────────────┘ └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

text


---

## 🛠️ Tech Stack

| Component | Technology | Version |
|-----------|------------|---------|
| **Backend** | FastAPI | 0.104.1 |
| **Frontend** | Streamlit | 1.28.1 |
| **ML Framework** | PyTorch / Transformers | 2.1.1 / 4.35.2 |
| **Embeddings** | sentence-transformers | 2.7.0 |
| **Vector DB** | ChromaDB | 0.4.24 |
| **Monitoring** | Prometheus + structlog | 0.19.0 / 23.2.0 |
| **Testing** | pytest | 7.4.3 |
| **Deployment** | Docker + Docker Compose | - |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** (tested on 3.10.11)
- **Git**
- **Virtual environment** (venv recommended)
- **8GB+ RAM** (for transformer models)

### 1. Clone & Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd Mental-Health-Agentic-AI-Platform

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
2. Install Dependencies
Bash

# Upgrade pip first
python -m pip install --upgrade pip

# Install all requirements
python -m pip install -r requirements.txt

# Verify critical imports
python -c "import torch; import transformers; import sentence_transformers; print('✅ All ML packages OK')"
Note: First run will download ~500MB of model weights from HuggingFace (cached for subsequent runs).

3. Configure Environment
Bash

# Copy example environment file
copy .env.example .env  # Windows
# cp .env.example .env  # Mac/Linux

# Edit .env if needed (defaults work for local development)
4. Start Backend (Terminal 1)
Bash

# From project root (Mental-Health-Agentic-AI-Platform/)
python -m backend.main
Expected output:

text

INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Loading emotion classifier...
INFO:     Loading embedding model...
INFO:     Initializing RAG pipeline...
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
⏱️ First startup takes 2-3 minutes while models download and load into memory.

5. Start Frontend (Terminal 2)
Bash

# New terminal window, same directory
venv\Scripts\activate  # or source venv/bin/activate
streamlit run frontend/app.py
Access points:

🖥️ Frontend UI: http://localhost:8501
🔌 API: http://localhost:8000
📚 API Docs: http://localhost:8000/docs
📊 Metrics: http://localhost:8001/metrics
🐳 Docker Deployment
Option A: Docker Compose (Recommended)
Bash

# Build and start all services
docker-compose -f docker/docker-compose.yml up --build

# Or run in background
docker-compose -f docker/docker-compose.yml up -d --build

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Stop
docker-compose down
Option B: Individual Containers
Bash

# Build backend
docker build -f docker/Dockerfile.backend -t mh-backend .

# Build frontend  
docker build -f docker/Dockerfile.frontend -t mh-frontend .

# Run backend
docker run -p 8000:8000 -p 8001:8001 mh-backend

# Run frontend
docker run -p 8501:8501 mh-frontend
🧪 Testing
Bash

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v
pytest tests/test_agents.py -v
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ -v --cov=backend --cov-report=html
📡 API Reference
POST /classify
Analyze text and return emotion, crisis assessment, and recommendations.

Request:

JSON

{
  "text": "I've been feeling really anxious about work lately",
  "include_explanations": false,
  "session_id": "optional-session-id"
}
Response:

JSON

{
  "emotion": "Fear / Anxiety",
  "confidence": 0.89,
  "all_predictions": [
    {"label": "Fear / Anxiety", "score": 0.89},
    {"label": "Neutral", "score": 0.08},
    {"label": "Sadness / Depression", "score": 0.02}
  ],
  "recommendations": [
    {
      "title": "4-7-8 Breathing Technique",
      "content": "Inhale for 4 seconds, hold for 7...",
      "relevance_score": 0.92,
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
  "processing_time_ms": 245.5,
  "timestamp": "2024-01-15T10:30:00",
  "model_version": "1.0.0"
}
GET /health
Health check endpoint.

Response:

JSON

{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": true,
  "timestamp": "2024-01-15T10:30:00"
}
⚙️ Environment Variables
Create a .env file in the project root:

env

# Application
DEBUG=false
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8000

# Models
EMOTION_MODEL=j-hartmann/emotion-english-distilroberta-base
EMBEDDING_MODEL=all-MiniLM-L6-v2
MAX_SEQUENCE_LENGTH=512

# RAG
CHROMA_PERSIST_DIR=./chroma_db
TOP_K_RETRIEVAL=3
SIMILARITY_THRESHOLD=0.7

# Crisis Detection
CRISIS_CONFIDENCE_THRESHOLD=0.75

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=8001
📁 Project Structure
text

Mental-Health-Agentic-AI-Platform/
├── backend/
│   ├── agents/              # Multi-agent orchestration
│   │   ├── orchestrator.py
│   │   ├── classification_agent.py
│   │   ├── crisis_agent.py
│   │   ├── rag_agent.py
│   │   └── wellness_agent.py
│   ├── models/              # ML models layer
│   │   ├── classifier.py    # DistilRoBERTa
│   │   ├── embeddings.py    # MiniLM embeddings
│   │   └── rag_pipeline.py  # ChromaDB integration
│   ├── api/                 # FastAPI routes
│   │   ├── routes.py
│   │   ├── schemas.py
│   │   └── middleware.py
│   ├── monitoring/          # Logging & metrics
│   │   ├── logger.py
│   │   └── metrics.py
│   ├── explainability/      # SHAP/LIME explanations
│   │   └── explainer.py
│   ├── config/              # Settings management
│   │   └── settings.py
│   └── main.py              # Application entry point
├── frontend/
│   └── app.py               # Streamlit UI
├── tests/                   # Test suite
│   ├── test_api.py
│   ├── test_agents.py
│   └── test_models.py
├── docker/                  # Container configs
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
├── datasets/                # Data storage
├── notebooks/               # Jupyter exploration
├── docs/                    # Documentation
├── requirements.txt         # Python dependencies
└── .env.example             # Environment template
🚨 Troubleshooting
ImportError: cannot import name 'cached_download'
Cause: sentence-transformers==2.2.2 incompatible with newer huggingface_hub

Fix:

Bash

python -m pip install sentence-transformers==2.7.0
ModuleNotFoundError: No module named 'backend'
Cause: Running from wrong directory or missing __init__.py

Fix:

Bash

# Must run from project root, not inside backend/
cd D:\projects\Mental-Health-Agentic-AI-Platform
python -m backend.main
API Offline in Frontend
Cause: Backend not running or CORS blocked

Fix:

Check backend is running: curl http://localhost:8000/health
Verify CORS origins in backend/main.py match your frontend URL
Check Windows Firewall isn't blocking port 8000
ChromaDB Persistence Errors
Cause: Windows file permissions or locked database

Fix:

Bash

# Stop backend, delete corrupted DB, restart
rmdir /s /q chroma_db  # Windows
rm -rf chroma_db       # Mac/Linux
CUDA/GPU Issues
Cause: PyTorch trying to use unavailable GPU

Fix: The code auto-detects CPU/CUDA. If you want to force CPU:

Python

# In backend/models/classifier.py
device = -1  # Force CPU
🤝 Contributing
Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit changes (git commit -m 'feat: add amazing feature')
Push to branch (git push origin feature/amazing-feature)
Open a Pull Request
Please read CONTRIBUTING.md for details on code style and testing requirements.

⚠️ Safety & Ethics
This platform uses AI to assist with mental health awareness but cannot:

Diagnose mental health conditions
Replace licensed therapists or psychiatrists
Handle emergency crises (always call 911 or 988)
Crisis Resources:

🇺🇸 988 Suicide & Crisis Lifeline: Call or text 988
🇺🇸 Crisis Text Line: Text HOME to 741741
🌍 International: Find your country at https://www.iasp.info/resources/Crisis_Centres/
📄 License
Distributed under the MIT License. See LICENSE for more information.

🙏 Acknowledgments
HuggingFace Transformers for the DistilRoBERTa model
j-hartmann for the emotion classification model
ChromaDB for vector storage
FastAPI and Streamlit for the excellent frameworks