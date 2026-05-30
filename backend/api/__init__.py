"""
API Package

FastAPI route definitions, request/response schemas,
and middleware components.

Components:
- routes.py     : GET /health, POST /classify endpoints
- schemas.py    : Pydantic request and response models
- middleware.py : Request logging and timing middleware

Endpoints:
    GET  /health    → HealthResponse
    POST /classify  → ClassificationResponse

Usage:
    from backend.api.routes import router
    app.include_router(router)
"""

from backend.api import routes
from backend.api import schemas

__all__ = [
    "routes",
    "schemas",
]