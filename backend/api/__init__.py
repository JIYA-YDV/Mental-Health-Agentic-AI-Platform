# backend/api/__init__.py
"""
API Package

FastAPI route definitions, request/response schemas,
and middleware components.

Components:
- routes.py     : GET /health, POST /classify endpoints
- schemas.py    : Pydantic request and response models
- middleware.py : Request logging and timing middleware

Usage:
    from backend.api.routes import router
    app.include_router(router)
"""

__all__ = [
    "routes",
    "schemas",
]