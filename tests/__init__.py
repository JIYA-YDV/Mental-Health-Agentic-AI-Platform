"""
Tests Package

Unit and integration test suite for the Mental Health
Agentic AI Platform.

Test modules:
- test_api.py    : FastAPI endpoint integration tests
- test_agents.py : Agent unit tests (classification, crisis, RAG)
- test_models.py : ML model layer unit tests

Run all tests:
    pytest tests/ -v

Run specific module:
    pytest tests/test_api.py -v
    pytest tests/test_agents.py -v
    pytest tests/test_models.py -v

Run with coverage:
    pytest tests/ -v --cov=backend --cov-report=html
"""