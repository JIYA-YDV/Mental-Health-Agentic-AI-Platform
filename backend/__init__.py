# backend/monitoring/__init__.py
"""
Monitoring Package

Observability layer including structured logging and Prometheus metrics.

Usage:
    from backend.monitoring.logger import setup_logging
    from backend.monitoring.metrics import record_request, start_metrics_server
"""

__all__ = [
    "setup_logging",
    "record_request",
    "start_metrics_server",
]