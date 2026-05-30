"""
Monitoring Package

Observability layer for the platform including:
- Structured logging via structlog
- Prometheus metrics instrumentation

Components:
- logger.py  : setup_logging() configures structlog for dev/prod
- metrics.py : Prometheus counters, histograms, gauges

Metrics exposed at: http://localhost:8001/metrics

Prometheus metrics defined:
    mh_platform_requests_total          Counter
    mh_platform_request_duration_ms     Histogram
    mh_platform_crisis_alerts_total     Counter
    mh_platform_active_sessions         Gauge
    mh_platform_confidence_score        Histogram

Usage:
    from backend.monitoring.logger import setup_logging
    from backend.monitoring.metrics import record_request, start_metrics_server

    setup_logging()
    start_metrics_server()
"""

from backend.monitoring.logger import setup_logging
from backend.monitoring.metrics import record_request, start_metrics_server

__all__ = [
    "setup_logging",
    "record_request",
    "start_metrics_server",
]