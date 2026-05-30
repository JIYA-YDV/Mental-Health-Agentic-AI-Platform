# backend/monitoring/metrics.py
from prometheus_client import (
    Counter, Histogram, Gauge,
    start_http_server, REGISTRY
)
import structlog

logger = structlog.get_logger(__name__)

# ── Prometheus Metrics ─────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "mh_platform_requests_total",
    "Total classification requests",
    ["emotion", "risk_level"]
)

REQUEST_LATENCY = Histogram(
    "mh_platform_request_duration_ms",
    "Request processing time in milliseconds",
    buckets=[50, 100, 250, 500, 1000, 2500, 5000]
)

CRISIS_ALERTS = Counter(
    "mh_platform_crisis_alerts_total",
    "Total crisis-level detections",
    ["risk_level"]
)

ACTIVE_SESSIONS = Gauge(
    "mh_platform_active_sessions",
    "Currently active user sessions"
)

CONFIDENCE_HISTOGRAM = Histogram(
    "mh_platform_confidence_score",
    "Distribution of model confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)


def record_request(
    emotion: str,
    risk_level: str,
    latency_ms: float,
    confidence: float,
    is_crisis: bool
) -> None:
    """Record metrics for a completed request."""
    try:
        REQUEST_COUNT.labels(
            emotion=emotion,
            risk_level=risk_level
        ).inc()

        REQUEST_LATENCY.observe(latency_ms)
        CONFIDENCE_HISTOGRAM.observe(confidence)

        if is_crisis:
            CRISIS_ALERTS.labels(risk_level=risk_level).inc()

    except Exception as e:
        # Never let metrics crash the main app
        logger.warning("Metrics recording failed", error=str(e))


def start_metrics_server() -> None:
    """
    Start Prometheus metrics HTTP server.
    Silently skips if port is already in use.
    """
    try:
        from backend.config.settings import settings
        if settings.ENABLE_METRICS:
            start_http_server(settings.METRICS_PORT)
            logger.info(
                "Metrics server started",
                port=settings.METRICS_PORT,
                url=f"http://localhost:{settings.METRICS_PORT}/metrics"
            )
    except OSError as e:
        # Port already in use — ignore, metrics still work internally
        logger.warning(
            "Metrics server port already in use, skipping",
            error=str(e)
        )
    except Exception as e:
        logger.warning("Could not start metrics server", error=str(e))