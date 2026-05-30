from prometheus_client import Counter, Histogram, Gauge, start_http_server
from backend.config.settings import settings
import structlog

logger = structlog.get_logger(__name__)

# ── Prometheus Metrics ──────────────────────────────────────────────────────

# Request counters
REQUEST_COUNT = Counter(
    "mh_platform_requests_total",
    "Total classification requests",
    ["emotion", "risk_level"]
)

# Latency histogram
REQUEST_LATENCY = Histogram(
    "mh_platform_request_duration_ms",
    "Request processing time in milliseconds",
    buckets=[50, 100, 250, 500, 1000, 2500, 5000]
)

# Crisis alerts counter
CRISIS_ALERTS = Counter(
    "mh_platform_crisis_alerts_total",
    "Total crisis-level detections",
    ["risk_level"]
)

# Active sessions gauge
ACTIVE_SESSIONS = Gauge(
    "mh_platform_active_sessions",
    "Currently active user sessions"
)

# Model confidence distribution
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
    REQUEST_COUNT.labels(
        emotion=emotion,
        risk_level=risk_level
    ).inc()

    REQUEST_LATENCY.observe(latency_ms)
    CONFIDENCE_HISTOGRAM.observe(confidence)

    if is_crisis:
        CRISIS_ALERTS.labels(risk_level=risk_level).inc()


def start_metrics_server() -> None:
    """Start Prometheus metrics HTTP server."""
    if settings.ENABLE_METRICS:
        try:
            start_http_server(settings.METRICS_PORT)
            logger.info(
                "Metrics server started",
                port=settings.METRICS_PORT
            )
        except Exception as e:
            logger.warning("Could not start metrics server", error=str(e))