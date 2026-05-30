# backend/api/middleware.py
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import structlog

logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every request with method, path, status, and duration.
    Adds X-Process-Time header to all responses.
    """

    async def dispatch(self, request: Request, call_next):
        start = time.time()

        logger.info(
            "Request received",
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else "unknown"
        )

        response = await call_next(request)

        duration_ms = round((time.time() - start) * 1000, 2)
        response.headers["X-Process-Time"] = str(duration_ms)

        logger.info(
            "Response sent",
            status_code=response.status_code,
            duration_ms=duration_ms
        )

        return response