from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import Response

from razorguard.api.routes import router


APP_VERSION = "0.2.0"

logger = logging.getLogger(
    "razorguard.api"
)


app = FastAPI(
    title="RazorGuard Risk Intelligence API",
    description=(
        "Investigator-facing API for transaction risk scoring, "
        "case management, and fraud intelligence."
    ),
    version=APP_VERSION,
)

app.include_router(router)


@app.middleware("http")
async def request_observability(
    request: Request,
    call_next,
) -> Response:
    """
    Attach correlation and timing metadata to every API response.

    X-Request-ID:
        Client supplied IDs are preserved. Otherwise a UUID is
        generated for the request.

    X-Process-Time-Ms:
        Server-side processing time in milliseconds.
    """

    request_id = (
        request.headers.get(
            "X-Request-ID"
        )
        or str(uuid.uuid4())
    )

    start = time.perf_counter()

    response = await call_next(
        request
    )

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000.0

    response.headers[
        "X-Request-ID"
    ] = request_id

    response.headers[
        "X-Process-Time-Ms"
    ] = f"{elapsed_ms:.3f}"

    logger.info(
        "api_request "
        "method=%s "
        "path=%s "
        "status=%s "
        "request_id=%s "
        "duration_ms=%.3f",
        request.method,
        request.url.path,
        response.status_code,
        request_id,
        elapsed_ms,
    )

    return response


@app.get("/health")
def health() -> dict[str, str]:
    """Basic liveness check."""

    return {
        "status": "ok",
        "service": "razorguard",
        "version": APP_VERSION,
    }


@app.get("/ready")
def ready() -> dict[str, str]:
    """Readiness check for the API process."""

    return {
        "status": "ready",
        "service": "razorguard",
    }