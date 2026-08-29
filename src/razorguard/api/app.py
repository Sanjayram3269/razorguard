from __future__ import annotations

from fastapi import FastAPI

from razorguard.api.routes import router


APP_VERSION = "0.2.0"


app = FastAPI(
    title="RazorGuard Risk Intelligence API",
    description=(
        "Investigator-facing API for transaction risk scoring, "
        "case management, and fraud intelligence."
    ),
    version=APP_VERSION,
)

app.include_router(router)


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