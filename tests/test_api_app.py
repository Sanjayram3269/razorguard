from __future__ import annotations

from fastapi.testclient import TestClient

from razorguard.api.app import APP_VERSION, app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "razorguard"
    assert data["version"] == APP_VERSION


def test_ready():
    response = client.get("/ready")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ready"
    assert data["service"] == "razorguard"


def test_openapi_available():
    response = client.get("/openapi.json")

    assert response.status_code == 200

    data = response.json()

    assert data["info"]["title"] == (
        "RazorGuard Risk Intelligence API"
    )

    assert "/health" in data["paths"]
    assert "/ready" in data["paths"]