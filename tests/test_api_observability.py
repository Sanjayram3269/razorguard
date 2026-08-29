from __future__ import annotations

from fastapi.testclient import TestClient

from razorguard.api.app import app


client = TestClient(app)


def test_health_returns_request_id():
    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    request_id = response.headers.get(
        "X-Request-ID"
    )

    assert request_id
    assert len(request_id) > 10


def test_health_preserves_client_request_id():
    request_id = "demo-request-123"

    response = client.get(
        "/health",
        headers={
            "X-Request-ID": request_id
        },
    )

    assert response.status_code == 200

    assert (
        response.headers[
            "X-Request-ID"
        ]
        == request_id
    )


def test_health_returns_processing_time():
    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    process_time = response.headers.get(
        "X-Process-Time-Ms"
    )

    assert process_time is not None

    assert float(
        process_time
    ) >= 0.0


def test_ready_returns_request_id():
    response = client.get(
        "/ready"
    )

    assert response.status_code == 200

    assert response.headers.get(
        "X-Request-ID"
    )


def test_request_ids_are_unique():
    first = client.get(
        "/health"
    )

    second = client.get(
        "/health"
    )

    first_id = first.headers[
        "X-Request-ID"
    ]

    second_id = second.headers[
        "X-Request-ID"
    ]

    assert first_id
    assert second_id
    assert first_id != second_id