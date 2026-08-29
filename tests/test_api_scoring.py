from __future__ import annotations

from fastapi.testclient import TestClient

import razorguard.api.routes as routes
from razorguard.api.app import app


client = TestClient(app)


def transaction_payload() -> dict:
    return {
        "transaction_id": "API-T001",
        "account_id": "A001",
        "merchant_id": "M001",
        "device_id": "D001",
        "timestamp": "2026-08-29T12:00:00",
        "amount": 5000.0,
        "ip_country": "US",
        "shipping_country": "IN",
        "payment_method": "card",
        "merchant_category": "electronics",
    }


class FakeModel:
    def predict_proba(self, X):
        return [
            [0.04, 0.96]
            for _ in range(len(X))
        ]


def test_score_endpoint():
    original_loader = routes.load_model

    routes.load_model = lambda: (
        FakeModel(),
        {
            "model": "test_model",
            "threshold": 0.74,
        },
    )

    try:
        response = client.post(
            "/v1/transactions/score",
            json=transaction_payload(),
        )
    finally:
        routes.load_model = original_loader

    assert response.status_code == 200

    data = response.json()

    assert data["transaction_id"] == "API-T001"
    assert 0 <= data["risk_score"] <= 100
    assert data["risk_level"] in {
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
    }
    assert data["decision"] in {
        "ALLOW",
        "REVIEW",
        "BLOCK",
    }

    assert data["model_probability"] == 0.96
    assert "evidence" in data
    assert isinstance(
        data["evidence"],
        list,
    )

    assert data["model"] == "test_model"
    assert data["model_threshold"] == 0.74


def test_score_endpoint_requires_transaction_id():
    payload = transaction_payload()

    del payload["transaction_id"]

    response = client.post(
        "/v1/transactions/score",
        json=payload,
    )

    assert response.status_code == 422


def test_score_endpoint_rejects_negative_amount():
    payload = transaction_payload()
    payload["amount"] = -10

    response = client.post(
        "/v1/transactions/score",
        json=payload,
    )

    assert response.status_code == 422


def test_score_endpoint_returns_503_when_model_missing():
    original_loader = routes.load_model

    def missing_model():
        raise FileNotFoundError(
            "test model missing"
        )

    routes.load_model = missing_model

    try:
        response = client.post(
            "/v1/transactions/score",
            json=transaction_payload(),
        )
    finally:
        routes.load_model = original_loader

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "test model missing"
    )