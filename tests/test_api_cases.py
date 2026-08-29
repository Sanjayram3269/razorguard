from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

import razorguard.api.routes as routes
from razorguard.api.app import app


client = TestClient(app)


def _case_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": "CASE-001",
                "transaction_id": "TX-001",
                "status": "OPEN",
                "priority": "HIGH",
                "assigned_to": None,
                "created_at": "2026-08-29T10:00:00",
                "updated_at": "2026-08-29T10:00:00",
                "risk_score": 91.5,
                "risk_level": "CRITICAL",
                "decision": "BLOCK",
                "primary_reason": "High network risk",
                "evidence_text": "shared device | velocity spike",
                "model_probability": 0.97,
                "network_score": 0.88,
                "investigation_narrative": (
                    "Transaction requires investigation."
                ),
            },
            {
                "case_id": "CASE-002",
                "transaction_id": "TX-002",
                "status": "ASSIGNED",
                "priority": "MEDIUM",
                "assigned_to": "analyst-2",
                "created_at": "2026-08-29T10:05:00",
                "updated_at": "2026-08-29T10:10:00",
                "risk_score": 62.0,
                "risk_level": "MEDIUM",
                "decision": "REVIEW",
                "primary_reason": "Behavioral deviation",
                "evidence_text": "amount anomaly",
                "model_probability": 0.71,
                "network_score": 0.35,
                "investigation_narrative": (
                    "Transaction shows behavioral deviation."
                ),
            },
        ]
    )


class FakeCaseStore:
    def __init__(self, path):
        self.path = path

    def list(
        self,
        status=None,
        assigned_to=None,
        priority=None,
    ):
        frame = _case_frame()

        if status is not None:
            frame = frame[
                frame["status"] == status
            ]

        if assigned_to is not None:
            frame = frame[
                frame["assigned_to"] == assigned_to
            ]

        if priority is not None:
            frame = frame[
                frame["priority"] == priority
            ]

        return frame.reset_index(drop=True)

    def get(self, case_id):
        frame = _case_frame()

        matches = frame[
            frame["case_id"] == case_id
        ]

        if matches.empty:
            return None

        return matches.iloc[0].to_dict()

    def assign(
        self,
        case_id,
        investigator,
        *,
        actor="system",
    ):
        case = self.get(case_id)

        if case is None:
            raise KeyError(
                f"case not found: {case_id}"
            )

        case["assigned_to"] = investigator
        case["status"] = "ASSIGNED"

        return case

    def update_status(
        self,
        case_id,
        status,
        *,
        actor="system",
        details="",
    ):
        case = self.get(case_id)

        if case is None:
            raise KeyError(
                f"case not found: {case_id}"
            )

        case["status"] = status

        return case

    def audit(self, case_id):
        if self.get(case_id) is None:
            return pd.DataFrame()

        return pd.DataFrame(
            [
                {
                    "case_id": case_id,
                    "timestamp": "2026-08-29T10:15:00",
                    "action": "STATUS_CHANGED",
                    "actor": "analyst-1",
                    "from_status": "OPEN",
                    "to_status": "ASSIGNED",
                    "details": "Started investigation",
                },
                {
                    "case_id": case_id,
                    "timestamp": "2026-08-29T10:20:00",
                    "action": "STATUS_CHANGED",
                    "actor": "analyst-1",
                    "from_status": "ASSIGNED",
                    "to_status": "INVESTIGATING",
                    "details": "Reviewing evidence",
                },
            ]
        )


def test_list_cases(monkeypatch):
    monkeypatch.setattr(
        routes,
        "CaseStore",
        FakeCaseStore,
    )

    response = client.get(
        "/v1/cases"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["total"] == 2
    assert len(body["cases"]) == 2
    assert body["cases"][0]["case_id"] == "CASE-001"


def test_list_cases_with_filters(monkeypatch):
    monkeypatch.setattr(
        routes,
        "CaseStore",
        FakeCaseStore,
    )

    response = client.get(
        "/v1/cases",
        params={
            "status": "ASSIGNED",
            "assigned_to": "analyst-2",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["total"] == 1
    assert body["cases"][0]["case_id"] == "CASE-002"


def test_get_case(monkeypatch):
    monkeypatch.setattr(
        routes,
        "CaseStore",
        FakeCaseStore,
    )

    response = client.get(
        "/v1/cases/CASE-001"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["case_id"] == "CASE-001"
    assert body["risk_level"] == "CRITICAL"
    assert body["decision"] == "BLOCK"
    assert body["risk_score"] == 91.5


def test_get_missing_case(monkeypatch):
    monkeypatch.setattr(
        routes,
        "CaseStore",
        FakeCaseStore,
    )

    response = client.get(
        "/v1/cases/DOES-NOT-EXIST"
    )

    assert response.status_code == 404


def test_assign_case(monkeypatch):
    monkeypatch.setattr(
        routes,
        "CaseStore",
        FakeCaseStore,
    )

    response = client.post(
        "/v1/cases/CASE-001/assign",
        json={
            "investigator": "analyst-7",
            "actor": "lead-1",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["case_id"] == "CASE-001"
    assert body["assigned_to"] == "analyst-7"
    assert body["status"] == "ASSIGNED"


def test_assign_missing_case(monkeypatch):
    monkeypatch.setattr(
        routes,
        "CaseStore",
        FakeCaseStore,
    )

    response = client.post(
        "/v1/cases/UNKNOWN/assign",
        json={
            "investigator": "analyst-7",
        },
    )

    assert response.status_code == 404


def test_transition_case(monkeypatch):
    monkeypatch.setattr(
        routes,
        "CaseStore",
        FakeCaseStore,
    )

    response = client.post(
        "/v1/cases/CASE-001/transition",
        json={
            "status": "INVESTIGATING",
            "actor": "analyst-7",
            "details": "Evidence review started",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["case_id"] == "CASE-001"
    assert body["status"] == "INVESTIGATING"


def test_transition_missing_case(monkeypatch):
    monkeypatch.setattr(
        routes,
        "CaseStore",
        FakeCaseStore,
    )

    response = client.post(
        "/v1/cases/UNKNOWN/transition",
        json={
            "status": "INVESTIGATING",
        },
    )

    assert response.status_code == 404


def test_case_audit(monkeypatch):
    monkeypatch.setattr(
        routes,
        "CaseStore",
        FakeCaseStore,
    )

    response = client.get(
        "/v1/cases/CASE-001/audit"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["case_id"] == "CASE-001"
    assert body["total"] == 2
    assert len(body["events"]) == 2

    assert (
        body["events"][0]["from_status"]
        == "OPEN"
    )

    assert (
        body["events"][1]["to_status"]
        == "INVESTIGATING"
    )


def test_audit_missing_case(monkeypatch):
    monkeypatch.setattr(
        routes,
        "CaseStore",
        FakeCaseStore,
    )

    response = client.get(
        "/v1/cases/UNKNOWN/audit"
    )

    assert response.status_code == 404


def test_case_assign_validation(monkeypatch):
    monkeypatch.setattr(
        routes,
        "CaseStore",
        FakeCaseStore,
    )

    response = client.post(
        "/v1/cases/CASE-001/assign",
        json={
            "investigator": "",
        },
    )

    assert response.status_code == 422