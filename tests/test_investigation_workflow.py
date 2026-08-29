from __future__ import annotations

import pandas as pd

from razorguard.investigation.store import CaseStore
from razorguard.investigation.workflow import (
    assign_case,
    create_investigation_cases,
    get_case_audit,
    reopen_case,
    transition_case,
)


class FakeModel:
    def predict_proba(self, X):
        return pd.DataFrame(
            {
                0: [0.05] * len(X),
                1: [0.95] * len(X),
            }
        ).to_numpy()


def transactions():
    return pd.DataFrame(
        {
            "transaction_id": ["T001", "T002"],
            "account_id": ["A1", "A2"],
            "merchant_id": ["M1", "M2"],
            "device_id": ["D1", "D2"],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 12:00:00",
                    "2026-01-01 12:05:00",
                ]
            ),
            "amount": [100.0, 5000.0],
            "ip_country": ["IN", "US"],
            "shipping_country": ["IN", "IN"],
            "payment_method": ["upi", "card"],
            "merchant_category": [
                "food",
                "electronics",
            ],
            "is_chargeback": [0, 1],
        }
    )


def test_workflow_persists_actionable_cases(tmp_path):
    path = tmp_path / "cases.parquet"

    store = CaseStore(path)

    scored, queue = create_investigation_cases(
        transactions(),
        FakeModel(),
        store,
    )

    assert len(scored) == 2
    assert len(queue) >= 1

    assert path.exists()

    persisted = store.list()

    assert len(persisted) >= 1
    assert all(
        persisted["status"] == "OPEN"
    )


def test_workflow_excludes_allow_cases(tmp_path):
    path = tmp_path / "cases.parquet"

    store = CaseStore(path)

    class LowRiskModel:
        def predict_proba(self, X):
            return pd.DataFrame(
                {
                    0: [0.99] * len(X),
                    1: [0.01] * len(X),
                }
            ).to_numpy()

    scored, queue = create_investigation_cases(
        transactions().iloc[[0]],
        LowRiskModel(),
        store,
    )

    assert len(scored) == 1
    assert scored.iloc[0]["decision"] == "ALLOW"
    assert queue.empty
    assert store.list().empty


def test_workflow_can_include_allow_cases(tmp_path):
    path = tmp_path / "cases.parquet"

    store = CaseStore(path)

    class LowRiskModel:
        def predict_proba(self, X):
            return pd.DataFrame(
                {
                    0: [0.99] * len(X),
                    1: [0.01] * len(X),
                }
            ).to_numpy()

    _, queue = create_investigation_cases(
        transactions().iloc[[0]],
        LowRiskModel(),
        store,
        include_allowed=True,
    )

    assert len(store.list()) == 1
    assert len(queue) == 0


def test_case_can_be_reopened_and_assigned(tmp_path):
    path = tmp_path / "cases.parquet"

    store = CaseStore(path)

    create_investigation_cases(
        transactions(),
        FakeModel(),
        store,
    )

    case_id = store.list().iloc[0]["case_id"]

    reopened = reopen_case(
        path,
        case_id,
    )

    assert reopened is not None

    assigned = assign_case(
        path,
        case_id,
        "investigator-01",
    )

    assert assigned[
        "assigned_to"
    ] == "investigator-01"


def test_case_lifecycle_is_audited(tmp_path):
    path = tmp_path / "cases.parquet"

    store = CaseStore(path)

    create_investigation_cases(
        transactions(),
        FakeModel(),
        store,
    )

    case_id = store.list().iloc[0]["case_id"]

    assign_case(
        path,
        case_id,
        "investigator-01",
    )

    transition_case(
        path,
        case_id,
        "IN_REVIEW",
        actor="investigator-01",
    )

    transition_case(
        path,
        case_id,
        "RESOLVED",
        actor="investigator-01",
        details="investigation completed",
    )

    case = reopen_case(
        path,
        case_id,
    )

    assert case["status"] == "RESOLVED"

    audit = get_case_audit(
        path,
        case_id,
    )

    assert len(audit) == 4

    assert list(
        audit["action"]
    ) == [
        "CASE_CREATED",
        "CASE_ASSIGNED",
        "STATUS_CHANGED",
        "STATUS_CHANGED",
    ]

    assert audit.iloc[-1][
        "to_status"
    ] == "RESOLVED"


def test_workflow_is_deterministic_for_case_identity(tmp_path):
    path = tmp_path / "cases.parquet"

    store = CaseStore(path)

    create_investigation_cases(
        transactions(),
        FakeModel(),
        store,
    )

    ids = set(
        store.list()["case_id"]
    )

    assert ids
    assert all(
        case_id.startswith("CASE-T")
        for case_id in ids
    )