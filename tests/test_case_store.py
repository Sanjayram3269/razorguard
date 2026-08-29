from __future__ import annotations

import pandas as pd
import pytest

from razorguard.investigation.store import (
    CaseStore,
)
from razorguard.risk.case import RiskCase


def make_case(
    transaction_id: str = "T001",
) -> RiskCase:
    return RiskCase(
        transaction_id=transaction_id,
        risk_score=91.5,
        risk_level="CRITICAL",
        decision="BLOCK",
        primary_reason=(
            "high combined transaction "
            "and network risk"
        ),
        evidence=[
            "transaction amount is elevated",
            "elevated network/entity risk",
        ],
        model_probability=0.94,
        network_score=18.2,
    )


def test_create_and_get_case(tmp_path):
    store = CaseStore(
        tmp_path / "cases.parquet"
    )

    created = store.create(
        make_case(),
        actor="system",
    )

    assert (
        created["case_id"]
        == "CASE-T001"
    )

    assert created["status"] == "OPEN"

    assert (
        created["priority"]
        == "CRITICAL"
    )

    loaded = store.get(
        "CASE-T001"
    )

    assert loaded is not None

    assert (
        loaded["transaction_id"]
        == "T001"
    )

    assert (
        loaded["risk_score"]
        == 91.5
    )


def test_case_persists_across_store_instances(
    tmp_path,
):
    path = tmp_path / "cases.parquet"

    CaseStore(path).create(
        make_case()
    )

    new_store = CaseStore(path)

    loaded = new_store.get(
        "CASE-T001"
    )

    assert loaded is not None

    assert (
        loaded["decision"]
        == "BLOCK"
    )


def test_duplicate_case_is_rejected(
    tmp_path,
):
    store = CaseStore(
        tmp_path / "cases.parquet"
    )

    store.create(
        make_case()
    )

    with pytest.raises(
        ValueError
    ):
        store.create(
            make_case()
        )


def test_assignment_is_persisted(
    tmp_path,
):
    path = tmp_path / "cases.parquet"

    store = CaseStore(path)

    store.create(
        make_case()
    )

    updated = store.assign(
        "CASE-T001",
        "investigator-01",
        actor="lead",
    )

    assert (
        updated["assigned_to"]
        == "investigator-01"
    )

    loaded = store.get(
        "CASE-T001"
    )

    assert (
        loaded["assigned_to"]
        == "investigator-01"
    )


def test_status_lifecycle(
    tmp_path,
):
    store = CaseStore(
        tmp_path / "cases.parquet"
    )

    store.create(
        make_case()
    )

    store.update_status(
        "CASE-T001",
        "IN_REVIEW",
        actor="investigator-01",
    )

    store.escalate(
        "CASE-T001",
        actor="investigator-01",
    )

    case = store.get(
        "CASE-T001"
    )

    assert (
        case["status"]
        == "ESCALATED"
    )


def test_terminal_case_cannot_change(
    tmp_path,
):
    store = CaseStore(
        tmp_path / "cases.parquet"
    )

    store.create(
        make_case()
    )

    store.resolve(
        "CASE-T001",
        actor="investigator-01",
    )

    with pytest.raises(
        ValueError
    ):
        store.update_status(
            "CASE-T001",
            "IN_REVIEW",
        )


def test_dismiss_case(
    tmp_path,
):
    store = CaseStore(
        tmp_path / "cases.parquet"
    )

    store.create(
        make_case()
    )

    dismissed = store.dismiss(
        "CASE-T001",
        actor="investigator-01",
        details="confirmed legitimate",
    )

    assert (
        dismissed["status"]
        == "DISMISSED"
    )


def test_list_filters(
    tmp_path,
):
    store = CaseStore(
        tmp_path / "cases.parquet"
    )

    store.create(
        make_case("T001")
    )

    store.create(
        RiskCase(
            transaction_id="T002",
            risk_score=72.0,
            risk_level="HIGH",
            decision="REVIEW",
            primary_reason=(
                "elevated network/entity risk"
            ),
            evidence=[
                "network signal"
            ],
            model_probability=0.7,
            network_score=10.0,
        )
    )

    store.assign(
        "CASE-T002",
        "investigator-02",
    )

    high = store.list(
        priority="HIGH"
    )

    assigned = store.list(
        assigned_to="investigator-02"
    )

    assert len(high) == 1

    assert (
        high.iloc[0]["case_id"]
        == "CASE-T002"
    )

    assert len(assigned) == 1

    assert (
        assigned.iloc[0]["case_id"]
        == "CASE-T002"
    )


def test_audit_history_is_append_only(
    tmp_path,
):
    path = tmp_path / "cases.parquet"

    store = CaseStore(path)

    store.create(
        make_case(),
        actor="system",
    )

    store.assign(
        "CASE-T001",
        "investigator-01",
        actor="lead",
    )

    store.update_status(
        "CASE-T001",
        "IN_REVIEW",
        actor="investigator-01",
    )

    audit = store.audit(
        "CASE-T001"
    )

    assert len(audit) == 3

    assert list(
        audit["action"]
    ) == [
        "CASE_CREATED",
        "CASE_ASSIGNED",
        "STATUS_CHANGED",
    ]

    assert (
        audit.iloc[0]["to_status"]
        == "OPEN"
    )

    assert (
        audit.iloc[2]["to_status"]
        == "IN_REVIEW"
    )


def test_missing_case_operations_fail(
    tmp_path,
):
    store = CaseStore(
        tmp_path / "cases.parquet"
    )

    assert (
        store.get(
            "CASE-MISSING"
        )
        is None
    )

    with pytest.raises(
        KeyError
    ):
        store.assign(
            "CASE-MISSING",
            "investigator-01",
        )


def test_invalid_filters_are_rejected(
    tmp_path,
):
    store = CaseStore(
        tmp_path / "cases.parquet"
    )

    with pytest.raises(
        ValueError
    ):
        store.list(
            status="INVALID"
        )

    with pytest.raises(
        ValueError
    ):
        store.list(
            priority="INVALID"
        )


def test_case_artifacts_exist(
    tmp_path,
):
    path = tmp_path / "cases.parquet"

    store = CaseStore(path)

    store.create(
        make_case()
    )

    assert path.exists()

    assert (
        path.with_name(
            "cases.audit.parquet"
        ).exists()
    )

    frame = pd.read_parquet(
        path
    )

    assert isinstance(
        frame,
        pd.DataFrame,
    )

    assert len(frame) == 1