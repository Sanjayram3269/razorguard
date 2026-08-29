from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from razorguard.risk.case import RiskCase


VALID_STATUSES = {
    "OPEN",
    "IN_REVIEW",
    "ESCALATED",
    "RESOLVED",
    "DISMISSED",
}

VALID_PRIORITIES = {
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
}


def _utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _case_id(transaction_id: str) -> str:
    """Create a stable case identifier from a transaction identifier."""
    if not transaction_id:
        raise ValueError(
            "transaction_id must not be empty"
        )

    return f"CASE-{transaction_id}"


class CaseStore:
    """
    Persistent investigator case store.

    The store maintains:
        - current case state
        - investigator assignment
        - risk evidence
        - investigation narrative
        - append-only audit history

    Storage:
        cases.parquet
        cases.audit.parquet
    """

    CASE_COLUMNS = [
        "case_id",
        "transaction_id",
        "status",
        "priority",
        "assigned_to",
        "created_at",
        "updated_at",
        "risk_score",
        "risk_level",
        "decision",
        "primary_reason",
        "evidence_text",
        "model_probability",
        "network_score",
        "investigation_narrative",
    ]

    AUDIT_COLUMNS = [
        "case_id",
        "timestamp",
        "action",
        "actor",
        "from_status",
        "to_status",
        "details",
    ]

    def __init__(
        self,
        path: str | Path,
    ) -> None:
        self.path = Path(path)

        self.audit_path = self.path.with_name(
            f"{self.path.stem}.audit.parquet"
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load_cases(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(
                columns=self.CASE_COLUMNS
            )

        frame = pd.read_parquet(
            self.path
        )

        for column in self.CASE_COLUMNS:
            if column not in frame.columns:
                frame[column] = None

        return frame[
            self.CASE_COLUMNS
        ].copy()

    def _load_audit(self) -> pd.DataFrame:
        if not self.audit_path.exists():
            return pd.DataFrame(
                columns=self.AUDIT_COLUMNS
            )

        frame = pd.read_parquet(
            self.audit_path
        )

        for column in self.AUDIT_COLUMNS:
            if column not in frame.columns:
                frame[column] = None

        return frame[
            self.AUDIT_COLUMNS
        ].copy()

    def _write_cases(
        self,
        frame: pd.DataFrame,
    ) -> None:
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        frame.to_parquet(
            self.path,
            index=False,
        )

    def _write_audit(
        self,
        frame: pd.DataFrame,
    ) -> None:
        self.audit_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        frame.to_parquet(
            self.audit_path,
            index=False,
        )

    def _append_audit(
        self,
        *,
        case_id: str,
        action: str,
        actor: str,
        from_status: str | None,
        to_status: str | None,
        details: str = "",
    ) -> None:
        audit = self._load_audit()

        event = pd.DataFrame(
            [
                {
                    "case_id": case_id,
                    "timestamp": _utc_now(),
                    "action": action,
                    "actor": actor,
                    "from_status": from_status,
                    "to_status": to_status,
                    "details": details,
                }
            ]
        )

        audit = pd.concat(
            [
                audit,
                event,
            ],
            ignore_index=True,
        )

        self._write_audit(
            audit
        )

    # ------------------------------------------------------------------
    # Case creation
    # ------------------------------------------------------------------

    def create(
        self,
        case: RiskCase | dict[str, Any],
        *,
        actor: str = "system",
        investigation_narrative: str = "",
    ) -> dict[str, Any]:
        """
        Persist a new investigation case.

        Case identity is deterministic and derived from
        transaction_id.
        """

        if isinstance(case, RiskCase):
            data = asdict(case)
        else:
            data = dict(case)

        transaction_id = str(
            data.get(
                "transaction_id",
                "",
            )
        )

        if not transaction_id:
            raise ValueError(
                "case must contain transaction_id"
            )

        case_id = _case_id(
            transaction_id
        )

        cases = self._load_cases()

        existing_ids = set(
            cases["case_id"]
            .dropna()
            .astype(str)
        )

        if case_id in existing_ids:
            raise ValueError(
                f"case already exists: {case_id}"
            )

        risk_level = str(
            data.get(
                "risk_level",
                "MEDIUM",
            )
        ).upper()

        if risk_level not in VALID_PRIORITIES:
            risk_level = "MEDIUM"

        now = _utc_now()

        row = {
            "case_id": case_id,
            "transaction_id": transaction_id,
            "status": "OPEN",
            "priority": risk_level,
            "assigned_to": None,
            "created_at": now,
            "updated_at": now,
            "risk_score": float(
                data.get(
                    "risk_score",
                    0.0,
                )
            ),
            "risk_level": data.get(
                "risk_level"
            ),
            "decision": data.get(
                "decision"
            ),
            "primary_reason": data.get(
                "primary_reason",
                "",
            ),
            "evidence_text": " | ".join(
                str(value)
                for value in data.get(
                    "evidence",
                    [],
                )
            ),
            "model_probability": float(
                data.get(
                    "model_probability",
                    0.0,
                )
            ),
            "network_score": float(
                data.get(
                    "network_score",
                    0.0,
                )
            ),
            "investigation_narrative": (
                investigation_narrative
            ),
        }

        cases = pd.concat(
            [
                cases,
                pd.DataFrame([row]),
            ],
            ignore_index=True,
        )

        self._write_cases(
            cases
        )

        self._append_audit(
            case_id=case_id,
            action="CASE_CREATED",
            actor=actor,
            from_status=None,
            to_status="OPEN",
            details="",
        )

        return dict(row)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(
        self,
        case_id: str,
    ) -> dict[str, Any] | None:
        """Return a single case by case ID."""

        cases = self._load_cases()

        matches = cases[
            cases["case_id"]
            .astype(str)
            == str(case_id)
        ]

        if matches.empty:
            return None

        return matches.iloc[0].to_dict()

    def list(
        self,
        *,
        status: str | None = None,
        assigned_to: str | None = None,
        priority: str | None = None,
    ) -> pd.DataFrame:
        """List cases with optional filters."""

        cases = self._load_cases()

        if status is not None:
            status = status.upper()

            if status not in VALID_STATUSES:
                raise ValueError(
                    f"invalid status: {status}"
                )

            cases = cases[
                cases["status"]
                == status
            ]

        if assigned_to is not None:
            cases = cases[
                cases["assigned_to"]
                == assigned_to
            ]

        if priority is not None:
            priority = priority.upper()

            if priority not in VALID_PRIORITIES:
                raise ValueError(
                    f"invalid priority: {priority}"
                )

            cases = cases[
                cases["priority"]
                == priority
            ]

        return cases.reset_index(
            drop=True
        )

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------

    def assign(
        self,
        case_id: str,
        investigator: str,
        *,
        actor: str = "system",
    ) -> dict[str, Any]:
        """Assign a case to an investigator."""

        if not investigator:
            raise ValueError(
                "investigator must not be empty"
            )

        case = self.get(
            case_id
        )

        if case is None:
            raise KeyError(
                f"case not found: {case_id}"
            )

        updated = self._update_case(
            case_id,
            {
                "assigned_to": investigator,
            },
            action="CASE_ASSIGNED",
            actor=actor,
            details=(
                f"assigned_to={investigator}"
            ),
        )

        return updated

    # ------------------------------------------------------------------
    # Status lifecycle
    # ------------------------------------------------------------------

    def update_status(
        self,
        case_id: str,
        status: str,
        *,
        actor: str = "system",
        details: str = "",
    ) -> dict[str, Any]:
        """Transition a case to a valid operational state."""

        status = status.upper()

        if status not in VALID_STATUSES:
            raise ValueError(
                f"invalid status: {status}"
            )

        case = self.get(
            case_id
        )

        if case is None:
            raise KeyError(
                f"case not found: {case_id}"
            )

        current = str(
            case["status"]
        )

        if current in {
            "RESOLVED",
            "DISMISSED",
        }:
            raise ValueError(
                f"case {case_id} is already terminal"
            )

        if status == current:
            raise ValueError(
                f"case {case_id} is already "
                f"in {status}"
            )

        return self._update_case(
            case_id,
            {
                "status": status,
            },
            action="STATUS_CHANGED",
            actor=actor,
            from_status=current,
            to_status=status,
            details=details,
        )

    def resolve(
        self,
        case_id: str,
        *,
        actor: str = "system",
        details: str = "",
    ) -> dict[str, Any]:
        """Resolve a case."""

        return self.update_status(
            case_id,
            "RESOLVED",
            actor=actor,
            details=details,
        )

    def dismiss(
        self,
        case_id: str,
        *,
        actor: str = "system",
        details: str = "",
    ) -> dict[str, Any]:
        """Dismiss a case."""

        return self.update_status(
            case_id,
            "DISMISSED",
            actor=actor,
            details=details,
        )

    def escalate(
        self,
        case_id: str,
        *,
        actor: str = "system",
        details: str = "",
    ) -> dict[str, Any]:
        """Escalate a case."""

        return self.update_status(
            case_id,
            "ESCALATED",
            actor=actor,
            details=details,
        )

    # ------------------------------------------------------------------
    # Internal update
    # ------------------------------------------------------------------

    def _update_case(
        self,
        case_id: str,
        updates: dict[str, Any],
        *,
        action: str,
        actor: str,
        from_status: str | None = None,
        to_status: str | None = None,
        details: str = "",
    ) -> dict[str, Any]:
        cases = self._load_cases()

        mask = (
            cases["case_id"]
            .astype(str)
            == str(case_id)
        )

        if not mask.any():
            raise KeyError(
                f"case not found: {case_id}"
            )

        index = cases.index[
            mask
        ][0]

        for key, value in updates.items():
            if key not in self.CASE_COLUMNS:
                raise ValueError(
                    f"unsupported case field: {key}"
                )

            cases.at[
                index,
                key,
            ] = value

        cases.at[
            index,
            "updated_at",
        ] = _utc_now()

        self._write_cases(
            cases
        )

        self._append_audit(
            case_id=case_id,
            action=action,
            actor=actor,
            from_status=from_status,
            to_status=to_status,
            details=details,
        )

        return cases.loc[
            index
        ].to_dict()

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def audit(
        self,
        case_id: str,
    ) -> pd.DataFrame:
        """Return the complete audit history for a case."""

        audit = self._load_audit()

        return (
            audit[
                audit["case_id"]
                .astype(str)
                == str(case_id)
            ]
            .reset_index(drop=True)
        )