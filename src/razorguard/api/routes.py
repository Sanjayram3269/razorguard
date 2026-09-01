from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from razorguard.api.models import (
    AnalyticsDistributionItem,
    AnalyticsMetricResponse,
    AuditEventResponse,
    AuditResponse,
    CaseAssignRequest,
    CaseIntelligenceResponse,
    CaseListResponse,
    CaseResponse,
    CaseTransitionRequest,
    CopilotRequest,
    CopilotResponse,
    CopilotStatusResponse,
    DashboardActivityItem,
    DashboardActivityResponse,
    DashboardDistributionItem,
    DashboardDistributionResponse,
    DashboardQueueItem,
    DashboardQueueResponse,
    DashboardSummaryResponse,
    ErrorResponse,
    InvestigationStepResponse,
    NetworkRiskSignals,
    NetworkSummaryResponse,
    NetworkTransactionResponse,
    PrioritizedEvidenceResponse,
    TransactionScoreRequest,
    TransactionScoreResponse,
    RiskClusterResponse,
    RiskClusterSignal,
    RiskClusterTimelineItem,
)

from razorguard.config import TRANSACTIONS_PATH
from razorguard.investigation.store import CaseStore
from razorguard.runner.batch import (
    load_model,
    score_runtime_transaction,
)
from razorguard.runtime.context import RuntimeContextStore

from razorguard.graph.builder import (
    build_entity_graph,
    graph_summary,
)

from razorguard.graph.investigator import (
    investigate_transaction,
)

from razorguard.graph.clusters import (
    build_risk_cluster,
)

from razorguard.graph.evidence import (
    build_coordinated_evidence,
)

from razorguard.graph.prioritization import (
    prioritize_evidence,
    group_by_tier,
    prioritized_to_dict,
)

from razorguard.investigation.path import (
    build_investigation_path,
    step_to_dict,
)

from razorguard.copilot.service import (
    answer_question,
    get_copilot_status,
)

router = APIRouter(
    prefix="/v1",
    tags=["risk"],
)


DEFAULT_CASES_PATH = Path(
    "artifacts/investigator_cases.parquet"
)


# ============================================================
# RUNTIME HELPERS
# ============================================================


def _load_runtime_model():
    """Load the currently configured RazorGuard model."""

    try:
        return load_model()

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc


def _request_to_frame(
    request: TransactionScoreRequest,
) -> pd.DataFrame:
    """Convert a validated API request into a transaction frame."""

    return pd.DataFrame(
        [request.model_dump()]
    )


def _runtime_context_store() -> RuntimeContextStore:
    """Return the configured runtime transaction history provider."""

    return RuntimeContextStore(
        TRANSACTIONS_PATH
    )


def _case_store() -> CaseStore:
    """Return the configured investigator case store."""

    return CaseStore(
        DEFAULT_CASES_PATH
    )

def _get_existing_case_id(
    store: CaseStore,
    transaction_id: str,
) -> str | None:
    """Return an existing case ID for a transaction, if present."""

    case_id = f"CASE-{transaction_id}"

    case = store.get(
        case_id
    )

    if case is None:
        return None

    return str(
        case["case_id"]
    )


# ============================================================
# CASE RESPONSE
# ============================================================


def _case_response(
    case: dict[str, Any],
) -> CaseResponse:
    """Convert a CaseStore record into an API response."""

    raw_assigned = case.get("assigned_to")
    if raw_assigned is None or (
        isinstance(raw_assigned, float)
        and pd.isna(raw_assigned)
    ):
        normalized_assigned: str | None = None
    elif str(raw_assigned).lower() in {
        "nan", "none", "", "null",
    }:
        normalized_assigned = None
    else:
        normalized_assigned = str(raw_assigned)

    return CaseResponse(
        case_id=str(
            case["case_id"]
        ),
        transaction_id=str(
            case["transaction_id"]
        ),
        status=str(
            case["status"]
        ),
        priority=str(
            case["priority"]
        ),
        assigned_to=normalized_assigned,
        created_at=str(
            case["created_at"]
        ),
        updated_at=str(
            case["updated_at"]
        ),
        risk_score=float(
            case["risk_score"]
        ),
        risk_level=str(
            case["risk_level"]
        ),
        decision=str(
            case["decision"]
        ),
        primary_reason=str(
            case["primary_reason"]
        ),
        evidence_text=str(
            case["evidence_text"]
        ),
        model_probability=float(
            case["model_probability"]
        ),
        network_score=float(
            case["network_score"]
        ),
        investigation_narrative=str(
            case["investigation_narrative"]
        ),
    )


# ============================================================
# TRANSACTION SCORING
# ============================================================


@router.post(
    "/transactions/score",
    response_model=TransactionScoreResponse,
    responses={
        503: {
            "model": ErrorResponse,
        },
        422: {
            "description": "Invalid transaction payload",
        },
    },
)
def score_transaction_api(
    request: TransactionScoreRequest,
) -> TransactionScoreResponse:
    """
    Score one live transaction using runtime historical context.

    Historical transactions strictly before the incoming
    transaction are supplied to the point-in-time feature engine.

    The incoming transaction itself is scored as the final event.

    Non-ALLOW decisions automatically create an investigator case.
    """

    model, metadata = _load_runtime_model()

    raw = _request_to_frame(
        request
    )

    transaction = (
        raw.iloc[0]
        .to_dict()
    )

    # --------------------------------------------------------
    # Runtime scoring
    # --------------------------------------------------------

    try:
        context_store = (
            _runtime_context_store()
        )

        result_data = (
            score_runtime_transaction(
                transaction=transaction,
                model=model,
                context_store=context_store,
            )
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "transaction scoring failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc

    if not result_data:
        raise HTTPException(
            status_code=500,
            detail=(
                "transaction scoring "
                "produced no result"
            ),
        )

    # --------------------------------------------------------
    # Evidence
    # --------------------------------------------------------

    evidence_text = str(
        result_data.get(
            "evidence_text",
            "",
        )
    )

    evidence = [
        item.strip()
        for item in evidence_text.split("|")
        if item.strip()
    ]

        # --------------------------------------------------------
    # Automatic investigator case creation
    # --------------------------------------------------------

    case_id: str | None = None

    if result_data["decision"] != "ALLOW":
        try:
            store = _case_store()

            transaction_id = str(
                result_data["transaction_id"]
            )

            existing_case_id = (
                _get_existing_case_id(
                    store,
                    transaction_id,
                )
            )

            if existing_case_id is not None:
                # Idempotent retry:
                # the investigation case already exists.
                case_id = existing_case_id

            else:
                created_case = store.create(
                    result_data["case"],
                    actor="api",
                    investigation_narrative=(
                        result_data["case"].get(
                            "investigation_narrative",
                            "",
                        )
                    ),
                )

                case_id = str(
                    created_case["case_id"]
                )

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "case creation failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            ) from exc

    # --------------------------------------------------------
    # API response
    # --------------------------------------------------------

    return TransactionScoreResponse(
        transaction_id=str(
            result_data[
                "transaction_id"
            ]
        ),
        risk_score=float(
            result_data[
                "risk_score"
            ]
        ),
        risk_level=str(
            result_data[
                "risk_level"
            ]
        ),
        decision=str(
            result_data[
                "decision"
            ]
        ),
        primary_reason=str(
            result_data[
                "primary_reason"
            ]
        ),
        evidence=evidence,
        model_probability=float(
            result_data[
                "model_probability"
            ]
        ),
        network_score=float(
            result_data[
                "network_score"
            ]
        ),
        behavioral_signal=float(
            result_data[
                "behavioral_signal"
            ]
        ),
        model=str(
            metadata.get(
                "model",
                "unknown",
            )
        ),
        model_threshold=float(
            metadata.get(
                "threshold",
                0.0,
            )
        ),
        case_id=case_id,
    )

# ============================================================
# NETWORK INTELLIGENCE
# ============================================================


@router.get(
    "/network/summary",
    response_model=NetworkSummaryResponse,
)
def network_summary() -> NetworkSummaryResponse:
    """
    Return graph-level network intelligence statistics.
    """

    try:
        transactions = pd.read_parquet(
            TRANSACTIONS_PATH
        )

        graph = build_entity_graph(
            transactions
        )

        summary = graph_summary(
            graph
        )

        return NetworkSummaryResponse(
            **summary
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "transaction dataset unavailable: "
                f"{exc}"
            ),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "network summary failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc

@router.get(
    "/network/transaction/{transaction_id}/cluster",
    response_model=RiskClusterResponse,
)
def network_transaction_cluster(
    transaction_id: str,
) -> RiskClusterResponse:
    """
    Return coordinated-risk cluster intelligence for one
    transaction.
    """

    try:
        transactions = pd.read_parquet(
            TRANSACTIONS_PATH
        )

        transaction_matches = transactions[
            transactions["transaction_id"].astype(str)
            == str(transaction_id)
        ]

        if transaction_matches.empty:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"transaction not found: "
                    f"{transaction_id}"
                ),
            )

        transaction = transaction_matches.iloc[0]

        cluster = build_risk_cluster(
            transactions,
            account_id=str(
                transaction["account_id"]
            ),
            device_id=str(
                transaction["device_id"]
            ),
            merchant_id=str(
                transaction["merchant_id"]
            ),
            cluster_id=(
                f"FR-{str(transaction_id)}"
            ),
        )

        return RiskClusterResponse(
            cluster_id=cluster.cluster_id,
            cluster_type=cluster.cluster_type,
            risk_score=cluster.risk_score,
            accounts=cluster.accounts,
            devices=cluster.devices,
            merchants=cluster.merchants,
            transactions=cluster.transactions,
            signals=[
                RiskClusterSignal(
                    **signal
                )
                for signal in cluster.signals
            ],
            evidence=cluster.evidence,
            timeline=[
                RiskClusterTimelineItem(
                    **item
                )
                for item in cluster.timeline
            ],
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "transaction dataset unavailable: "
                f"{exc}"
            ),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "network cluster investigation failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc
    
@router.get(
    "/network/transaction/{transaction_id}",
    response_model=NetworkTransactionResponse,
)
def network_transaction(
    transaction_id: str,
) -> NetworkTransactionResponse:
    """
    Return explainable network context for one transaction.
    """

    try:
        transactions = pd.read_parquet(
            TRANSACTIONS_PATH
        )

        result = investigate_transaction(
            transactions=transactions,
            transaction_id=transaction_id,
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "transaction dataset unavailable: "
                f"{exc}"
            ),
        ) from exc

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=(
                f"transaction not found: "
                f"{transaction_id}"
            ),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "network investigation failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc

    return NetworkTransactionResponse(
        transaction_id=str(
            result["transaction_id"]
        ),
        timestamp=str(
            result["timestamp"]
        ),
        account_id=str(
            result["account_id"]
        ),
        device_id=str(
            result["device_id"]
        ),
        merchant_id=str(
            result["merchant_id"]
        ),
        account_history_count=int(
            result["account_history_count"]
        ),
        accounts_seen_on_device=[
            str(item)
            for item in result[
                "accounts_seen_on_device"
            ]
        ],
        accounts_seen_at_merchant=[
            str(item)
            for item in result[
                "accounts_seen_at_merchant"
            ]
        ],
        related_transaction_count=int(
            result[
                "related_transaction_count"
            ]
        ),
        network_risk_signals=NetworkRiskSignals(
            **result[
                "network_risk_signals"
            ]
        ),
    )

# ============================================================
# CASE MANAGEMENT
# ============================================================


@router.get(
    "/cases",
    response_model=CaseListResponse,
)
def list_cases(
    status: str | None = Query(
        default=None
    ),
    assigned_to: str | None = Query(
        default=None
    ),
    priority: str | None = Query(
        default=None
    ),
    search: str | None = Query(
        default=None,
        min_length=1,
        max_length=200,
    ),
    sort_by: str = Query(
        default="risk_score"
    ),
    sort_order: str = Query(
        default="desc"
    ),
    page: int = Query(
        default=1,
        ge=1,
    ),
    page_size: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
) -> CaseListResponse:
    """
    List investigator cases with server-side filtering,
    searching, sorting, and pagination.

    Supported filters:
        status
        assigned_to
        priority
        search

    Supported sorting:
        risk_score
        model_probability
        network_score
        created_at
        updated_at
        transaction_id
        case_id
        priority
        status
    """

    store = _case_store()

    try:
        frame = store.list(
            status=status,
            assigned_to=assigned_to,
            priority=priority,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    if search is not None:
        query = search.strip().lower()

        if query:
            searchable = (
                frame["case_id"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains(
                    query,
                    regex=False,
                )
                |
                frame["transaction_id"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains(
                    query,
                    regex=False,
                )
                |
                frame["primary_reason"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains(
                    query,
                    regex=False,
                )
            )

            frame = frame[searchable]

    # --------------------------------------------------------
    # Sorting
    # --------------------------------------------------------

    allowed_sort_fields = {
        "risk_score",
        "model_probability",
        "network_score",
        "created_at",
        "updated_at",
        "transaction_id",
        "case_id",
        "priority",
        "status",
    }

    sort_by = sort_by.lower()

    if sort_by not in allowed_sort_fields:
        raise HTTPException(
            status_code=422,
            detail=(
                f"invalid sort_by: {sort_by}. "
                f"Allowed values: "
                f"{', '.join(sorted(allowed_sort_fields))}"
            ),
        )

    sort_order = sort_order.lower()

    if sort_order not in {
        "asc",
        "desc",
    }:
        raise HTTPException(
            status_code=422,
            detail=(
                "sort_order must be either "
                "'asc' or 'desc'"
            ),
        )

    ascending = sort_order == "asc"

    frame = frame.sort_values(
        by=sort_by,
        ascending=ascending,
        kind="stable",
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Pagination
    # --------------------------------------------------------

    total = len(frame)

    total_pages = max(
        1,
        (total + page_size - 1)
        // page_size,
    )

    if total > 0 and page > total_pages:
        raise HTTPException(
            status_code=404,
            detail=(
                f"page {page} is out of range; "
                f"total_pages={total_pages}"
            ),
        )

    start = (
        (page - 1)
        * page_size
    )

    end = start + page_size

    page_frame = frame.iloc[
        start:end
    ]

    cases = [
        _case_response(
            row.to_dict()
        )
        for _, row in page_frame.iterrows()
    ]

    return CaseListResponse(
        cases=cases,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get(
    "/cases/{case_id}",
    response_model=CaseResponse,
)
def get_case(
    case_id: str,
) -> CaseResponse:
    """
    Retrieve one investigator case.
    """

    store = _case_store()

    case = store.get(
        case_id
    )

    if case is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"case not found: "
                f"{case_id}"
            ),
        )

    return _case_response(
        case
    )


@router.post(
    "/cases/{case_id}/assign",
    response_model=CaseResponse,
)
def assign_case_api(
    case_id: str,
    request: CaseAssignRequest,
) -> CaseResponse:
    """
    Assign an investigator case.
    """

    store = _case_store()

    try:
        case = store.assign(
            case_id,
            request.investigator,
            actor=request.actor,
        )

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return _case_response(
        case
    )


@router.post(
    "/cases/{case_id}/transition",
    response_model=CaseResponse,
)
def transition_case_api(
    case_id: str,
    request: CaseTransitionRequest,
) -> CaseResponse:
    """
    Transition a case through the investigation lifecycle.
    """

    store = _case_store()

    try:
        case = store.update_status(
            case_id,
            request.status,
            actor=request.actor,
            details=request.details,
        )

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return _case_response(
        case
    )


# ============================================================
# AUDIT
# ============================================================


@router.get(
    "/cases/{case_id}/audit",
    response_model=AuditResponse,
)
def get_case_audit_api(
    case_id: str,
) -> AuditResponse:
    """
    Retrieve the append-only audit history for a case.
    """

    store = _case_store()

    if store.get(case_id) is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"case not found: "
                f"{case_id}"
            ),
        )

    frame = store.audit(
        case_id
    )

    events = [
        AuditEventResponse(
            case_id=str(
                row["case_id"]
            ),
            timestamp=str(
                row["timestamp"]
            ),
            action=str(
                row["action"]
            ),
            actor=str(
                row["actor"]
            ),
            from_status=(
                None
                if pd.isna(
                    row["from_status"]
                )
                else str(
                    row["from_status"]
                )
            ),
            to_status=(
                None
                if pd.isna(
                    row["to_status"]
                )
                else str(
                    row["to_status"]
                )
            ),
            details=str(
                row["details"]
            ),
        )
        for _, row in frame.iterrows()
    ]

    return AuditResponse(
        case_id=case_id,
        events=events,
        total=len(events),
    )

# ============================================================
# DASHBOARD SUMMARY
# ============================================================

@router.get(
    "/dashboard/summary",
    response_model=DashboardSummaryResponse,
)
def dashboard_summary() -> DashboardSummaryResponse:
    """
    Return aggregate investigator dashboard metrics.
    """

    store = _case_store()

    frame = store.list()

    if frame.empty:
        return DashboardSummaryResponse(
            open_cases=0,
            critical_cases=0,
            high_cases=0,
            medium_cases=0,
            low_cases=0,
            average_risk_score=0.0,
            total_cases=0,
        )

    open_cases = int(
        (~frame["status"].isin(
            ["RESOLVED", "DISMISSED"]
        )).sum()
    )

    critical_cases = int(
        (frame["priority"] == "CRITICAL").sum()
    )

    high_cases = int(
        (frame["priority"] == "HIGH").sum()
    )

    medium_cases = int(
        (frame["priority"] == "MEDIUM").sum()
    )

    low_cases = int(
        (frame["priority"] == "LOW").sum()
    )

    average_risk_score = round(
        float(frame["risk_score"].mean()),
        4,
    )

    return DashboardSummaryResponse(
        open_cases=open_cases,
        critical_cases=critical_cases,
        high_cases=high_cases,
        medium_cases=medium_cases,
        low_cases=low_cases,
        average_risk_score=average_risk_score,
        total_cases=int(len(frame)),
    )

# ============================================================
# DASHBOARD DISTRIBUTION
# ============================================================


@router.get(
    "/dashboard/distribution",
    response_model=DashboardDistributionResponse,
)
def dashboard_distribution() -> DashboardDistributionResponse:
    """
    Return risk-priority distribution for dashboard visualization.
    """

    store = _case_store()

    frame = store.list()

    if frame.empty:
        return DashboardDistributionResponse(
            items=[],
            total=0,
        )

    total = len(frame)

    counts = (
        frame["priority"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
        .value_counts()
    )

    ordered_labels = [
        "CRITICAL",
        "HIGH",
        "MEDIUM",
        "LOW",
    ]

    items = []

    for label in ordered_labels:
        count = int(
            counts.get(
                label,
                0,
            )
        )

        items.append(
            DashboardDistributionItem(
                label=label,
                count=count,
                percentage=round(
                    (count / total) * 100,
                    2,
                )
                if total
                else 0.0,
            )
        )

    return DashboardDistributionResponse(
        items=items,
        total=total,
    )


# ============================================================
# DASHBOARD RECENT ACTIVITY
# ============================================================


@router.get(
    "/dashboard/activity",
    response_model=DashboardActivityResponse,
)
def dashboard_activity(
    limit: int = Query(
        default=10,
        ge=1,
        le=50,
    ),
) -> DashboardActivityResponse:
    """
    Return the most recent investigator activity.
    """

    store = _case_store()

    audit = store._load_audit()

    if audit.empty:
        return DashboardActivityResponse(
            items=[],
            total=0,
        )

    audit = audit.sort_values(
        "timestamp",
        ascending=False,
        kind="stable",
    ).head(limit)

    items = [
        DashboardActivityItem(
            case_id=str(row["case_id"]),
            transaction_id=(
                str(row["case_id"])
                .replace("CASE-", "", 1)
            ),
            action=str(row["action"]),
            actor=str(row["actor"]),
            timestamp=str(row["timestamp"]),
            details=str(row["details"]),
        )
        for _, row in audit.iterrows()
    ]

    return DashboardActivityResponse(
        items=items,
        total=len(items),
    )


# ============================================================
# DASHBOARD PRIORITY QUEUE
# ============================================================


@router.get(
    "/dashboard/queue",
    response_model=DashboardQueueResponse,
)
def dashboard_queue(
    limit: int = Query(
        default=10,
        ge=1,
        le=50,
    ),
) -> DashboardQueueResponse:
    """
    Return the highest-priority actionable cases.
    """

    store = _case_store()

    frame = store.list()

    if frame.empty:
        return DashboardQueueResponse(
            items=[],
            total=0,
        )

    actionable = frame[
        ~frame["status"].isin(
            [
                "RESOLVED",
                "DISMISSED",
            ]
        )
    ].copy()

    if actionable.empty:
        return DashboardQueueResponse(
            items=[],
            total=0,
        )

    decision_priority = {
        "BLOCK": 0,
        "REVIEW": 1,
        "ALLOW": 2,
    }

    actionable["_decision_priority"] = (
        actionable["decision"]
        .map(decision_priority)
        .fillna(99)
    )

    actionable = (
        actionable
        .sort_values(
            [
                "_decision_priority",
                "risk_score",
                "model_probability",
                "network_score",
            ],
            ascending=[
                True,
                False,
                False,
                False,
            ],
            kind="stable",
        )
        .head(limit)
    )

    items = [
        DashboardQueueItem(
            case_id=str(row["case_id"]),
            transaction_id=str(row["transaction_id"]),
            priority=str(row["priority"]),
            risk_score=float(row["risk_score"]),
            risk_level=str(row["risk_level"]),
            decision=str(row["decision"]),
            primary_reason=str(row["primary_reason"]),
        )
        for _, row in actionable.iterrows()
    ]

    return DashboardQueueResponse(
        items=items,
        total=len(items),
    )

# ============================================================
# RISK ANALYTICS
# ============================================================

@router.get(
    "/analytics/overview",
    response_model=AnalyticsMetricResponse,
)
def analytics_overview() -> AnalyticsMetricResponse:
    """
    Return aggregate risk analytics for investigator cases.

    Analytics are calculated server-side from the persistent
    investigator case store so the frontend does not need to
    download the complete case dataset.
    """

    store = _case_store()
    frame = store.list()

    if frame.empty:
        empty = []

        return AnalyticsMetricResponse(
            total_cases=0,
            average_risk_score=0.0,
            median_risk_score=0.0,
            maximum_risk_score=0.0,
            average_model_probability=0.0,
            average_network_score=0.0,
            priority_distribution=empty,
            risk_level_distribution=empty,
            decision_distribution=empty,
            status_distribution=empty,
            top_reasons=empty,
        )

    total = len(frame)

    def distribution(column: str) -> list[dict]:
        counts = (
            frame[column]
            .fillna("UNKNOWN")
            .astype(str)
            .value_counts()
        )

        return [
            {
                "label": str(label),
                "count": int(count),
                "percentage": round(
                    (float(count) / total) * 100,
                    2,
                ),
            }
            for label, count in counts.items()
        ]

    reasons = (
        frame["primary_reason"]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .head(8)
    )

    top_reasons = [
        {
            "label": str(label),
            "count": int(count),
            "percentage": round(
                (float(count) / total) * 100,
                2,
            ),
        }
        for label, count in reasons.items()
    ]

    return AnalyticsMetricResponse(
        total_cases=int(total),

        average_risk_score=round(
            float(frame["risk_score"].mean()),
            2,
        ),

        median_risk_score=round(
            float(frame["risk_score"].median()),
            2,
        ),

        maximum_risk_score=round(
            float(frame["risk_score"].max()),
            2,
        ),

        average_model_probability=round(
            float(frame["model_probability"].mean()),
            4,
        ),

        average_network_score=round(
            float(frame["network_score"].mean()),
            2,
        ),

        priority_distribution=[
            AnalyticsDistributionItem(**item)
            for item in distribution("priority")
        ],

        risk_level_distribution=[
            AnalyticsDistributionItem(**item)
            for item in distribution("risk_level")
        ],

        decision_distribution=[
            AnalyticsDistributionItem(**item)
            for item in distribution("decision")
        ],

        status_distribution=[
            AnalyticsDistributionItem(**item)
            for item in distribution("status")
        ],

        top_reasons=[
            AnalyticsDistributionItem(**item)
            for item in top_reasons
        ],
    )


# ============================================================
# CASE INTELLIGENCE
# ============================================================


@router.get(
    "/cases/{case_id}/intelligence",
    response_model=CaseIntelligenceResponse,
)
def case_intelligence(
    case_id: str,
) -> CaseIntelligenceResponse:
    """
    Return coordinated-risk intelligence for one case.

    Combines case data, network transaction intelligence,
    cluster intelligence, evidence synthesis, and
    investigation path into a single response.
    """

    store = _case_store()

    case = store.get(case_id)

    if case is None:
        raise HTTPException(
            status_code=404,
            detail=f"case not found: {case_id}",
        )

    transaction_id = str(case["transaction_id"])

    # --------------------------------------------------------
    # Fetch network transaction intelligence
    # --------------------------------------------------------

    network_data: dict[str, Any] | None = None

    try:
        transactions = pd.read_parquet(
            TRANSACTIONS_PATH
        )

        network_result = investigate_transaction(
            transactions=transactions,
            transaction_id=transaction_id,
        )

        network_data = network_result

    except Exception:
        network_data = None

    # --------------------------------------------------------
    # Fetch cluster intelligence
    # --------------------------------------------------------

    cluster_data: dict[str, Any] | None = None

    has_temporal_burst = False

    try:
        if network_data is not None:
            cluster = build_risk_cluster(
                transactions,
                account_id=str(
                    network_data["account_id"]
                ),
                device_id=str(
                    network_data["device_id"]
                ),
                merchant_id=str(
                    network_data["merchant_id"]
                ),
                cluster_id=f"FR-{transaction_id}",
            )

            cluster_data = {
                "cluster_id": cluster.cluster_id,
                "cluster_type": cluster.cluster_type,
                "risk_score": cluster.risk_score,
                "accounts": cluster.accounts,
                "devices": cluster.devices,
                "merchants": cluster.merchants,
                "transactions": cluster.transactions,
                "signals": cluster.signals,
                "evidence": cluster.evidence,
                "timeline": cluster.timeline,
            }

            has_temporal_burst = any(
                s.get("type") == "TEMPORAL_BURST"
                for s in cluster.signals
            )

    except Exception:
        cluster_data = None

    # --------------------------------------------------------
    # Network signals
    # --------------------------------------------------------

    network_signals: dict[str, Any] = {}
    accounts_on_device: list[str] = []
    accounts_at_merchant: list[str] = []

    if network_data is not None:
        network_signals = network_data.get(
            "network_risk_signals", {}
        )
        accounts_on_device = network_data.get(
            "accounts_seen_on_device", []
        )
        accounts_at_merchant = network_data.get(
            "accounts_seen_at_merchant", []
        )

    # --------------------------------------------------------
    # Audit history
    # --------------------------------------------------------

    audit_frame = store.audit(case_id)
    has_audit_events = not audit_frame.empty

    # --------------------------------------------------------
    # Evidence synthesis
    # --------------------------------------------------------

    evidence_items = build_coordinated_evidence(
        network_risk_signals=network_signals or None,
        accounts_seen_on_device=accounts_on_device or None,
        accounts_seen_at_merchant=accounts_at_merchant or None,
        account_history_count=int(
            network_data.get("account_history_count", 0)
            if network_data else 0
        ),
        related_transaction_count=int(
            network_data.get("related_transaction_count", 0)
            if network_data else 0
        ),
        cluster_signals=(
            [dict(s) for s in cluster_data["signals"]]
            if cluster_data
            else None
        ),
        cluster_evidence=(
            cluster_data["evidence"]
            if cluster_data
            else None
        ),
        cluster_type=(
            cluster_data["cluster_type"]
            if cluster_data
            else None
        ),
        cluster_risk_score=(
            cluster_data["risk_score"]
            if cluster_data
            else None
        ),
        cluster_accounts=(
            cluster_data["accounts"]
            if cluster_data
            else None
        ),
        cluster_devices=(
            cluster_data["devices"]
            if cluster_data
            else None
        ),
        cluster_merchants=(
            cluster_data["merchants"]
            if cluster_data
            else None
        ),
        cluster_transactions=(
            cluster_data["transactions"]
            if cluster_data
            else None
        ),
        risk_score=float(case["risk_score"]),
        model_probability=float(
            case["model_probability"]
        ),
        network_score=float(case["network_score"]),
    )

    # --------------------------------------------------------
    # Evidence prioritization
    # --------------------------------------------------------

    prioritized = prioritize_evidence(
        evidence_items
    )

    grouped = group_by_tier(prioritized)

    evidence_summary = {
        "PRIMARY": len(grouped["PRIMARY"]),
        "SUPPORTING": len(grouped["SUPPORTING"]),
        "CONTEXTUAL": len(grouped["CONTEXTUAL"]),
        "TOTAL": len(prioritized),
    }

    # --------------------------------------------------------
    # Investigation path
    # --------------------------------------------------------

    steps = build_investigation_path(
        status=str(case["status"]),
        risk_score=float(case["risk_score"]),
        decision=str(case["decision"]),
        model_probability=float(
            case["model_probability"]
        ),
        network_score=float(
            case["network_score"]
        ),
        assigned_to=(
            None
            if case.get("assigned_to") is None
            else str(case["assigned_to"])
        ),
        has_audit_events=has_audit_events,
        device_shared=bool(
            network_signals.get("device_shared", False)
        ),
        merchant_shared=bool(
            network_signals.get("merchant_shared", False)
        ),
        new_device=bool(
            network_signals.get(
                "new_device_for_account", False
            )
        ),
        accounts_on_device=len(accounts_on_device),
        accounts_at_merchant=len(accounts_at_merchant),
        cluster_type=(
            cluster_data["cluster_type"]
            if cluster_data
            else None
        ),
        cluster_risk_score=(
            cluster_data["risk_score"]
            if cluster_data
            else None
        ),
        cluster_accounts=(
            cluster_data["accounts"]
            if cluster_data
            else None
        ),
        cluster_devices=(
            cluster_data["devices"]
            if cluster_data
            else None
        ),
        cluster_transactions=(
            cluster_data["transactions"]
            if cluster_data
            else None
        ),
        has_temporal_burst=has_temporal_burst,
    )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return CaseIntelligenceResponse(
        case_id=str(case["case_id"]),
        transaction_id=transaction_id,
        risk_score=float(case["risk_score"]),
        risk_level=str(case["risk_level"]),
        decision=str(case["decision"]),
        primary_reason=str(
            case["primary_reason"]
        ),
        evidence=[
            PrioritizedEvidenceResponse(
                **prioritized_to_dict(p)
            )
            for p in prioritized
        ],
        investigation_steps=[
            InvestigationStepResponse(
                **step_to_dict(s)
            )
            for s in steps
        ],
        evidence_summary=evidence_summary,
    )


# ============================================================
# INVESTIGATION COPILOT
# ============================================================


@router.get(
    "/copilot/status",
    response_model=CopilotStatusResponse,
)
def copilot_status() -> CopilotStatusResponse:
    """
    Return the current copilot provider status.
    """

    status = get_copilot_status()

    return CopilotStatusResponse(**status)


@router.post(
    "/copilot/ask",
    response_model=CopilotResponse,
)
def copilot_ask(
    case_id: str,
    request: CopilotRequest,
) -> CopilotResponse:
    """
    Answer an investigator question using grounded evidence.

    The copilot uses ONLY verified RazorGuard data.
    It never modifies case state or risk decisions.
    """

    store = _case_store()

    case = store.get(case_id)

    if case is None:
        raise HTTPException(
            status_code=404,
            detail=f"case not found: {case_id}",
        )

    transaction_id = str(case["transaction_id"])

    # --------------------------------------------------------
    # Gather intelligence
    # --------------------------------------------------------

    network_data: dict[str, Any] | None = None

    try:
        transactions = pd.read_parquet(
            TRANSACTIONS_PATH
        )

        network_result = investigate_transaction(
            transactions=transactions,
            transaction_id=transaction_id,
        )

        network_data = network_result
    except Exception:
        network_data = None

    cluster_data: dict[str, Any] | None = None
    has_temporal_burst = False

    try:
        if network_data is not None:
            cluster = build_risk_cluster(
                transactions,
                account_id=str(network_data["account_id"]),
                device_id=str(network_data["device_id"]),
                merchant_id=str(network_data["merchant_id"]),
                cluster_id=f"FR-{transaction_id}",
            )

            cluster_data = {
                "cluster_id": cluster.cluster_id,
                "cluster_type": cluster.cluster_type,
                "risk_score": cluster.risk_score,
                "accounts": cluster.accounts,
                "devices": cluster.devices,
                "merchants": cluster.merchants,
                "transactions": cluster.transactions,
                "signals": cluster.signals,
                "evidence": cluster.evidence,
                "timeline": cluster.timeline,
            }

            has_temporal_burst = any(
                s.get("type") == "TEMPORAL_BURST"
                for s in cluster.signals
            )
    except Exception:
        cluster_data = None

    # --------------------------------------------------------
    # Network signals
    # --------------------------------------------------------

    network_signals: dict[str, Any] = {}
    accounts_on_device: list[str] = []
    accounts_at_merchant: list[str] = []

    if network_data is not None:
        network_signals = network_data.get("network_risk_signals", {})
        accounts_on_device = network_data.get("accounts_seen_on_device", [])
        accounts_at_merchant = network_data.get("accounts_seen_at_merchant", [])

    # --------------------------------------------------------
    # Evidence synthesis
    # --------------------------------------------------------

    evidence_items = build_coordinated_evidence(
        network_risk_signals=network_signals or None,
        accounts_seen_on_device=accounts_on_device or None,
        accounts_seen_at_merchant=accounts_at_merchant or None,
        account_history_count=int(
            network_data.get("account_history_count", 0)
            if network_data else 0
        ),
        related_transaction_count=int(
            network_data.get("related_transaction_count", 0)
            if network_data else 0
        ),
        cluster_signals=(
            [dict(s) for s in cluster_data["signals"]]
            if cluster_data else None
        ),
        cluster_evidence=(
            cluster_data["evidence"]
            if cluster_data else None
        ),
        cluster_type=(
            cluster_data["cluster_type"]
            if cluster_data else None
        ),
        cluster_risk_score=(
            cluster_data["risk_score"]
            if cluster_data else None
        ),
        cluster_accounts=(
            cluster_data["accounts"]
            if cluster_data else None
        ),
        cluster_devices=(
            cluster_data["devices"]
            if cluster_data else None
        ),
        cluster_merchants=(
            cluster_data["merchants"]
            if cluster_data else None
        ),
        cluster_transactions=(
            cluster_data["transactions"]
            if cluster_data else None
        ),
        risk_score=float(case["risk_score"]),
        model_probability=float(case["model_probability"]),
        network_score=float(case["network_score"]),
    )

    prioritized = prioritize_evidence(evidence_items)
    grouped = group_by_tier(prioritized)

    evidence_summary = {
        "PRIMARY": len(grouped["PRIMARY"]),
        "SUPPORTING": len(grouped["SUPPORTING"]),
        "CONTEXTUAL": len(grouped["CONTEXTUAL"]),
        "TOTAL": len(prioritized),
    }

    # --------------------------------------------------------
    # Investigation path
    # --------------------------------------------------------

    steps = build_investigation_path(
        status=str(case["status"]),
        risk_score=float(case["risk_score"]),
        decision=str(case["decision"]),
        model_probability=float(case["model_probability"]),
        network_score=float(case["network_score"]),
        assigned_to=(
            None if case.get("assigned_to") is None
            else str(case["assigned_to"])
        ),
        has_audit_events=not store.audit(case_id).empty,
        device_shared=bool(network_signals.get("device_shared", False)),
        merchant_shared=bool(network_signals.get("merchant_shared", False)),
        new_device=bool(network_signals.get("new_device_for_account", False)),
        accounts_on_device=len(accounts_on_device),
        accounts_at_merchant=len(accounts_at_merchant),
        cluster_type=(
            cluster_data["cluster_type"]
            if cluster_data else None
        ),
        cluster_risk_score=(
            cluster_data["risk_score"]
            if cluster_data else None
        ),
        cluster_accounts=(
            cluster_data["accounts"]
            if cluster_data else None
        ),
        cluster_devices=(
            cluster_data["devices"]
            if cluster_data else None
        ),
        cluster_transactions=(
            cluster_data["transactions"]
            if cluster_data else None
        ),
        has_temporal_burst=has_temporal_burst,
    )

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    audit_frame = store.audit(case_id)
    audit_events = []

    for _, row in audit_frame.iterrows():
        audit_events.append({
            "action": str(row["action"]),
            "actor": str(row["actor"]),
            "timestamp": str(row["timestamp"]),
            "details": str(row["details"]),
        })

    # --------------------------------------------------------
    # Answer
    # --------------------------------------------------------

    response = answer_question(
        question=request.question,
        case_context=dict(case),
        evidence=[prioritized_to_dict(p) for p in prioritized],
        network_data=network_data,
        cluster_data=cluster_data,
        investigation_steps=[step_to_dict(s) for s in steps],
        audit_events=audit_events,
        evidence_summary=evidence_summary,
    )

    return CopilotResponse(
        answer=response.answer,
        key_evidence=response.key_evidence,
        interpretation=response.interpretation,
        recommended_focus=response.recommended_focus,
        grounding=response.grounding,
    )