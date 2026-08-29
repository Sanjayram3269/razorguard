from razorguard.cases.enrichment import (
    build_account_context,
    build_behavior_context,
    build_investigation_narrative,
    build_network_context,
    enrich_case,
)
from razorguard.risk.case import RiskCase


def make_case():
    return RiskCase(
        transaction_id="T001",
        risk_score=91.5,
        risk_level="CRITICAL",
        decision="BLOCK",
        primary_reason="high combined transaction and network risk",
        evidence=[
            "transaction amount is elevated",
            "elevated network/entity risk",
        ],
        model_probability=0.95,
        network_score=18.0,
    )


def make_transaction():
    return {
        "transaction_id": "T001",
        "account_id": "A001",
        "merchant_id": "M001",
        "device_id": "D001",
        "account_age_days": 8,
        "prior_tx_count": 12,
        "account_id_prior_count_60m": 6,
        "account_id_prior_count_1440m": 9,
        "prior_unique_merchants": 14,
        "is_dormant_return": 1,
        "amount_zscore": 3.2,
        "account_velocity_ratio": 2.4,
        "location_mismatch": 1,
        "prior_accounts_per_device": 5,
        "prior_accounts_per_merchant": 44,
        "prior_merchants_per_device": 7,
        "account_device_novelty": 1,
        "account_merchant_novelty": 1,
    }


def test_account_context_is_point_in_time():
    context = build_account_context(
        make_transaction()
    )

    assert context["account_id"] == "A001"
    assert context["prior_count_60m"] == 6
    assert context["prior_transaction_count"] == 12
    assert context["dormant_return"] is True


def test_network_context_preserves_relationship_evidence():
    context = build_network_context(
        make_transaction(),
        18.0,
    )

    assert context["device_id"] == "D001"
    assert context["prior_accounts_per_device"] == 5
    assert context["prior_accounts_per_merchant"] == 44
    assert context["network_score"] == 18.0


def test_behavior_context_is_bounded():
    context = build_behavior_context(
        make_transaction(),
        1.7,
    )

    assert context["behavioral_signal"] == 1.0
    assert context["location_mismatch"] is True
    assert context["amount_zscore"] == 3.2


def test_narrative_contains_actionable_context():
    case = make_case()

    narrative = build_investigation_narrative(
        case,
        build_account_context(make_transaction()),
        build_network_context(
            make_transaction(),
            case.network_score,
        ),
        build_behavior_context(
            make_transaction(),
            0.92,
        ),
    )

    assert "CRITICAL" in narrative
    assert "BLOCK" in narrative
    assert "velocity" in narrative.lower()
    assert "device" in narrative.lower()
    assert "network risk" in narrative.lower()


def test_enrich_case_produces_complete_artifact():
    case = make_case()

    enriched = enrich_case(
        case,
        make_transaction(),
        behavioral_signal=0.92,
    )

    assert enriched["transaction_id"] == "T001"
    assert enriched["risk_score"] == 91.5
    assert enriched["account_context"]["account_id"] == "A001"
    assert enriched["network_context"]["device_id"] == "D001"
    assert enriched["behavior_context"]["behavioral_signal"] == 0.92
    assert enriched["investigation_narrative"]