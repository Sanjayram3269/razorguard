from razorguard.risk.case import RiskCase
from razorguard.risk.fusion import fuse_risk, risk_level
from razorguard.risk.policy import policy_decision


def test_risk_score_is_bounded():
    score = fuse_risk(
        model_probability=1.0,
        network_score=1000.0,
        behavioral_signal=1.0,
    )

    assert 0.0 <= score <= 100.0


def test_high_risk_policy_requires_review_or_block():
    score = fuse_risk(
        model_probability=0.90,
        network_score=8.0,
        behavioral_signal=1.0,
    )

    assert score >= 55
    assert policy_decision(score) in {
        "REVIEW",
        "BLOCK",
    }


def test_risk_levels_are_deterministic():
    assert risk_level(10) == "LOW"
    assert risk_level(50) == "MEDIUM"
    assert risk_level(75) == "HIGH"
    assert risk_level(90) == "CRITICAL"


def test_risk_case_serializes():
    case = RiskCase(
        transaction_id="T0001",
        risk_score=91.0,
        risk_level="CRITICAL",
        decision="BLOCK",
        primary_reason="network anomaly",
        evidence=["shared device", "new merchant"],
        model_probability=0.91,
        network_score=8.1,
    )

    payload = case.to_dict()

    assert payload["transaction_id"] == "T0001"
    assert payload["decision"] == "BLOCK"
    assert len(payload["evidence"]) == 2