"""Investigation copilot service.

Orchestrates evidence context construction and LLM interaction
to provide grounded investigation assistance.

The copilot is advisory only.  It never modifies case state,
risk scores, or decisions.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from razorguard.copilot.context import (
    CopilotEvidenceContext,
    build_copilot_context,
    context_to_prompt,
)
from razorguard.copilot.provider import (
    CopilotResponse,
    LLMProvider,
    ProviderError,
    ProviderUnavailableError,
    get_provider,
)


logger = logging.getLogger(__name__)


# ============================================================
# SYSTEM PROMPT
# ============================================================


SYSTEM_PROMPT = """You are RazorGuard Investigation Copilot.

You ASSIST human fraud investigators by summarizing and explaining RazorGuard evidence.

CRITICAL RULES:
1. Use ONLY the RazorGuard evidence supplied in the context.
2. Never invent evidence, entities, transactions, or external facts.
3. Clearly distinguish VERIFIED RAZORGUARD EVIDENCE from your INTERPRETATION.
4. If evidence is insufficient, explicitly say so.
5. Never override the deterministic RazorGuard decision.
6. Never instruct the system to automatically change a case decision.
7. Recommendations are advisory only — the investigator makes the final decision.
8. Keep responses concise and actionable.

RESPONSE FORMAT:
- answer: Direct answer to the investigator's question
- key_evidence: List the most relevant verified evidence items
- interpretation: Your interpretation (clearly marked as AI interpretation)
- recommended_focus: What the investigator should focus on next
- grounding: Always indicate which parts are verified evidence vs interpretation

Always start your response with a brief summary, then provide details.
Mark AI-generated interpretations clearly with [AI INTERPRETATION].
Mark investigative suggestions clearly with [INVESTIGATIVE SUGGESTION].
"""  # noqa: E501


# ============================================================
# SUGGESTED QUESTIONS
# ============================================================


SUGGESTED_QUESTIONS = [
    "Why was this case flagged?",
    "What is the strongest evidence?",
    "What makes this transaction unusual?",
    "Summarize this case for me.",
    "What relationships should I review?",
    "Explain the coordinated-risk cluster.",
    "What should I investigate next?",
    "Why does this merchant matter?",
]


# ============================================================
# SERVICE
# ============================================================


def get_copilot_status() -> dict[str, Any]:
    """Return the current copilot status."""

    provider = get_provider()

    return {
        "available": provider.is_available(),
        "provider": type(provider).__name__,
    }


def answer_question(
    *,
    question: str,
    case_context: dict[str, Any],
    evidence: list[dict[str, Any]],
    network_data: dict[str, Any] | None,
    cluster_data: dict[str, Any] | None,
    investigation_steps: list[dict[str, Any]],
    audit_events: list[dict[str, Any]],
    evidence_summary: dict[str, int],
) -> CopilotResponse:
    """Answer an investigator question using grounded evidence.

    Returns a structured CopilotResponse.  If the LLM is
    unavailable, returns a deterministic fallback grounded
    in the evidence.
    """

    # Build evidence context
    ctx = build_copilot_context(
        case=case_context,
        evidence=evidence,
        network_data=network_data,
        cluster_data=cluster_data,
        investigation_steps=investigation_steps,
        audit_events=audit_events,
        evidence_summary=evidence_summary,
    )

    # Try LLM
    provider = get_provider()

    if provider.is_available():
        try:
            evidence_text = context_to_prompt(ctx)

            user_message = (
                f"Evidence Context:\n\n{evidence_text}\n\n"
                f"Investigator Question: {question}\n\n"
                f"Please answer using ONLY the verified RazorGuard "
                f"evidence above.  Mark any interpretation clearly."
            )

            raw_response = provider.generate(
                system_prompt=SYSTEM_PROMPT,
                user_message=user_message,
            )

            return _parse_llm_response(raw_response, ctx)

        except ProviderUnavailableError:
            logger.info("LLM provider unavailable, using fallback")
        except ProviderTimeoutError:
            logger.warning("LLM provider timed out, using fallback")
        except ProviderError as exc:
            logger.warning("LLM provider error: %s", exc)

    # Deterministic fallback
    return _fallback_answer(question, ctx)


def _parse_llm_response(
    raw: str,
    ctx: CopilotEvidenceContext,
) -> CopilotResponse:
    """Parse an LLM response into structured format."""

    # Extract key evidence from the response
    key_evidence = [
        e.title for e in ctx.evidence[:5]
    ]

    return CopilotResponse(
        answer=raw,
        key_evidence=key_evidence,
        interpretation="[AI INTERPRETATION] See answer above.",
        recommended_focus=_extract_focus(ctx),
        grounding="AI INTERPRETATION — verify against RazorGuard evidence",
    )


def _extract_focus(ctx: CopilotEvidenceContext) -> str:
    """Extract the recommended focus from the context."""

    if ctx.investigation_path:
        top = ctx.investigation_path[0]
        return f"[INVESTIGATIVE SUGGESTION] {top.title}: {top.reason}"

    if ctx.evidence:
        top = ctx.evidence[0]
        return f"Review: {top.title} — {top.investigative_relevance}"

    return "No specific recommendation available from the evidence."


def _fallback_answer(
    question: str,
    ctx: CopilotEvidenceContext,
) -> CopilotResponse:
    """Provide a deterministic fallback answer grounded in evidence."""

    question_lower = question.lower()

    # Summarize case
    if any(word in question_lower for word in ["summarize", "summary", "overview"]):
        return _summarize_case(ctx)

    # Strongest evidence
    if any(word in question_lower for word in ["strongest", "best", "most important", "key"]):
        return _strongest_evidence(ctx)

    # Why flagged
    if any(word in question_lower for word in ["why", "flagged", "reason", "risk"]):
        return _why_flagged(ctx)

    # What to investigate next
    if any(word in question_lower for word in ["next", "investigate", "focus", "step"]):
        return _what_next(ctx)

    # Relationships
    if any(word in question_lower for word in ["relationship", "connected", "entity", "account", "device", "merchant"]):
        return _relationships(ctx)

    # Cluster
    if any(word in question_lower for word in ["cluster", "coordinated", "network"]):
        return _explain_cluster(ctx)

    # Unusual
    if any(word in question_lower for word in ["unusual", "anomal", "strange", "different"]):
        return _what_unusual(ctx)

    # Default: general summary
    return _summarize_case(ctx)


def _summarize_case(ctx: CopilotEvidenceContext) -> CopilotResponse:
    """Summarize the case using verified evidence."""

    parts: list[str] = []

    parts.append(
        f"Case {ctx.case.case_id} involves transaction "
        f"{ctx.case.transaction_id}."
    )

    parts.append(
        f"The case is classified as {ctx.case.risk_level} risk "
        f"with a {ctx.case.decision} decision "
        f"(risk score: {ctx.case.risk_score:.1f})."
    )

    if ctx.case.primary_reason:
        parts.append(f"Primary reason: {ctx.case.primary_reason}.")

    if ctx.evidence:
        primary = [e for e in ctx.evidence if e.tier == "PRIMARY"]
        if primary:
            parts.append(
                f"Primary evidence ({len(primary)} items): "
                + "; ".join(e.title for e in primary[:3])
                + "."
            )

    if ctx.network:
        parts.append(
            f"Network intelligence: account {ctx.network.account_id}, "
            f"device {ctx.network.device_id}, "
            f"merchant {ctx.network.merchant_id}."
        )

    if ctx.cluster and ctx.cluster.account_count > 1:
        parts.append(
            f"Coordinated-risk cluster: {ctx.cluster.cluster_type} "
            f"with {ctx.cluster.account_count} connected accounts."
        )

    answer = " ".join(parts)

    return CopilotResponse(
        answer=answer,
        key_evidence=[e.title for e in ctx.evidence[:5]],
        interpretation="[VERIFIED RAZORGUARD EVIDENCE] All facts above are from the RazorGuard system.",
        recommended_focus=_extract_focus(ctx),
        grounding="VERIFIED EVIDENCE",
    )


def _strongest_evidence(ctx: CopilotEvidenceContext) -> CopilotResponse:
    """Identify the strongest evidence items."""

    if not ctx.evidence:
        return CopilotResponse(
            answer="No evidence items have been synthesized for this case.",
            key_evidence=[],
            interpretation="[VERIFIED] No evidence available.",
            recommended_focus="Review the case data manually.",
            grounding="VERIFIED EVIDENCE",
        )

    primary = [e for e in ctx.evidence if e.tier == "PRIMARY"]
    supporting = [e for e in ctx.evidence if e.tier == "SUPPORTING"]

    parts: list[str] = []

    if primary:
        parts.append(
            f"The strongest evidence ({len(primary)} PRIMARY items):"
        )
        for e in primary[:3]:
            parts.append(f"  - {e.title} [{e.severity}]: {e.explanation}")

    if supporting:
        parts.append(
            f"\nSupporting evidence ({len(supporting)} items):"
        )
        for e in supporting[:3]:
            parts.append(f"  - {e.title} [{e.severity}]: {e.explanation}")

    return CopilotResponse(
        answer="\n".join(parts),
        key_evidence=[e.title for e in ctx.evidence[:5]],
        interpretation="[VERIFIED RAZORGUARD EVIDENCE] All evidence items are from the RazorGuard synthesis engine.",
        recommended_focus=_extract_focus(ctx),
        grounding="VERIFIED EVIDENCE",
    )


def _why_flagged(ctx: CopilotEvidenceContext) -> CopilotResponse:
    """Explain why the case was flagged."""

    parts: list[str] = []

    parts.append(
        f"This case was flagged because {ctx.case.primary_reason}."
    )

    parts.append(
        f"The risk score is {ctx.case.risk_score:.1f} "
        f"({ctx.case.risk_level}), with a {ctx.case.decision} decision."
    )

    parts.append(
        f"Model probability: {ctx.case.model_probability:.1%}, "
        f"Network score: {ctx.case.network_score:.1f}."
    )

    if ctx.evidence:
        high = [e for e in ctx.evidence if e.severity == "HIGH"]
        if high:
            parts.append(
                f"\nHigh-severity evidence ({len(high)} items):"
            )
            for e in high[:3]:
                parts.append(f"  - {e.title}: {e.explanation}")

    return CopilotResponse(
        answer="\n".join(parts),
        key_evidence=[e.title for e in ctx.evidence[:3]],
        interpretation="[VERIFIED RAZORGUARD EVIDENCE] All risk signals are from the deterministic RazorGuard system.",
        recommended_focus=_extract_focus(ctx),
        grounding="VERIFIED EVIDENCE",
    )


def _what_next(ctx: CopilotEvidenceContext) -> CopilotResponse:
    """Recommend next investigation steps."""

    if not ctx.investigation_path:
        return CopilotResponse(
            answer="No investigation steps have been generated for this case.",
            key_evidence=[],
            interpretation="[VERIFIED] No steps available.",
            recommended_focus="Review the case evidence manually.",
            grounding="VERIFIED EVIDENCE",
        )

    parts: list[str] = []
    parts.append("Recommended next steps:")

    for step in ctx.investigation_path[:5]:
        parts.append(
            f"  [P{step.priority}] {step.title}: {step.reason}"
        )

    return CopilotResponse(
        answer="\n".join(parts),
        key_evidence=[s.title for s in ctx.investigation_path[:3]],
        interpretation="[INVESTIGATIVE SUGGESTION] These steps are generated by the RazorGuard investigation path engine.",
        recommended_focus=ctx.investigation_path[0].title if ctx.investigation_path else "",
        grounding="VERIFIED EVIDENCE",
    )


def _relationships(ctx: CopilotEvidenceContext) -> CopilotResponse:
    """Explain entity relationships."""

    if not ctx.network:
        return CopilotResponse(
            answer="No network intelligence is available for this transaction.",
            key_evidence=[],
            interpretation="[VERIFIED] No network data available.",
            recommended_focus="Check if the transaction dataset is loaded.",
            grounding="VERIFIED EVIDENCE",
        )

    parts: list[str] = []

    parts.append(
        f"Transaction {ctx.case.transaction_id} involves:"
    )
    parts.append(f"  Account: {ctx.network.account_id}")
    parts.append(f"  Device: {ctx.network.device_id}")
    parts.append(f"  Merchant: {ctx.network.merchant_id}")

    if ctx.network.accounts_on_device > 1:
        parts.append(
            f"\nThe device is shared across {ctx.network.accounts_on_device} "
            f"accounts. This is a significant coordination signal."
        )

    if ctx.network.accounts_at_merchant > 3:
        parts.append(
            f"The merchant has been used by {ctx.network.accounts_at_merchant} "
            f"accounts. High merchant concentration warrants review."
        )

    if ctx.network.new_device_for_account:
        parts.append(
            "This device is NEW for this account, which may indicate "
            "account compromise or a legitimate device change."
        )

    return CopilotResponse(
        answer="\n".join(parts),
        key_evidence=[
            f"Device: {ctx.network.device_id}",
            f"Merchant: {ctx.network.merchant_id}",
        ],
        interpretation="[VERIFIED RAZORGUARD EVIDENCE] All relationship data is from the RazorGuard network intelligence engine.",
        recommended_focus="Review the shared device and merchant relationships.",
        grounding="VERIFIED EVIDENCE",
    )


def _explain_cluster(ctx: CopilotEvidenceContext) -> CopilotResponse:
    """Explain the coordinated-risk cluster."""

    if not ctx.cluster:
        return CopilotResponse(
            answer="No coordinated-risk cluster was detected for this transaction.",
            key_evidence=[],
            interpretation="[VERIFIED] No cluster detected.",
            recommended_focus="Review individual network signals instead.",
            grounding="VERIFIED EVIDENCE",
        )

    parts: list[str] = []

    parts.append(
        f"The coordinated-risk cluster is classified as "
        f"{ctx.cluster.cluster_type} with a risk score of "
        f"{ctx.cluster.risk_score:.1f}."
    )

    parts.append(
        f"It connects {ctx.cluster.account_count} accounts, "
        f"{ctx.cluster.device_count} devices, "
        f"and {ctx.cluster.merchant_count} merchants "
        f"across {ctx.cluster.transaction_count} transactions."
    )

    if ctx.cluster.signals:
        parts.append("\nCluster signals:")
        for sig in ctx.cluster.signals:
            parts.append(f"  - {sig}")

    if ctx.cluster.evidence_statements:
        parts.append("\nEvidence:")
        for ev in ctx.cluster.evidence_statements:
            parts.append(f"  - {ev}")

    return CopilotResponse(
        answer="\n".join(parts),
        key_evidence=ctx.cluster.signals[:3],
        interpretation="[VERIFIED RAZORGUARD EVIDENCE] All cluster data is from the RazorGuard coordinated-risk engine.",
        recommended_focus="Review all connected entities in the cluster.",
        grounding="VERIFIED EVIDENCE",
    )


def _what_unusual(ctx: CopilotEvidenceContext) -> CopilotResponse:
    """Identify what makes the transaction unusual."""

    parts: list[str] = []

    parts.append(
        f"Transaction {ctx.case.transaction_id} has several "
        f"unusual characteristics:"
    )

    unusual: list[str] = []

    if ctx.network:
        if ctx.network.new_device_for_account:
            unusual.append("New device for this account")
        if ctx.network.new_merchant_for_account:
            unusual.append("New merchant relationship")
        if ctx.network.device_shared:
            unusual.append(
                f"Device shared across {ctx.network.accounts_on_device} accounts"
            )
        if ctx.network.merchant_shared:
            unusual.append(
                f"Merchant used by {ctx.network.accounts_at_merchant} accounts"
            )

    if ctx.cluster and ctx.cluster.cluster_type == "COORDINATED_NETWORK":
        unusual.append(
            f"Part of a COORDINATED_NETWORK cluster "
            f"(score: {ctx.cluster.risk_score:.1f})"
        )

    if unusual:
        for item in unusual:
            parts.append(f"  - {item}")
    else:
        parts.append("  No specific anomalies detected beyond the risk score.")

    return CopilotResponse(
        answer="\n".join(parts),
        key_evidence=unusual[:5],
        interpretation="[VERIFIED RAZORGUARD EVIDENCE] All anomaly signals are from the RazorGuard system.",
        recommended_focus="Review the unusual characteristics listed above.",
        grounding="VERIFIED EVIDENCE",
    )
