"""Evaluate evidence-backed publication without prescribing research methods."""

from __future__ import annotations

from pydantic import JsonValue

from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import VerificationResult
from onyx.asv3.models import OutcomeStatus, ToolOutcome


def question_inventory(questions: list[str]) -> list[dict[str, JsonValue]]:
    return [
        {"question_id": f"q{index}", "question": text}
        for index, text in enumerate(questions)
    ]


def publication_gap(
    answer: str,
    review: VerificationResult,
    questions: list[str],
    ledger: EvidenceLedger,
    *,
    require_sources: bool = True,
    allow_explicit_gaps: bool = False,
    verification_call_id: str | None = None,
) -> ToolOutcome | None:
    reasons: list[str] = []
    cited = set(extract_citation_numbers(answer))
    allowed = ledger.citation_mapping()
    if cited - allowed.keys():
        reasons.append("The draft contains unknown or non-citable source numbers.")
    if require_sources and verification_call_id is not None:
        missing_delivery = (
            cited | set(review.evidence_numbers)
        ) - ledger.completely_delivered(verification_call_id)
        if missing_delivery:
            reasons.append(
                "Original evidence was not completely delivered to this verifier call: "
                + str(sorted(missing_delivery))
            )
    explicit_gap_only = (
        allow_explicit_gaps
        and review.safe_to_publish
        and not review.unsupported_claims
        and bool(review.question_results)
        and all(
            item.status in ("incomplete", "uncertain") and item.missing_conditions
            for item in review.question_results
        )
    )
    if require_sources and not cited and not explicit_gap_only:
        reasons.append("Legal conclusions need original, citable corpus evidence.")
    requested = {str(item["question_id"]) for item in question_inventory(questions)}
    actual = [item.question_id for item in review.question_results]
    if set(actual) != requested or len(actual) != len(requested):
        reasons.append(
            "Verification must cover the complete question inventory, including alternatives."
        )
    if not review.safe_to_publish or review.unsupported_claims:
        reasons.append(
            "The answer still contains unsupported conclusions or is unsafe to publish."
        )
    if not allow_explicit_gaps and (
        review.status != "supported" or review.missing_conditions
    ):
        reasons.append("Decisive conditions or counterfactuals remain unresolved.")
    for result in review.question_results:
        if result.status != "supported":
            if (
                not allow_explicit_gaps
                or result.status == "contradicted"
                or not result.missing_conditions
            ):
                reasons.append(
                    f"Unresolved or contradicted information need: {result.question_id}."
                )
            continue
        if result.missing_conditions:
            reasons.append(
                f"Missing conditions in supported information need: {result.question_id}."
            )
        if require_sources and not result.evidence_numbers:
            reasons.append(
                f"No original support for information need: {result.question_id}."
            )
        if set(result.evidence_numbers) - cited:
            reasons.append(
                f"The cited answer does not carry support for information need: {result.question_id}."
            )
    if set(review.evidence_numbers) - allowed.keys():
        reasons.append("Verifier used an unknown or non-citable source.")
    if not reasons:
        return None
    return ToolOutcome(
        status=OutcomeStatus.PARTIAL,
        summary="Evidence-backed finalization is incomplete; choose a useful next action or report the exact gap.",
        data={
            "gaps": reasons,
            "review": review.model_dump(mode="json"),
            "instruction": "A found source may need delivery or complete reading, not another search. Choose methods yourself. Do not substitute general legal knowledge.",
        },
    )
