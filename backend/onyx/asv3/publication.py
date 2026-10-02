"""Evaluate evidence-backed publication without prescribing research methods."""

from __future__ import annotations

from pydantic import JsonValue

from onyx.asv3.authority import unresolved_authority_gap
from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import VerificationResult
from onyx.asv3.models import OutcomeStatus, ToolOutcome
from onyx.asv3.quotations import normalized, unmatched_quoted_terms


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
    require_direct_authority: bool = False,
    scenario: str = "",
    require_quotation_checks: bool = False,
) -> ToolOutcome | None:
    reasons: list[str] = []
    cited = set(extract_citation_numbers(answer))
    authority_gap = (
        unresolved_authority_gap(answer, ledger)
        if require_sources and require_direct_authority
        else None
    )
    if authority_gap is not None:
        reasons.append(str(authority_gap["gaps"]))
    quote_gaps: list[dict[str, JsonValue]] = []
    if require_sources and require_quotation_checks:
        checks = {check.term_id: check for check in review.quotation_checks}
        for term in unmatched_quoted_terms(answer, scenario, ledger):
            check = checks.get(str(term["term_id"]))
            # An unmatched literal cannot become supported just by being labelled so.
            if check is None or check.kind in {"literal", "unsupported"}:
                quote_gaps.append(term)
                continue
            evidence = (
                ledger.get(check.evidence_number) if check.evidence_number else None
            )
            numbers = term["evidence_numbers"]
            if (
                evidence is None
                or not isinstance(numbers, list)
                or check.evidence_number not in numbers
                or not check.source_quote.strip()
                or normalized(check.source_quote) not in normalized(evidence.text)
            ):
                quote_gaps.append(term)
        if quote_gaps:
            reasons.append(
                "Quoted wording has no literal inline-source match or validated translation/application witness."
            )
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
            **({"missing": authority_gap["missing"]} if authority_gap else {}),
            **({"unmatched_quoted_terms": quote_gaps} if quote_gaps else {}),
            "review": review.model_dump(mode="json"),
            "instruction": "A found source may need delivery or complete reading, not another search. Choose methods yourself. Do not substitute general legal knowledge.",
        },
    )
