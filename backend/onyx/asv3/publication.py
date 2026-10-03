"""Evaluate evidence-backed publication without prescribing research methods."""

from __future__ import annotations

from pydantic import JsonValue

from onyx.asv3.assertions import (
    assertion_inventory,
    assertion_support_defect,
    assertion_witness_valid,
)
from onyx.asv3.authority import unresolved_authority_gap
from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import VerificationResult
from onyx.asv3.models import OutcomeStatus, ToolOutcome
from onyx.asv3.quotations import normalized, unmatched_quoted_terms
from onyx.asv3.research_state import ResearchState
from onyx.asv3.scenario import question_determinations


def question_inventory(questions: list[str]) -> list[dict[str, JsonValue]]:
    determinations = question_determinations(questions)
    return [
        {
            "question_id": f"q{index}",
            "question": text,
            "determinations": [
                dict(item)
                for item in determinations
                if item["question_id"] == f"q{index}"
            ],
        }
        for index, text in enumerate(questions)
    ]


def determination_support_gaps(
    answer: str,
    review: VerificationResult,
    questions: list[str],
    *,
    require_sources: bool,
    allow_explicit_gaps: bool,
) -> list[dict[str, JsonValue]]:
    """Require each requested outcome to point to its independently assessed answer."""
    determination_gaps: list[dict[str, JsonValue]] = []
    units_by_id = {unit["unit_id"]: unit for unit in assertion_inventory(answer)}
    assessments = {item.unit_id: item for item in review.assertion_results}
    questions_by_id = {item.question_id: item for item in review.question_results}
    for row in question_inventory(questions):
        question_id = str(row["question_id"])
        question_review = questions_by_id.get(question_id)
        results = question_review.determinations if question_review else []
        expected = [
            item
            for item in question_determinations(questions)
            if item["question_id"] == question_id
        ]
        by_id = {item.determination_id: item for item in results}
        if set(by_id) != {item["determination_id"] for item in expected} or len(
            by_id
        ) != len(results):
            determination_gaps.append(
                {
                    "question_id": question_id,
                    "defect": "Each requested determination needs its own current assessment.",
                }
            )
        for item in expected:
            assessed = by_id.get(item["determination_id"])
            if assessed is None:
                determination_gaps.append(
                    {
                        **item,
                        "defect": "This requested determination was not assessed.",
                    }
                )
                continue
            if assessed.status != "supported":
                if (
                    allow_explicit_gaps
                    and assessed.status != "contradicted"
                    and assessed.missing_conditions
                ):
                    continue
                determination_gaps.append(
                    {**item, "assessment": assessed.model_dump(mode="json")}
                )
                continue
            bound_units = [units_by_id.get(key) for key in assessed.answer_unit_ids]
            bound_checks = [assessments.get(key) for key in assessed.answer_unit_ids]
            inline_numbers = {
                number
                for unit in bound_units
                if unit
                for number in unit["evidence_numbers"]
            }
            if (
                assessed.missing_conditions
                or not bound_units
                or any(unit is None for unit in bound_units)
                or any(
                    check is None
                    or check.status != "supported"
                    or check.basis not in {"original", "scenario"}
                    for check in bound_checks
                )
                or (require_sources and not assessed.evidence_numbers)
                or set(assessed.evidence_numbers) - inline_numbers
            ):
                determination_gaps.append(
                    {
                        **item,
                        "assessment": assessed.model_dump(mode="json"),
                        "defect": "Support must come from the assessed answer blocks addressing this determination, not another answer in the same question.",
                    }
                )
    return determination_gaps


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
    research_state: ResearchState | None = None,
    require_assertion_checks: bool = False,
    require_determination_checks: bool = False,
) -> ToolOutcome | None:
    if review.format_error is not None:
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="Publication assessment format failed; the draft and originals are retained.",
            data={
                "verification_format_error": review.format_error,
                "instruction": "Request a fresh original-source assessment of this exact draft. A format failure is not a legal evidence gap and does not require rewriting the answer or rereading already retained sources. Publication still requires a valid assessment.",
            },
        )
    reasons: list[str] = []
    cited = set(extract_citation_numbers(answer))
    assertion_gaps: list[dict[str, JsonValue]] = []
    if require_sources and require_assertion_checks:
        units = assertion_inventory(answer)
        checks = {item.unit_id: item for item in review.assertion_results}
        if len(checks) != len(review.assertion_results) or set(checks) != {
            str(unit["unit_id"]) for unit in units
        }:
            reasons.append(
                "Every answer block, including uncited claims, needs a current assessment of its own basis; a general approval is insufficient."
            )
        originals = {
            number: item.text
            for number in cited
            if (item := ledger.get(number)) is not None
        }
        for unit in units:
            check = checks.get(str(unit["unit_id"]))
            defect = (
                assertion_support_defect(
                    unit,
                    check,
                    originals,
                    scenario,
                    allow_explicit_gaps=allow_explicit_gaps,
                )
                if check
                else "This block was not assessed."
            )
            if defect:
                assertion_gaps.append(
                    {
                        **unit,
                        "defect": defect,
                        "assessment": check.model_dump(mode="json") if check else None,
                    }
                )
        if assertion_gaps:
            reasons.append(
                "Legal assertions require their own inline originals; scenario facts, presentation and explicit evidence gaps require their distinct valid basis."
            )
    determination_gaps = (
        determination_support_gaps(
            answer,
            review,
            questions,
            require_sources=require_sources,
            allow_explicit_gaps=allow_explicit_gaps,
        )
        if require_determination_checks
        else []
    )
    if determination_gaps:
        reasons.append(
            "Independent outcomes and alternatives inside numbered questions need their own assessed answer support."
        )
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
            if check is None or check.kind not in {"translation", "application"}:
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
            cited
            | set(review.evidence_numbers)
            | {n for need in review.need_results for n in need.evidence_numbers}
            | {
                n
                for item in review.question_results
                for part in item.determinations
                for n in part.evidence_numbers
            }
            | {item.witness.citation for item in review.omitted_material_source_details}
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
    if review.omitted_supported_details:
        reasons.append(
            "Useful original-supported details were lost: "
            + str(review.omitted_supported_details)
        )
    determination_ids = {
        item["determination_id"] for item in question_determinations(questions)
    }
    for omission in review.omitted_material_source_details:
        original = ledger.get(omission.witness.citation)
        if (
            original is None
            or omission.witness.citation not in allowed
            or not assertion_witness_valid(
                omission.witness, {omission.witness.citation: original.text}
            )
            or set(omission.determination_ids) - determination_ids
            or len(set(omission.determination_ids)) != len(omission.determination_ids)
        ):
            reasons.append(
                "A material-detail assessment has an invalid original witness or requested determination identity; correct the assessment against retained originals."
            )
        else:
            reasons.append(
                "An applicable detail is present in the supplied original but absent from the answer: "
                + omission.detail
            )
    if research_state is not None:
        if research_state.require_need_bindings:
            uncovered = research_state.uncovered_questions()
            if uncovered:
                reasons.append(
                    "Original questions lack material research needs and completion tests: "
                    + str(uncovered)
                )
        state = research_state.export()
        needs = state.get("needs", [])
        material = (
            {
                str(row["need_id"])
                for row in needs
                if isinstance(row, dict)
                and row.get("material")
                and row.get("status") != "out_of_scope"
            }
            if isinstance(needs, list)
            else set()
        )
        actual_needs = [item.need_id for item in review.need_results]
        if material != set(actual_needs) or len(actual_needs) != len(material):
            reasons.append(
                "Review must assess every material research need against its completion test, independently of which norms the answer names."
            )
        for item in review.need_results:
            if item.status == "supported":
                if (
                    item.missing_conditions
                    or (require_sources and not item.evidence_numbers)
                    or set(item.evidence_numbers) - cited
                ):
                    reasons.append(
                        f"Incomplete original support for research need: {item.need_id}."
                    )
            elif (
                not allow_explicit_gaps
                or item.status == "contradicted"
                or not item.missing_conditions
            ):
                reasons.append(f"Unresolved material research need: {item.need_id}.")
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
    if (
        set(review.evidence_numbers)
        | {n for need in review.need_results for n in need.evidence_numbers}
    ) - allowed.keys():
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
            **({"assertion_gaps": assertion_gaps} if assertion_gaps else {}),
            **(
                {"determination_gaps": determination_gaps} if determination_gaps else {}
            ),
            **(
                {
                    "omitted_material_source_details": [
                        item.model_dump(mode="json")
                        for item in review.omitted_material_source_details
                    ]
                }
                if review.omitted_material_source_details
                else {}
            ),
            "review": review.model_dump(mode="json"),
            "instruction": "A found source may need delivery or complete reading, not another search. Source-bound omissions require a targeted answer edit using the supplied original; do not search again for text already delivered. Choose methods yourself. Do not substitute general legal knowledge.",
        },
    )
