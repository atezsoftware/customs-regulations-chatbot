"""Publish supported outcomes while explicitly retaining a precise unresolved condition."""

import json

import pytest

from onyx.asv3.assertions import (
    AssertionVerification,
    AssertionWitness,
    assertion_inventory,
)
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import (
    DeterminationVerification,
    QuestionVerification,
    ResearchModel,
    SourceConditionAuditResult,
    SourceConditionCheck,
    SourceConditionResolution,
    VerificationResult,
)
from onyx.asv3.models import RunContext
from onyx.asv3.publication import publication_gap
from onyx.asv3.research_state import ResearchState
from onyx.asv3.runtime import _evidence_record  # pyright: ignore[reportPrivateUsage]
from onyx.asv3.source_conditions import answer_hash, complete_condition_review
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.asv3.test_citation_contract import original_ledger
from tests.unit.onyx.asv3.test_model_adapter import scripted_model, text_response

pytestmark = pytest.mark.usefixtures("empty_source_inventory")


def partial_case() -> tuple[
    EvidenceLedger,
    RunContext,
    ResearchState,
    str,
    VerificationResult,
    SourceConditionCheck,
]:
    ledger, context = original_ledger()
    questions = ["What operation is permitted?", "What is the exception's effect?"]
    state = ResearchState(questions, context)
    missing = "The applicability of the referenced exception remains unresolved."
    answer = "The operation is permitted [1].\n\n" + missing
    units = assertion_inventory(answer)
    review = VerificationResult(
        status="incomplete",
        explanation="The first outcome is supported; the second explicitly remains open.",
        safe_to_publish=True,
        required_conditions=[],
        missing_conditions=[missing],
        evidence_numbers=[1],
        assertion_results=[
            AssertionVerification(
                unit_id=units[0]["unit_id"],
                status="supported",
                witnesses=[
                    AssertionWitness(citation=1, source_quote="condition AND exception")
                ],
            ),
            AssertionVerification(
                unit_id=units[1]["unit_id"],
                status="uncertain",
                basis="evidence_gap",
                missing_conditions=[missing],
            ),
        ],
        question_results=[
            QuestionVerification(
                question_id="q0",
                status="supported",
                evidence_numbers=[1],
                missing_conditions=[],
                determinations=[
                    DeterminationVerification(
                        determination_id="q0:d0",
                        status="supported",
                        answer_unit_ids=[units[0]["unit_id"]],
                        evidence_numbers=[1],
                        missing_conditions=[],
                    )
                ],
            ),
            QuestionVerification(
                question_id="q1",
                status="uncertain",
                evidence_numbers=[],
                missing_conditions=[missing],
                determinations=[
                    DeterminationVerification(
                        determination_id="q1:d0",
                        status="uncertain",
                        answer_unit_ids=[units[1]["unit_id"]],
                        evidence_numbers=[],
                        missing_conditions=[missing],
                    )
                ],
            ),
        ],
    )
    condition = SourceConditionCheck(
        witness=AssertionWitness(citation=3, source_quote="condition AND exception"),
        determination_ids=["q1:d0"],
        detail="The referenced exception needs its operative scope.",
        applicability="It affects the second requested outcome.",
        disposition="uncertain",
        answer_unit_ids=[units[1]["unit_id"]],
    )
    state.source_conditions.remember([condition], ledger)
    return ledger, context, state, answer, review, condition


def attach_audit(
    ledger: EvidenceLedger,
    answer: str,
    review: VerificationResult,
    condition: SourceConditionCheck,
) -> VerificationResult:
    evidence = json.loads(
        _evidence_record(ledger, answer, include_supplemental_originals=True)
    )
    ledger.record_delivery("main", LLMFlow.ASV3_VERIFICATION.value, evidence)
    ledger.record_delivery("conditions", LLMFlow.ASV3_CONDITION_REVIEW.value, evidence)
    return review.model_copy(
        update={
            "condition_review": SourceConditionAuditResult(
                examined_citations=[1, 2, 3], conditions=[condition]
            ),
            "condition_review_call_id": "conditions",
            "condition_review_answer_hash": answer_hash(answer),
        }
    )


def test_precisely_disclosed_uncertainty_preserves_other_supported_outcomes() -> None:
    ledger, _context, state, answer, review, condition = partial_case()
    review = attach_audit(ledger, answer, review, condition)
    gap = publication_gap(
        answer,
        review,
        list(state.questions),
        ledger,
        research_state=state,
        verification_call_id="main",
        require_assertion_checks=True,
        require_determination_checks=True,
        require_condition_review=True,
        allow_explicit_gaps=True,
    )
    assert gap is None
    assert (
        publication_gap(
            answer,
            review,
            list(state.questions),
            ledger,
            research_state=state,
            require_condition_review=True,
            allow_explicit_gaps=False,
        )
        is not None
    )
    assert len(state.source_conditions.required_conditions()) == 1


@pytest.mark.parametrize("statute_in_gap", [False, True])
def test_partial_answer_does_not_bypass_governing_original_for_supported_outcome(
    statute_in_gap: bool,
) -> None:
    ledger, _context, state, answer, review, condition = partial_case()
    if statute_in_gap:
        replacement = (
            "The scope of the referenced exception in Law 789 remains unresolved."
        )
        answer = answer.replace(review.missing_conditions[0], replacement)
        review.missing_conditions = [replacement]
        review.assertion_results[1].missing_conditions = [replacement]
        review.question_results[1].missing_conditions = [replacement]
        review.question_results[1].determinations[0].missing_conditions = [replacement]
    else:
        answer = answer.replace(
            "The operation is permitted", "Law 789 permits the operation"
        )
    units = assertion_inventory(answer)
    for index, unit in enumerate(units):
        review.assertion_results[index].unit_id = unit["unit_id"]
        review.question_results[index].determinations[0].answer_unit_ids = [
            unit["unit_id"]
        ]
    condition.answer_unit_ids = [units[1]["unit_id"]]
    review = attach_audit(ledger, answer, review, condition)
    gap = publication_gap(
        answer,
        review,
        list(state.questions),
        ledger,
        research_state=state,
        verification_call_id="main",
        require_assertion_checks=True,
        require_determination_checks=True,
        require_condition_review=True,
        allow_explicit_gaps=True,
    )
    assert (gap is None) is statute_in_gap


@pytest.mark.parametrize(
    "defect",
    [
        "omitted",
        "unbound",
        "wrong_outcome",
        "empty_proof",
        "stale_unit",
        "pretend_supported",
    ],
)
def test_partial_publication_cannot_hide_a_known_omission_or_unbound_uncertainty(
    defect: str,
) -> None:
    ledger, _context, state, answer, review, condition = partial_case()
    if defect == "omitted":
        condition.disposition = "omitted"
    elif defect == "unbound":
        condition.answer_unit_ids = []
    elif defect == "wrong_outcome":
        condition.determination_ids = ["q0:d0"]
    elif defect == "empty_proof":
        review.assertion_results[1].missing_conditions = []
    elif defect == "stale_unit":
        condition.answer_unit_ids = ["au-stale"]
    else:
        review.question_results[1].determinations[0].status = "supported"
    # No retained-ID mismatch is needed to reject the invalid partial answer.
    state = ResearchState(list(state.questions), _context)
    review = attach_audit(ledger, answer, review, condition)
    assert (
        publication_gap(
            answer,
            review,
            list(state.questions),
            ledger,
            research_state=state,
            verification_call_id="main",
            require_assertion_checks=True,
            require_determination_checks=True,
            require_condition_review=True,
            allow_explicit_gaps=True,
        )
        is not None
    )


@pytest.mark.parametrize("provider", ["vertex_ai", "anthropic", "openai"])
@pytest.mark.parametrize("partial", [False, True])
def test_independent_condition_review_keeps_precise_gap_open_without_erasing_answer(
    provider: str, partial: bool
) -> None:
    ledger, context, state, answer, review, condition = partial_case()
    rows = state.source_conditions.required_conditions()
    identity = str(rows[0]["condition_id"])
    evidence = _evidence_record(ledger, answer, include_supplemental_originals=True)
    ledger.record_delivery(
        "main", LLMFlow.ASV3_VERIFICATION.value, json.loads(evidence)
    )
    llm = scripted_model()
    llm.config.model_provider = provider
    llm.invoke.return_value = text_response(
        SourceConditionAuditResult(
            examined_citations=[1, 2, 3],
            conditions=[],
            resolutions=[
                SourceConditionResolution(
                    condition_id=identity,
                    disposition="uncertain",
                    answer_unit_ids=condition.answer_unit_ids,
                )
            ],
        ).model_dump(mode="json")
    )
    model = ResearchModel(llm, context)
    model.last_call_id = "main"
    result = complete_condition_review(
        review,
        model,
        ledger,
        answer=answer,
        scenario="facts",
        questions=list(state.questions),
        evidence=evidence,
        language="en",
        consume_budget=True,
        condition_memory=state.source_conditions,
        allow_explicit_gaps=partial,
    )
    assert result.safe_to_publish is partial
    assert result.status == "incomplete" and result.missing_conditions
    assert model.last_call_id == "main" and llm.invoke.call_count == 1
    assert (
        publication_gap(
            answer,
            result,
            list(state.questions),
            ledger,
            research_state=state,
            require_assertion_checks=True,
            require_determination_checks=True,
            require_condition_review=True,
            verification_call_id="main",
            allow_explicit_gaps=partial,
        )
        is None
    ) is partial
    assert len(state.source_conditions.required_conditions()) == 1
    assert state.source_conditions.materialize(
        SourceConditionAuditResult(examined_citations=[1, 2, 3], conditions=[]), ledger
    )[1]
