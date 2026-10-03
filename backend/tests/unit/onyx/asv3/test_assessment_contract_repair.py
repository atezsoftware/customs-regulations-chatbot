"""Repair positive witness bookkeeping without changing the answer or negative law review."""

import json
from typing import Any

import pytest

from onyx.asv3.assertions import (
    AssertionVerification,
    AssertionWitness,
    assertion_inventory,
)
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import (
    DeterminationVerification,
    ResearchModel,
    VerificationResult,
)
from onyx.asv3.models import RunContext
from onyx.asv3.publication import publication_gap
from onyx.asv3.runtime import _evidence_record
from tests.unit.onyx.asv3.test_citation_contract import original_ledger, supported
from tests.unit.onyx.asv3.test_model_adapter import (
    scripted_model,
    text_response,
    tool_response,
)


def review_fixture(
    *, bad_quote: bool = True, negative: bool = False
) -> tuple[EvidenceLedger, RunContext, str, str, VerificationResult]:
    ledger, context = original_ledger()
    answer = "The condition applies [1]."
    if negative:
        answer += "\n\nAn automatic consequence follows [2]."
    units = assertion_inventory(answer)
    review = supported([1])
    review.assertion_results = [
        AssertionVerification(
            unit_id=units[0]["unit_id"],
            status="supported",
            witnesses=[
                AssertionWitness(
                    citation=1,
                    source_quote="condition...exception"
                    if bad_quote
                    else "condition AND exception",
                )
            ],
            explanation="The original condition supports this rule.",
        )
    ]
    if negative:
        review.status = "incomplete"
        review.safe_to_publish = False
        review.unsupported_claims = ["The automatic consequence"]
        review.assertion_results.append(
            AssertionVerification(
                unit_id=units[1]["unit_id"],
                status="unsupported",
                witnesses=[],
                missing_conditions=["An operative consequence is missing."],
                explanation="The original condition alone does not establish this outcome.",
            )
        )
    data = json.dumps(
        {
            "claim": answer,
            "scenario": "A hypothetical request",
            "questions": [
                {"question_id": "q0", "question": "Does the condition apply?"}
            ],
            "assertion_units": units,
            "evidence": _evidence_record(ledger, answer),
        }
    )
    return ledger, context, answer, data, review


def repaired_entry(review: VerificationResult) -> dict[str, Any]:
    entry = review.assertion_results[0].model_copy(deep=True)
    entry.witnesses[0].source_quote = "condition AND exception"
    return entry.model_dump(mode="json")


@pytest.mark.parametrize("native", [False, True])
def test_only_invalid_witnesses_are_patched_with_exact_original_delivery(
    native: bool,
) -> None:
    ledger, context, answer, data, review = review_fixture()
    patch = {
        "assertion_results": [repaired_entry(review)],
        "question_results": [],
        "need_results": [],
    }
    llm = scripted_model()
    llm.invoke.side_effect = [
        text_response(review.model_dump(mode="json")),
        tool_response(json.dumps({"parameter": patch}), "json_tool_call")
        if native
        else text_response(patch),
    ]
    model = ResearchModel(llm, context)
    result = model.invoke_verification("Assess the originals", data)
    assert llm.invoke.call_count == 2
    assert result.safe_to_publish and result.format_error is None
    supplied = json.loads(llm.invoke.call_args.kwargs["prompt"][1].content)
    assert supplied["claim"] == answer
    assert supplied["assessment_contract_repair"]["required_ids"] == {
        "assertion_results": [review.assertion_results[0].unit_id],
        "question_results": [],
        "need_results": [],
    }
    assert supplied["evidence"] == json.loads(data)["evidence"]
    assert model.last_call_id is not None
    assert ledger.completely_delivered(model.last_call_id) == {1}
    assert (
        publication_gap(
            answer,
            result,
            ["question"],
            ledger,
            require_assertion_checks=True,
            verification_call_id=model.last_call_id,
        )
        is None
    )


def test_valid_witness_review_needs_no_extra_call() -> None:
    _ledger, context, _answer, data, review = review_fixture(bad_quote=False)
    llm = scripted_model()
    llm.invoke.return_value = text_response(review.model_dump(mode="json"))
    result = ResearchModel(llm, context).invoke_verification("Assess", data)
    assert llm.invoke.call_count == 1
    assert result == review


def test_research_assessment_patch_consumes_its_own_shared_decision() -> None:
    _ledger, context, _answer, data, review = review_fixture()
    patch = {
        "assertion_results": [repaired_entry(review)],
        "question_results": [],
        "need_results": [],
    }
    llm = scripted_model()
    llm.invoke.side_effect = [
        text_response(review.model_dump(mode="json")),
        text_response(patch),
    ]
    context.consume_research_decision()
    result = ResearchModel(llm, context).invoke_verification(
        "Assess", data, consume_budget=False
    )
    assert result.safe_to_publish
    assert context.budget.snapshot()["decisions"] == 2


def test_negative_assessments_survive_a_patch_unchanged() -> None:
    _ledger, context, _answer, data, review = review_fixture(negative=True)
    patch = {
        "assertion_results": [repaired_entry(review)],
        "question_results": [],
        "need_results": [],
    }
    llm = scripted_model()
    llm.invoke.side_effect = [
        text_response(review.model_dump(mode="json")),
        text_response(patch),
    ]
    result = ResearchModel(llm, context).invoke_verification("Assess", data)
    assert llm.invoke.call_count == 2
    assert result.assertion_results[1] == review.assertion_results[1]
    assert result.unsupported_claims == review.unsupported_claims
    assert result.status == "incomplete" and not result.safe_to_publish


@pytest.mark.parametrize("error", ["nonliteral", "extra_id", "wrong_citation"])
def test_invalid_patch_cannot_grant_approval_or_erase_negative_results(
    error: str,
) -> None:
    ledger, context, answer, data, review = review_fixture(negative=True)
    entry = repaired_entry(review)
    patch = {"assertion_results": [entry], "question_results": [], "need_results": []}
    if error == "nonliteral":
        entry["witnesses"][0]["source_quote"] = "invented operative text"
    elif error == "wrong_citation":
        entry["witnesses"][0]["citation"] = 2
    else:
        other = review.assertion_results[1].model_copy(update={"status": "supported"})
        patch["assertion_results"].append(other.model_dump(mode="json"))
    llm = scripted_model()
    llm.invoke.side_effect = [
        text_response(review.model_dump(mode="json")),
        text_response(patch),
    ]
    result = ResearchModel(llm, context).invoke_verification("Assess", data)
    assert llm.invoke.call_count == 2
    assert result.format_error is not None and not result.safe_to_publish
    assert result.assertion_results == review.assertion_results
    assert result.unsupported_claims == review.unsupported_claims
    assert publication_gap(answer, result, ["question"], ledger) is not None


def test_reviewer_can_decline_to_supply_a_supported_witness() -> None:
    ledger, context, answer, data, review = review_fixture()
    entry = review.assertion_results[0].model_copy(
        update={
            "status": "uncertain",
            "witnesses": [],
            "missing_conditions": ["The cited original cannot establish this rule."],
        }
    )
    patch = {
        "assertion_results": [entry.model_dump(mode="json")],
        "question_results": [],
        "need_results": [],
    }
    llm = scripted_model()
    llm.invoke.side_effect = [
        text_response(review.model_dump(mode="json")),
        text_response(patch),
    ]
    result = ResearchModel(llm, context).invoke_verification("Assess", data)
    assert result.format_error is None
    assert result.status == "incomplete" and not result.safe_to_publish
    assert result.assertion_results[0].status == "uncertain"
    assert (
        publication_gap(
            answer, result, ["question"], ledger, require_assertion_checks=True
        )
        is not None
    )


def test_uncited_background_is_not_required_inline_support() -> None:
    _ledger, context, _answer, data, review = review_fixture(bad_quote=False)
    review.question_results[0].evidence_numbers = [1, 2]
    entry = review.question_results[0].model_copy(update={"evidence_numbers": [1]})
    patch = {
        "assertion_results": [],
        "question_results": [entry.model_dump(mode="json")],
        "need_results": [],
    }
    llm = scripted_model()
    llm.invoke.side_effect = [
        text_response(review.model_dump(mode="json")),
        text_response(patch),
    ]
    result = ResearchModel(llm, context).invoke_verification("Assess", data)
    assert result.question_results[0].evidence_numbers == [1]
    assert result.safe_to_publish and result.format_error is None


@pytest.mark.parametrize("attempt", ["upgrade", "omit"])
def test_bookkeeping_patch_cannot_erase_a_negative_requested_outcome(
    attempt: str,
) -> None:
    _ledger, context, _answer, data, review = review_fixture(bad_quote=False)
    review.question_results[0].evidence_numbers = [1, 2]
    negative = DeterminationVerification(
        determination_id="q0:d0",
        status="incomplete",
        answer_unit_ids=[],
        evidence_numbers=[],
        missing_conditions=["The later settlement is unsupported."],
    )
    review.question_results[0].determinations = [negative]
    entry = review.question_results[0].model_copy(deep=True)
    entry.evidence_numbers = [1]
    if attempt == "upgrade":
        entry.determinations[0].status = "supported"
        entry.determinations[0].missing_conditions = []
    else:
        entry.determinations = []
    patch = {
        "assertion_results": [],
        "question_results": [entry.model_dump(mode="json")],
        "need_results": [],
    }
    llm = scripted_model()
    llm.invoke.side_effect = [
        text_response(review.model_dump(mode="json")),
        text_response(patch),
    ]
    result = ResearchModel(llm, context).invoke_verification("Assess", data)
    assert result.question_results[0].determinations == [negative]
    assert not result.safe_to_publish
    assert llm.invoke.call_count == 2
