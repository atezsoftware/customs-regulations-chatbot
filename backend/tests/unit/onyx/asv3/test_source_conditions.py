"""Separate omitted-original conditions from claim support and new research."""

import json

import pytest

from onyx.asv3.assertions import (
    AssertionVerification,
    AssertionWitness,
    assertion_inventory,
    assertion_support_defect,
)
from onyx.asv3.llm_adapter import (
    ResearchModel,
    SourceConditionAuditResult,
    SourceConditionCheck,
)
from onyx.asv3.publication import publication_gap
from onyx.asv3.runtime import _evidence_record
from onyx.asv3.source_conditions import (
    answer_hash,
    complete_condition_review,
    condition_omissions,
    condition_payload,
    condition_review_defects,
)
from onyx.asv3.witnesses import original_witness_spans
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.asv3.test_citation_contract import original_ledger, supported
from tests.unit.onyx.asv3.test_model_adapter import scripted_model, text_response


@pytest.fixture
def condition() -> SourceConditionCheck:
    ledger, _context = original_ledger()
    item = ledger.get(3)
    assert item is not None
    return SourceConditionCheck(
        witness=AssertionWitness(
            citation=3, witness_id=original_witness_spans(3, item.text)[0]["witness_id"]
        ),
        determination_ids=["q0:d0"],
        detail="An applicable condition is stated in a separate implementing original.",
        applicability="The requested operation depends on this prerequisite.",
        disposition="omitted",
    )


@pytest.mark.parametrize("provider", ["vertex_ai", "anthropic", "openai"])
def test_focused_review_retains_uncited_original_and_main_call_identity(
    provider: str, condition: SourceConditionCheck
) -> None:
    ledger, context = original_ledger()
    answer = "The operation is permitted [1]."
    evidence = _evidence_record(
        ledger, answer, include_witness_spans=True, include_supplemental_originals=True
    )
    audit = SourceConditionAuditResult(
        examined_citations=[1, 2, 3], conditions=[condition]
    )
    llm = scripted_model()
    llm.config.model_provider = provider
    llm.invoke.return_value = text_response(audit.model_dump(mode="json"))
    model = ResearchModel(llm, context)
    model.last_call_id = "main-assertion-review"
    review = complete_condition_review(
        supported([1]),
        model,
        ledger,
        answer=answer,
        scenario="facts",
        questions=["What procedure applies?"],
        evidence=evidence,
        language="en",
        consume_budget=True,
    )
    assert model.last_call_id == "main-assertion-review"
    assert review.status == "incomplete" and not review.safe_to_publish
    assert (
        review.condition_review is not None
        and review.condition_review_call_id is not None
    )
    assert ledger.completely_delivered(review.condition_review_call_id) == {1, 2, 3}
    assert (
        ledger.delivery_flow(review.condition_review_call_id)
        == LLMFlow.ASV3_CONDITION_REVIEW.value
    )
    assert review.question_results == supported([1]).question_results
    assert llm.invoke.call_count == 1 and llm.invoke.call_args.kwargs["tools"] is None
    assert llm.invoke.call_args.kwargs["tool_choice"].value == "none"
    for partial in (True, False):
        gap = publication_gap(
            answer,
            review,
            ["What procedure applies?"],
            ledger,
            scenario="facts",
            require_condition_review=True,
            allow_explicit_gaps=partial,
        )
        assert gap is not None and condition.detail in str(gap.data)
        assert "do not search again" in str(gap.data["instruction"])


@pytest.mark.parametrize(
    "defect",
    [
        "wrong_witness",
        "wrong_determination",
        "wrong_unit",
        "missing_units",
        "non_literal_exclusion",
        "incomplete_inventory",
        "wrong_flow",
    ],
)
def test_independent_assessment_rejects_foreign_identity_or_unproved_exclusion(
    defect: str, condition: SourceConditionCheck
) -> None:
    ledger, _context = original_ledger()
    answer = "The operation is permitted [1]."
    audit = SourceConditionAuditResult(
        examined_citations=[1, 2, 3], conditions=[condition]
    )
    rows = json.loads(
        _evidence_record(ledger, answer, include_supplemental_originals=True)
    )
    ledger.record_delivery("audit", LLMFlow.ASV3_CONDITION_REVIEW.value, rows)
    if defect == "wrong_witness":
        condition.witness.citation = 2
    elif defect == "wrong_determination":
        condition.determination_ids = ["q999:d0"]
    elif defect == "wrong_unit":
        condition.answer_unit_ids = ["stale-answer-unit"]
    elif defect == "missing_units":
        condition.disposition = "covered"
    elif defect == "non_literal_exclusion":
        condition.disposition = "not_applicable"
        condition.scenario_quotes = ["not in the supplied scenario"]
    elif defect == "incomplete_inventory":
        audit.examined_citations = [1]
    else:
        ledger.record_delivery("main", LLMFlow.ASV3_VERIFICATION.value, rows)
    assert condition_review_defects(
        audit,
        answer,
        "facts",
        ["What procedure applies?"],
        ledger,
        "main" if defect == "wrong_flow" else "audit",
        expected_citations={1, 2, 3},
    )


def test_positive_condition_needs_its_own_current_inline_original(
    condition: SourceConditionCheck,
) -> None:
    condition.disposition = "covered"
    answer = "The condition is met [1]."
    condition.answer_unit_ids = [assertion_inventory(answer)[0]["unit_id"]]
    audit = SourceConditionAuditResult(
        examined_citations=[1, 2, 3], conditions=[condition]
    )
    assert condition_omissions(audit, answer)
    answer = "The condition is met [3]."
    condition.answer_unit_ids = [assertion_inventory(answer)[0]["unit_id"]]
    assert not condition_omissions(audit, answer)


def test_literal_scenario_exclusion_does_not_force_unrelated_research(
    condition: SourceConditionCheck,
) -> None:
    ledger, _context = original_ledger()
    condition.disposition = "not_applicable"
    condition.scenario_quotes = ["The operation is domestic."]
    audit = SourceConditionAuditResult(
        examined_citations=[1, 2, 3], conditions=[condition]
    )
    ledger.record_delivery(
        "audit",
        LLMFlow.ASV3_CONDITION_REVIEW.value,
        json.loads(
            _evidence_record(ledger, "Rule [1].", include_supplemental_originals=True)
        ),
    )
    assert not condition_review_defects(
        audit,
        "Rule [1].",
        "The operation is domestic.",
        ["What procedure applies?"],
        ledger,
        "audit",
    )
    assert not condition_omissions(audit, "Rule [1].")


def test_unconditional_answer_uses_valid_exact_draft_receipt_without_inventing_rules() -> (
    None
):
    ledger, context = original_ledger()
    answer = "Definition [1]."
    evidence = _evidence_record(ledger, answer, include_supplemental_originals=True)
    llm = scripted_model()
    llm.invoke.return_value = text_response(
        {"examined_citations": [1, 2, 3], "conditions": []}
    )
    review = complete_condition_review(
        supported([1]),
        ResearchModel(llm, context),
        ledger,
        answer=answer,
        scenario="facts",
        questions=["What is the definition?"],
        evidence=evidence,
        language="en",
        consume_budget=True,
    )
    assert (
        publication_gap(
            answer,
            review,
            ["What is the definition?"],
            ledger,
            scenario="facts",
            require_condition_review=True,
        )
        is None
    )
    restored = type(review).model_validate_json(review.model_dump_json())
    assert restored.condition_review_answer_hash == answer_hash(answer)
    assert (
        publication_gap(
            answer + " Changed wording.",
            restored,
            ["What is the definition?"],
            ledger,
            scenario="facts",
            require_condition_review=True,
        )
        is not None
    )
    assert (
        publication_gap(
            answer,
            supported([1]),
            ["What is the definition?"],
            ledger,
            require_condition_review=True,
        )
        is not None
    )


def test_audit_payload_does_not_duplicate_draft_history_or_positive_review() -> None:
    ledger, _context = original_ledger()
    answer = "The operation is permitted [1]."
    payload, numbers = condition_payload(
        answer,
        "facts",
        ["What procedure applies?"],
        _evidence_record(
            ledger,
            answer,
            include_witness_spans=True,
            include_supplemental_originals=True,
        ),
        "en",
    )
    assert numbers == {1, 2, 3}
    assert payload["required_evidence_numbers"] == [1, 2, 3]
    assert (
        not {"claim", "draft", "review", "tool_history", "research_state"}
        & payload.keys()
    )
    assert json.dumps(payload).count(answer) == 1
    originals = payload["original_evidence"]
    assert isinstance(originals, list)
    for row in originals:
        assert isinstance(row, dict)
        assert isinstance(row["citation"], int)
        original = ledger.get(row["citation"])
        assert (
            original is not None
            and row["text"] == original.text
            and row["text_hash"] == original.text_hash
        )


def test_invalid_audit_repairs_once_with_same_originals(
    condition: SourceConditionCheck,
) -> None:
    ledger, context = original_ledger()
    answer = "The operation is permitted [1]."
    evidence = _evidence_record(ledger, answer, include_supplemental_originals=True)
    llm = scripted_model()
    llm.invoke.side_effect = [
        text_response({"examined_citations": [1], "conditions": []}),
        text_response(
            {
                "examined_citations": [1, 2, 3],
                "conditions": [condition.model_dump(mode="json")],
            }
        ),
    ]
    review = complete_condition_review(
        supported([1]),
        ResearchModel(llm, context),
        ledger,
        answer=answer,
        scenario="facts",
        questions=["What procedure applies?"],
        evidence=evidence,
        language="en",
        consume_budget=True,
    )
    assert not review.safe_to_publish and review.format_error is None
    assert llm.invoke.call_count == 2
    assert review.condition_review_call_id is not None
    assert ledger.completely_delivered(review.condition_review_call_id) == {1, 2, 3}


def test_negative_main_review_is_preserved_without_another_approval_call() -> None:
    ledger, context = original_ledger()
    review = supported([1])
    review.safe_to_publish = False
    review.status = "incomplete"
    review.missing_conditions = ["Missing actual exception"]
    llm = scripted_model()
    result = complete_condition_review(
        review,
        ResearchModel(llm, context),
        ledger,
        answer="Rule [1].",
        scenario="facts",
        questions=["What procedure applies?"],
        evidence=_evidence_record(ledger, "Rule [1]."),
        language="en",
        consume_budget=True,
    )
    assert result.missing_conditions == review.missing_conditions
    assert not result.safe_to_publish and llm.invoke.call_count == 0


def test_uncertain_applicability_is_not_reported_as_a_proved_omission(
    condition: SourceConditionCheck,
) -> None:
    ledger, context = original_ledger()
    condition.disposition = "uncertain"
    llm = scripted_model()
    llm.invoke.return_value = text_response(
        {
            "examined_citations": [1, 2, 3],
            "conditions": [condition.model_dump(mode="json")],
        }
    )
    review = complete_condition_review(
        supported([1]),
        ResearchModel(llm, context),
        ledger,
        answer="Rule [1].",
        scenario="facts",
        questions=["What procedure applies?"],
        evidence=_evidence_record(
            ledger, "Rule [1].", include_supplemental_originals=True
        ),
        language="en",
        consume_budget=True,
    )
    assert not review.safe_to_publish and review.missing_conditions
    assert not review.omitted_material_source_details
    gap = publication_gap(
        "Rule [1].",
        review,
        ["What procedure applies?"],
        ledger,
        require_condition_review=True,
    )
    assert gap is not None
    gaps = gap.data["source_condition_gaps"]
    assert isinstance(gaps, list) and isinstance(gaps[0], dict)
    assert gaps[0]["disposition"] == "uncertain"


def test_repeated_invalid_audit_cannot_publish_or_discard_main_assessment() -> None:
    ledger, context = original_ledger()
    llm = scripted_model()
    llm.invoke.return_value = text_response(
        {"examined_citations": [999], "conditions": []}
    )
    main = supported([1])
    review = complete_condition_review(
        main,
        ResearchModel(llm, context),
        ledger,
        answer="Rule [1].",
        scenario="facts",
        questions=["What procedure applies?"],
        evidence=_evidence_record(
            ledger, "Rule [1].", include_supplemental_originals=True
        ),
        language="en",
        consume_budget=True,
    )
    assert llm.invoke.call_count == 2
    assert (
        not review.safe_to_publish
        and review.status == "uncertain"
        and review.format_error
    )
    assert (
        review.question_results == main.question_results
        and review.evidence_numbers == [1]
    )


def test_introductory_clause_is_assessed_with_its_first_cited_list_item() -> None:
    answer = "Under the applicable provisions:\n\n- Release requires a signed certificate [1].\n- A fee is also due [2]."
    units = assertion_inventory(answer)
    assert len(units) == 2
    assert (
        units[0]["text"]
        == "Under the applicable provisions:\n\n- Release requires a signed certificate [1]."
    )
    assert units[0]["evidence_numbers"] == [1] and not units[0]["presentation_only"]
    assert units[1]["text"] == "- A fee is also due [2]."
    check = AssertionVerification(
        unit_id=units[0]["unit_id"], status="supported", basis="presentation"
    )
    assert (
        assertion_support_defect(
            units[0], check, {1: "Release requires a signed certificate."}, "facts"
        )
        is not None
    )


def test_cited_introductory_claim_is_not_merged_or_hidden_as_presentation() -> None:
    units = assertion_inventory(
        "A separate legal claim [1]:\n\n- Release requires a certificate [2]."
    )
    assert len(units) == 2
    assert units[0]["evidence_numbers"] == [1] and units[1]["evidence_numbers"] == [2]
