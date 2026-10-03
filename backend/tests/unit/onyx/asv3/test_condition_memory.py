"""Material obligations survive model omission, rewrites and scoped checkpoint restore."""

import json

import pytest

from onyx.asv3.assertions import AssertionWitness, assertion_inventory
from onyx.asv3.condition_memory import SourceConditionMemory
from onyx.asv3.llm_adapter import (
    ResearchModel,
    SourceConditionAuditResult,
    SourceConditionCheck,
    SourceConditionResolution,
)
from onyx.asv3.models import RunStopped
from onyx.asv3.publication import publication_gap
from onyx.asv3.research_state import ResearchState, ResearchUpdate
from onyx.asv3.runtime import _evidence_record  # pyright: ignore[reportPrivateUsage]
from onyx.asv3.source_conditions import answer_hash, complete_condition_review
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.asv3.test_citation_contract import original_ledger, supported
from tests.unit.onyx.asv3.test_model_adapter import scripted_model, text_response

pytestmark = pytest.mark.usefixtures("empty_source_inventory")


@pytest.fixture
def requirement() -> SourceConditionCheck:
    return SourceConditionCheck(
        witness=AssertionWitness(citation=3, source_quote="condition AND exception"),
        determination_ids=["q0:d0"],
        detail="The independent exception must also be considered.",
        applicability="It qualifies the requested operation.",
        disposition="omitted",
    )


def test_same_original_can_have_two_independent_obligations(
    requirement: SourceConditionCheck,
) -> None:
    ledger, context = original_ledger()
    state = ResearchState(["What applies?"], context)
    other = requirement.model_copy(
        update={"detail": "The procedural proof is also required."}
    )
    state.source_conditions.remember([requirement, other], ledger)
    assert len(state.source_conditions.required_conditions()) == 2
    assessed, defects = state.source_conditions.materialize(
        SourceConditionAuditResult(
            examined_citations=[1, 2, 3], conditions=[requirement]
        ),
        ledger,
    )
    assert defects and len(assessed.conditions) == 1
    assert state.preferred_citations() == [3]


def test_requirements_stay_visible_before_background_findings_under_context_pressure(
    requirement: SourceConditionCheck,
) -> None:
    ledger, context = original_ledger()
    state = ResearchState(["What applies?"], context)
    state.source_conditions.remember([requirement], ledger)
    state.update(
        ResearchUpdate.model_validate(
            {
                "needs": [
                    {
                        "need_id": f"background{index}",
                        "question_ids": ["q0"],
                        "purpose": "Background explanation " * 30,
                        "completion_test": "Original support for this related fact",
                    }
                    for index in range(8)
                ]
            }
        ),
        ledger,
    )
    bounded = state.view(max_chars=2000)
    conditions = bounded["source_conditions"]
    omitted_needs = bounded["needs_omitted"]
    assert isinstance(conditions, list) and len(conditions) == 1
    assert bounded["source_conditions_omitted"] == 0
    assert isinstance(omitted_needs, int) and omitted_needs > 0
    assert len(json.dumps(bounded, ensure_ascii=False)) <= 2000
    smaller = state.view(max_chars=600)
    assert smaller["source_conditions"] == []
    assert smaller["source_conditions_omitted"] == 1
    assert len(state.source_conditions.required_conditions()) == 1


def test_retained_originals_cannot_silently_fall_out_of_evidence_delivery(
    requirement: SourceConditionCheck,
) -> None:
    ledger, context = original_ledger()
    state = ResearchState(["What applies?"], context)
    state.source_conditions.remember([requirement], ledger)
    answer = "General permission [1]."
    required = state.source_conditions.citations()
    payload = ledger.serialize_records([1, 3], required=(1, 3))
    records = json.loads(
        _evidence_record(
            ledger,
            answer,
            required_numbers=required,
            preferred_numbers=[2],
            include_supplemental_originals=True,
            max_chars=len(payload) + 10,
        )
    )
    assert [record["citation"] for record in records] == [1, 3]
    for record in records:
        original = ledger.get(record["citation"])
        assert original is not None
        assert not record["truncated"] and record["text"] == original.text
    with pytest.raises(RunStopped, match="Complete cited evidence"):
        _evidence_record(ledger, answer, required_numbers=required, max_chars=100)
    llm = scripted_model()
    result = complete_condition_review(
        supported([1]),
        ResearchModel(llm, context),
        ledger,
        answer=answer,
        scenario="facts",
        questions=["What applies?"],
        evidence=_evidence_record(ledger, answer),
        language="en",
        consume_budget=True,
        condition_memory=state.source_conditions,
    )
    assert not result.safe_to_publish and result.status == "incomplete"
    assert result.missing_conditions
    llm.invoke.assert_not_called()


@pytest.mark.parametrize("provider", ["vertex_ai", "anthropic", "openai"])
def test_empty_next_audit_cannot_close_known_material_requirement(
    provider: str,
    requirement: SourceConditionCheck,
) -> None:
    ledger, context = original_ledger()
    state = ResearchState(["What applies?"], context)
    state.source_conditions.remember([requirement], ledger)
    llm = scripted_model()
    llm.config.model_provider = provider
    llm.invoke.return_value = text_response(
        {"examined_citations": [1, 2, 3], "conditions": []}
    )
    result = complete_condition_review(
        supported([1]),
        ResearchModel(llm, context),
        ledger,
        answer="General permission [1].",
        scenario="facts",
        questions=["What applies?"],
        evidence=_evidence_record(
            ledger, "General permission [1].", include_supplemental_originals=True
        ),
        language="en",
        consume_budget=True,
        condition_memory=state.source_conditions,
    )
    assert not result.safe_to_publish and result.format_error
    assert llm.invoke.call_count == 2
    assert len(state.source_conditions.required_conditions()) == 1


@pytest.mark.parametrize(
    "disposition", ["covered", "not_applicable", "omitted", "uncertain"]
)
def test_resolution_requires_current_answer_or_literal_exclusion(
    disposition: str,
    requirement: SourceConditionCheck,
) -> None:
    ledger, context = original_ledger()
    state = ResearchState(["What applies?"], context)
    state.source_conditions.remember([requirement], ledger)
    identity = str(state.source_conditions.required_conditions()[0]["condition_id"])
    answer = "The exception limits the operation [3]."
    resolution = SourceConditionResolution.model_validate(
        {
            "condition_id": identity,
            "disposition": disposition,
            "answer_unit_ids": [assertion_inventory(answer)[0]["unit_id"]]
            if disposition == "covered"
            else [],
            "scenario_quotes": ["The operation is domestic."]
            if disposition == "not_applicable"
            else [],
        }
    )
    audit = SourceConditionAuditResult(
        examined_citations=[1, 2, 3], conditions=[], resolutions=[resolution]
    )
    llm = scripted_model()
    llm.invoke.return_value = text_response(audit.model_dump(mode="json"))
    model = ResearchModel(llm, context)
    result = complete_condition_review(
        supported([3]),
        model,
        ledger,
        answer=answer,
        scenario="The operation is domestic.",
        questions=["What applies?"],
        evidence=_evidence_record(ledger, answer, include_supplemental_originals=True),
        language="en",
        consume_budget=True,
        condition_memory=state.source_conditions,
    )
    assert result.safe_to_publish is (disposition in {"covered", "not_applicable"})
    assert len(state.source_conditions.required_conditions()) == 1
    gap = publication_gap(
        answer,
        result,
        ["What applies?"],
        ledger,
        scenario="The operation is domestic.",
        require_condition_review=True,
        research_state=state,
    )
    assert (gap is None) is result.safe_to_publish
    assert (
        publication_gap(
            answer + " Rewritten.",
            result,
            ["What applies?"],
            ledger,
            scenario="The operation is domestic.",
            require_condition_review=True,
            research_state=state,
        )
        is not None
    )


def test_targeted_research_can_resolve_reference_with_new_operative_original(
    requirement: SourceConditionCheck,
) -> None:
    ledger, context = original_ledger()
    state = ResearchState(["What applies?"], context)
    requirement.disposition = "uncertain"
    state.source_conditions.remember([requirement], ledger)
    identity = str(state.source_conditions.required_conditions()[0]["condition_id"])
    answer = "The governing original establishes the exception [2]."
    audit = SourceConditionAuditResult(
        examined_citations=[1, 2, 3],
        conditions=[],
        resolutions=[
            SourceConditionResolution(
                condition_id=identity,
                disposition="covered",
                answer_unit_ids=[assertion_inventory(answer)[0]["unit_id"]],
                witness=AssertionWitness(
                    citation=2, source_quote="condition AND exception"
                ),
            )
        ],
    )
    llm = scripted_model()
    llm.invoke.return_value = text_response(audit.model_dump(mode="json"))
    model = ResearchModel(llm, context)
    result = complete_condition_review(
        supported([2]),
        model,
        ledger,
        answer=answer,
        scenario="facts",
        questions=["What applies?"],
        evidence=_evidence_record(ledger, answer, include_supplemental_originals=True),
        language="en",
        consume_budget=True,
        condition_memory=state.source_conditions,
    )
    assert result.safe_to_publish
    assert (
        publication_gap(
            answer,
            result,
            ["What applies?"],
            ledger,
            research_state=state,
            require_condition_review=True,
        )
        is None
    )
    assert len(state.source_conditions.required_conditions()) == 1
    assert state.source_conditions.citations() == [3]


@pytest.mark.parametrize(
    "defect",
    [
        "missing_inline",
        "stale_unit",
        "invented_exclusion",
        "foreign_source",
        "unknown_id",
    ],
)
def test_plausible_resolution_labels_cannot_supply_invalid_proof(
    defect: str,
    requirement: SourceConditionCheck,
) -> None:
    ledger, context = original_ledger()
    state = ResearchState(["What applies?"], context)
    state.source_conditions.remember([requirement], ledger)
    identity = str(state.source_conditions.required_conditions()[0]["condition_id"])
    answer = (
        "General permission [1]."
        if defect == "missing_inline"
        else "The exception applies [3]."
    )
    resolution = SourceConditionResolution(
        condition_id=identity,
        disposition="covered",
        answer_unit_ids=[assertion_inventory(answer)[0]["unit_id"]],
    )
    if defect == "stale_unit":
        resolution.answer_unit_ids = [
            assertion_inventory("Old wording [3].")[0]["unit_id"]
        ]
    elif defect == "invented_exclusion":
        resolution.disposition = "not_applicable"
        resolution.answer_unit_ids = []
        resolution.scenario_quotes = ["Invented facts"]
    elif defect == "foreign_source":
        resolution.witness = AssertionWitness(citation=999, source_quote="exception")
    elif defect == "unknown_id":
        resolution.condition_id = "sc-another-run"
    llm = scripted_model()
    llm.invoke.return_value = text_response(
        SourceConditionAuditResult(
            examined_citations=[1, 2, 3],
            conditions=[],
            resolutions=[resolution],
        ).model_dump(mode="json")
    )
    result = complete_condition_review(
        supported([3]),
        ResearchModel(llm, context),
        ledger,
        answer=answer,
        scenario="facts",
        questions=["What applies?"],
        evidence=_evidence_record(ledger, answer, include_supplemental_originals=True),
        language="en",
        consume_budget=True,
        condition_memory=state.source_conditions,
    )
    assert not result.safe_to_publish


@pytest.mark.parametrize(
    "defect", [None, "run", "scope", "questions", "hash", "duplicate"]
)
def test_checkpoint_keeps_requirements_and_rejects_foreign_identity(
    defect: str | None,
    requirement: SourceConditionCheck,
) -> None:
    ledger, context = original_ledger()
    state = ResearchState(["What applies?"], context)
    state.source_conditions.remember([requirement], ledger)
    saved = state.source_conditions.export()
    if defect in {"run", "scope"}:
        saved["run_id" if defect == "run" else "scope_hash"] = "foreign"
    elif defect == "questions":
        saved["questions"] = ["Other request"]
    elif defect in {"hash", "duplicate"}:
        rows = saved["conditions"]
        assert isinstance(rows, list) and isinstance(rows[0], dict)
        if defect == "hash":
            rows[0]["text_hash"] = "changed"
        else:
            rows.append(rows[0])
    restored = SourceConditionMemory(
        state.run_id, state.scope_hash, list(state.questions)
    )
    if defect is not None:
        with pytest.raises(ValueError):
            restored.restore(saved, ledger)
        return
    restored.restore(saved, ledger)
    assert restored.export() == saved
    assert "Complete source 3" not in json.dumps(saved)
    assert restored.materialize(
        SourceConditionAuditResult(examined_citations=[1, 2, 3], conditions=[]), ledger
    )[1]


def test_guard_rejects_injected_empty_receipt_after_checkpoint_restore(
    requirement: SourceConditionCheck,
) -> None:
    ledger, context = original_ledger()
    state = ResearchState(["What applies?"], context)
    state.source_conditions.remember([requirement], ledger)
    restored = ResearchState(["What applies?"], context)
    restored.restore(state.export(), ledger)
    answer = "General permission [1]."
    rows = json.loads(
        _evidence_record(ledger, answer, include_supplemental_originals=True)
    )
    ledger.record_delivery("empty-audit", LLMFlow.ASV3_CONDITION_REVIEW.value, rows)
    review = supported([1]).model_copy(
        update={
            "condition_review": SourceConditionAuditResult(
                examined_citations=[1, 2, 3], conditions=[]
            ),
            "condition_review_call_id": "empty-audit",
            "condition_review_answer_hash": answer_hash(answer),
        }
    )
    assert (
        publication_gap(
            answer,
            review,
            ["What applies?"],
            ledger,
            research_state=restored,
            require_condition_review=True,
        )
        is not None
    )
