"""Verify judicial section continuation and exact retained draft transitions."""

import copy
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.judicial_sections import (
    judicial_disposition_missing,
    judicial_witness_section,
)
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import CapabilityCall, EvidenceItem, OutcomeStatus, RunContext
from onyx.asv3.retained_answer import bind_retained_answer
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from tests.unit.onyx.asv3.test_experimental_workflow import terminal_registry
from tests.unit.onyx.asv3.test_legal_source_reviews import deliver, review, seen
from tests.unit.onyx.asv3.test_native_model_adapter import (
    adaptive_tool_view,
    last_payload,
    model,
    native_action,
)
from tests.unit.onyx.asv3.test_shared_originals import full_record
from tests.unit.onyx.asv3.test_tuned_focused_reads import judicial_chunk
from tests.unit.onyx.asv3.test_tuned_source_followthrough import tuned_context


def reasoning_originals() -> list[EvidenceItem]:
    first = judicial_chunk("III. ESASIN İNCELENMESİ\nThe court examines the rule.", 0)
    second = judicial_chunk("23. The rule requires a statutory basis.", 1)
    second.metadata["heading_path"] = [
        "Court decision",
        "23. The rule requires a statutory basis.",
    ]
    heading = judicial_chunk("V. HÜKÜM", 2)
    return [first, second, heading]


def test_numbered_reasoning_and_descriptive_metadata_do_not_supply_disposition() -> (
    None
):
    originals = reasoning_originals()
    assert (
        judicial_witness_section(
            originals[1], 0, len(originals[1].text), source_context=originals
        )
        == "reasoning"
    )
    assert judicial_disposition_missing(originals)
    originals.append(
        judicial_chunk(
            "The challenged phrase is annulled with the following limits.", 3
        )
    )
    assert not judicial_disposition_missing(originals)


def test_lettered_party_arguments_return_to_parent_merits_section() -> None:
    originals = [
        judicial_chunk(
            "III. MERITS\nA. Scope of the rule\nThe court examines scope.", 0
        ),
        judicial_chunk(
            "B. Grounds of the application\nThe applicant requests annulment.", 1
        ),
        judicial_chunk(
            "C. Constitutional assessment\n23. The court examines legality.", 2
        ),
    ]
    assert (
        judicial_witness_section(
            originals[1],
            originals[1].text.index("The applicant"),
            len(originals[1].text),
            source_context=originals,
        )
        == "argument_only"
    )
    assert (
        judicial_witness_section(
            originals[2],
            originals[2].text.index("23."),
            len(originals[2].text),
            source_context=originals,
        )
        == "reasoning"
    )
    assert judicial_disposition_missing(originals)


@pytest.mark.parametrize("broken", ["gap", "date", "derived", "source"])
def test_disposition_heading_cannot_leak_across_unverified_context(broken: str) -> None:
    originals = reasoning_originals()
    current = judicial_chunk("The challenged phrase is annulled.", 3)
    if broken == "gap":
        current.metadata["position"] = 4
    elif broken == "date":
        current.metadata["read_as_of_date"] = "2025-01-02"
    elif broken == "derived":
        originals[-1].metadata["derived"] = True
    else:
        originals[-1].source_id = "foreign-decision"
    assert (
        judicial_witness_section(
            current, 0, len(current.text), source_context=originals
        )
        == "unknown"
    )


def test_reasoning_review_stays_open_until_candidate_disposition_body_is_delivered() -> (
    None
):
    context, ledger, reviews = tuned_context()
    seen(context, ledger, reviews)
    numbers = ledger.add(reasoning_originals(), context)
    deliver(ledger, "answer-call", [1, *numbers])
    context.services["last_model_call_id"] = "answer-call"
    original = ledger.get(numbers[1])
    assert original is not None
    witness = {
        "citation": numbers[1],
        "start_char": 0,
        "end_char": len(original.text),
    }
    call = CapabilityCall(
        name="submit_answer",
        arguments={
            "answer": f"The court's outcome changes the rule [{numbers[1]}].",
            "basis": "originals",
            "_related_source_reviews": [review(witnesses=[witness])],
        },
    )
    rejected = terminal_registry([]).dispatch(call, context)
    assert rejected.status == OutcomeStatus.INVALID
    diagnostic = rejected.data["related_source_review_error"]
    assert isinstance(diagnostic, dict)
    assert diagnostic["code"] == "missing_judicial_disposition"
    assert reviews.view(context, ledger, {1, *numbers})["pending_lead_ids"]


def test_pending_reasoning_uses_exact_acquisition_cursor_instead_of_terminal_retry() -> (
    None
):
    context, ledger, reviews = tuned_context()
    seen(context, ledger, reviews)
    numbers = ledger.add(reasoning_originals(), context)
    state = reviews.view(context, ledger, {1, *numbers})
    rows = cast(list[dict[str, JsonValue]], state["reviews"])
    rows[0]["source_range_read"] = {"has_more": True, "next_position": 3}
    actions = ResearchModel._related_source_acquisition_state(state, ledger)
    assert actions == [
        {
            "lead_id": rows[0]["lead_id"],
            "source_id": "decision",
            "reason": "reasoning_without_disposition_body",
            "suggested_acquisition": {
                "name": "read_source_range",
                "arguments": {"source_id": "decision", "start": 3},
            },
        }
    ]
    number = ledger.add([judicial_chunk("The request is granted.", 3)], context)[0]
    rows[0]["available_original_citations"] = [*numbers, number]
    assert ResearchModel._related_source_acquisition_state(state, ledger) == []


def test_saved_reasoning_assessment_does_not_bypass_tuned_disposition_check() -> None:
    context, ledger, reviews = tuned_context()
    seen(context, ledger, reviews)
    numbers = ledger.add(reasoning_originals(), context)
    original = ledger.get(numbers[1])
    assert original is not None
    deliver(ledger, "prior-call", [1, *numbers])
    context.services["last_model_call_id"] = "prior-call"
    context.services.pop("asv3_workflow_variant")
    context.services["research_profile"] = "experimental"
    assessment = review(
        witnesses=[
            {
                "citation": numbers[1],
                "start_char": 0,
                "end_char": len(original.text),
            }
        ]
    )
    outcome = terminal_registry([]).dispatch(
        CapabilityCall(
            name="submit_answer",
            arguments={
                "answer": f"Rule [{numbers[1]}].",
                "basis": "originals",
                "_related_source_reviews": [assessment],
            },
        ),
        context,
    )
    assert outcome.status == OutcomeStatus.FOUND
    context.services["asv3_workflow_variant"] = ASV3_TUNED_VARIANT
    context.services["research_profile"] = "normal"
    state = reviews.view(context, ledger, {1, *numbers})
    assert state["pending_lead_ids"]
    rows = cast(list[dict[str, JsonValue]], state["reviews"])
    assert rows[0]["status"] == "pending" and rows[0]["review"] is None
    assert (
        reviews.publication_gap(f"Rule [{numbers[1]}].", "prior-call", context, ledger)
        is not None
    )


def test_tuned_native_commit_keeps_whole_draft_and_fresh_source_delivery() -> None:
    context, ledger, _reviews = tuned_context()
    draft = "First supported condition [1].\n\nSecond supported exception [1].\n"
    selected = model()
    selected.invoke.return_value = native_action("submit_retained_answer", {})
    current = adaptive_tool_view(original_evidence=[full_record(ledger, 1)]).model_copy(
        update={
            "tools": terminal_registry([]).definitions(context),
            "draft_to_repair": draft,
            "publication_gap": {"summary": "Correct publication metadata."},
        }
    )
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    decision = adapter.decide(current)
    payload = last_payload(selected)
    assert "retained_answer" in payload
    assert decision.calls[0].name == "submit_answer"
    assert decision.calls[0].arguments["answer"] == draft
    assert decision.calls[0].arguments["basis"] == "originals"
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1}
    assert selected.invoke.call_count == 1


def test_protected_normal_and_plain_experimental_do_not_expose_retained_commits() -> (
    None
):
    definitions = terminal_registry([]).definitions(RunContext())
    for profile in ("normal", "experimental"):
        context = RunContext(
            services={"research_profile": profile, "experimental_parallel": False}
        )
        bound, reference = bind_retained_answer(
            copy.deepcopy(definitions), context, "Some draft [1].", request="Question"
        )
        assert reference is None
        assert bound == definitions
