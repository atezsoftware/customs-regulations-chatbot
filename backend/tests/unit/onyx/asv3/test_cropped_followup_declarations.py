"""Same-terminal declarations are staged without approving or rewriting the answer."""

import copy
import hashlib
import json
from pathlib import Path
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.cropped_completion import (
    CroppedValidatedFollowups,
    append_cropped_required_followups,
    build_cropped_completion_context,
)
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    CapabilityCall,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.outcome_map import OutcomeMap, OutcomeUpdate
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.research_state import ResearchState, ResearchUpdate

FIXTURE = Path(__file__).parent / "fixtures/cropped_followup_same_terminal_5232.json"


def fixture() -> tuple[
    RunContext,
    EvidenceLedger,
    ResearchState,
    OutcomeMap,
    CapabilityRegistry,
    CapabilityCall,
    list[str],
    dict[str, JsonValue],
]:
    captured = cast(dict[str, JsonValue], json.loads(FIXTURE.read_text()))
    questions = cast(list[str], captured["questions"])
    request = cast(str, captured["request"])
    context = RunContext(services={"asv3_exp_cropped": True, "lean_native_mode": True})
    ledger = EvidenceLedger()
    research = ResearchState(questions, context)
    research.update(ResearchUpdate.model_validate({"needs": captured["needs"]}), ledger)
    outcomes = OutcomeMap(questions, context)
    context.services.update(
        evidence=ledger,
        research_state=research,
        outcome_map=outcomes,
        last_model_call_id="unchanged-source-writer",
    )
    context.services["cropped_completion_context"] = build_cropped_completion_context(
        context, request
    )
    checked: list[str] = []

    def publication_guard(args: dict[str, JsonValue], child: RunContext) -> ToolOutcome:
        # A metadata fix must still reach (and obey) the ordinary body guard.
        checked.append(cast(str, args["answer"]))
        assert child.services["last_model_call_id"] == "unchanged-source-writer"
        return ToolOutcome(status=OutcomeStatus.DENIED, summary="Unapproved legal body")

    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="submit_partial_answer",
                description="Guarded publication",
                parameters={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
                handler=publication_guard,
            )
        ]
    )
    call = CapabilityCall(
        name="submit_partial_answer",
        arguments=cast(dict[str, JsonValue], captured["arguments"]),
    )
    return context, ledger, research, outcomes, registry, call, checked, captured


def test_actual_5232_declarations_bind_both_followups_without_state_mutation() -> None:
    context, ledger, research, outcomes, registry, call, checked, captured = fixture()
    before = (ledger.export(), research.export(), outcomes.export())
    original_call = call.model_dump(mode="json")
    answer = cast(str, call.arguments["answer"])
    provenance = cast(dict[str, JsonValue], captured["provenance"])
    assert hashlib.sha256(answer.encode()).hexdigest() == provenance["answer_sha256"]
    assert outcomes.outcome_ids() == []
    for _ in range(2):
        assert registry.research_binding_gap(call, context) is None
        assert (ledger.export(), research.export(), outcomes.export()) == before
    assert not checked
    assert registry.outcome_metadata_gap(call, context) is None
    assert outcomes.outcome_ids() == ["rail", "tir", "adr", "liability", "tax"]
    assert outcomes.revision == 1
    child = context.child()
    child.depth = context.depth
    child.services["applied_outcome_metadata_call"] = call
    assert registry.dispatch(call, child).status == OutcomeStatus.DENIED
    assert checked == [answer]
    validated = child.services["cropped_validated_required_followups"]
    assert isinstance(validated, CroppedValidatedFollowups)
    assert [item.need_id for item in validated.items] == ["adr", "loss"]
    assert [item.outcome_id for item in validated.items] == ["adr", "liability"]
    assert outcomes.revision == 1
    assert call.model_dump(mode="json") == original_call
    assert context.services["last_model_call_id"] == "unchanged-source-writer"
    assert ledger.export() == before[0]
    # Render semantics only: the denied guard above did not publish this text.
    rendered = append_cropped_required_followups(
        answer.strip(), validated, context, cast(str, captured["request"])
    )
    assert rendered.startswith(answer.strip() + "\n\n")
    raw = cast(list[dict[str, JsonValue]], call.arguments["_required_followups"])
    assert all(rendered.count(cast(str, row["question"])) == 1 for row in raw)


@pytest.mark.parametrize(
    "fault",
    [
        "unknown",
        "question",
        "duplicate",
        "coverage",
        "literal_fact",
        "conflicting_declarations",
    ],
)
def test_invalid_staged_metadata_never_mutates_live_state(fault: str) -> None:
    context, ledger, research, outcomes, registry, call, checked, _ = fixture()
    rows = cast(list[dict[str, JsonValue]], call.arguments["_outcomes"])
    followups = cast(list[dict[str, JsonValue]], call.arguments["_required_followups"])
    if fault == "unknown":
        followups[0]["outcome_id"] = "invented"
    elif fault == "question":
        rows[2]["question_ids"] = ["q99"]
    elif fault == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif fault == "literal_fact":
        rows[2]["decisive_facts"] = ["This fact was not supplied."]
    elif fault == "conflicting_declarations":
        call.arguments["_coverage"] = {"outcomes": []}
    else:
        call.arguments["_coverage"] = {
            "resolutions": [
                {"outcome_id": "adr", "status": "supported", "evidence_numbers": [999]}
            ]
        }
    before = (
        ledger.export(),
        research.export(),
        outcomes.export(),
        context.budget.snapshot(),
    )
    gap = registry.dispatch(call, context)
    assert gap.status == OutcomeStatus.INVALID
    assert (
        gap.data[
            "invalid_required_followups"
            if fault == "unknown"
            else "invalid_outcome_metadata"
        ]
        is True
    )
    if fault != "unknown":
        ordinary = call.model_copy(deep=True)
        ordinary.arguments.pop("_required_followups")
        assert registry.outcome_metadata_gap(ordinary, context) == gap
    assert not checked
    assert (
        ledger.export(),
        research.export(),
        outcomes.export(),
        context.budget.snapshot(),
    ) == before


@pytest.mark.parametrize("fault", ["owner", "scope", "research", "body", "declaration"])
def test_staged_receipt_does_not_transfer_to_changed_execution(fault: str) -> None:
    context, ledger, research, outcomes, registry, call, checked, _ = fixture()
    assert registry.research_binding_gap(call, context) is None
    if fault == "owner":
        context.services["task_id"] = "foreign"
    elif fault == "scope":
        context.scope["foreign"] = True
    elif fault == "research":
        research.update(ResearchUpdate(active_need_ids=["adr"]), ledger)
    elif fault == "body":
        call.arguments["answer"] = "A different body [999]."
        # Different call bytes have a different cache key; new-citation followups fail.
        rows = cast(list[dict[str, JsonValue]], call.arguments["_required_followups"])
        rows[0]["question"] = "Should source [181] be examined?"
    else:
        rows = cast(list[dict[str, JsonValue]], call.arguments["_outcomes"])
        rows[2]["question_ids"] = ["q99"]
    before = outcomes.export()
    assert registry.dispatch(call, context).status == OutcomeStatus.INVALID
    assert outcomes.export() == before
    assert not checked


def test_append_rejects_changed_meaning_of_the_same_declared_outcome() -> None:
    context, ledger, _, outcomes, registry, call, _, captured = fixture()
    assert registry.outcome_metadata_gap(call, context) is None
    child = context.child()
    child.depth = context.depth
    child.services["applied_outcome_metadata_call"] = call
    registry.dispatch(call, child)
    validated = child.services["cropped_validated_required_followups"]
    assert isinstance(validated, CroppedValidatedFollowups)
    rows = copy.deepcopy(cast(list[dict[str, JsonValue]], call.arguments["_outcomes"]))
    rows[2]["detail"] = "An unrelated replacement result."
    outcomes.update(OutcomeUpdate.model_validate({"outcomes": rows}), ledger)
    with pytest.raises(ValueError, match="meaning changed"):
        append_cropped_required_followups(
            cast(str, call.arguments["answer"]).strip(),
            validated,
            context,
            cast(str, captured["request"]),
        )


def test_valid_new_outcome_cannot_bind_a_need_from_another_question() -> None:
    from tests.unit.onyx.asv3.test_cropped_completion import (
        ANSWER,
        item,
        terminal_registry,
    )
    from tests.unit.onyx.asv3.test_cropped_completion import (
        fixture as existing_fixture,
    )

    context, ledger, research, outcomes = existing_fixture()
    checked: list[str] = []
    registry = terminal_registry(context, checked)
    call = CapabilityCall(
        name="submit_answer",
        arguments={
            "answer": ANSWER,
            "_outcomes": [
                {
                    "outcome_id": "new_other",
                    "question_ids": ["q1"],
                    "detail": "Appeal result",
                }
            ],
            "_required_followups": [item(outcome_id="new_other")],
        },
    )
    before = (ledger.export(), research.export(), outcomes.export())
    result = registry.dispatch(call, context)
    assert result.status == OutcomeStatus.INVALID
    assert (
        result.data["detail"]
        == "Followup need and outcome belong to unrelated questions"
    )
    assert (ledger.export(), research.export(), outcomes.export()) == before
    assert not checked
