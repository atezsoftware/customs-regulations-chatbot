import copy
import json

import pytest
from jsonschema import Draft202012Validator
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import CapabilityCall, HarnessView, OutcomeStatus, RunContext
from onyx.asv3.outcome_map import OutcomeMap, OutcomeUpdate
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.research_state import ResearchState, build_research_specs
from tests.unit.onyx.asv3.test_native_model_adapter import (
    last_payload,
    model,
    native_response,
)

TASK = "Which permission applies? When does it start?"


def setup() -> tuple[RunContext, ResearchState, OutcomeMap, CapabilityRegistry]:
    context = RunContext(
        depth=1,
        scope={"corpus": "captured-authorized-scope"},
        services={
            "lean_native_mode": True,
            "research_profile": "experimental",
            "experimental_parallel": True,
            "independent_question": True,
            "task_id": "owned-procedure-task",
            "task_outcome_ids": ["procedure"],
            "scenario_request": "The complete fixed scenario and all three questions.",
        },
        timeout_seconds=float("inf"),
    )
    ledger = EvidenceLedger()
    state = ResearchState([TASK], context)
    outcomes = OutcomeMap(["First outcome?", TASK, "Third outcome?"], context)
    outcomes.update(
        OutcomeUpdate.model_validate({"outcomes": [parent_outcome()]}), ledger
    )
    context.services.update(evidence=ledger, research_state=state, outcome_map=outcomes)
    return (
        context,
        state,
        outcomes,
        CapabilityRegistry(build_research_specs(state, ledger)),
    )


def parent_outcome() -> dict[str, JsonValue]:
    return {
        "outcome_id": "procedure",
        "question_ids": ["q1"],
        "detail": "The separately assigned permission and start event.",
    }


def update(question: str = "q0", determination: str = "q0:d1") -> dict[str, JsonValue]:
    return {
        "needs": [
            {
                "need_id": "start_event",
                "question_ids": [question],
                "determination_ids": [determination],
                "purpose": "Establish the event that starts the applicable procedure.",
                "completion_test": "Read the operative source defining the start event.",
            }
        ],
        "active_need_ids": ["start_event"],
    }


def parameters(tools: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    for definition in tools:
        function = definition["function"]
        assert isinstance(function, dict)
        if function["name"] == "update_research":
            result = function["parameters"]
            assert isinstance(result, dict)
            return result
    raise AssertionError("Missing update_research")


def need_properties(schema: dict[str, JsonValue]) -> dict[str, JsonValue]:
    definitions = schema.get("$defs")
    if isinstance(definitions, dict):
        need = definitions["ResearchNeed"]
    else:
        properties = schema["properties"]
        assert isinstance(properties, dict)
        needs = properties["needs"]
        assert isinstance(needs, dict)
        need = needs["items"]
    assert isinstance(need, dict)
    result = need["properties"]
    assert isinstance(result, dict)
    return result


@pytest.mark.parametrize(
    "profile,parallel,independent,depth,bound",
    [
        ("experimental", True, True, 1, True),
        ("experimental", False, True, 1, False),
        ("normal", True, True, 1, False),
        ("deep", True, True, 1, False),
        ("experimental", True, False, 1, False),
        ("experimental", True, True, 0, False),
    ],
)
def test_first_native_decision_has_matching_local_schema_and_visible_bindings(
    profile: str, parallel: bool, independent: bool, depth: int, bound: bool
) -> None:
    context, state, outcomes, registry = setup()
    context.depth = depth
    context.services.update(
        research_profile=profile,
        experimental_parallel=parallel,
        independent_question=independent,
    )
    spec = registry.get("update_research")
    assert spec is not None
    original_schema = copy.deepcopy(spec.parameters)
    before_outcomes = outcomes.export()
    selected, secondary = model(), model()
    arguments = {**update(), "_outcomes": [parent_outcome()]}
    selected.invoke.return_value = native_response(
        "update_research", json.dumps(arguments)
    )
    adapter = ResearchModel(
        selected,
        context,
        research_llm=secondary if profile == "experimental" else None,
        lean_native_mode=True,
    )
    tools = registry.definitions(context)
    decision = adapter.decide(
        HarnessView(
            request=TASK,
            questions=[TASK],
            facts=[],
            receipts=[],
            evidence=[],
            tools=tools,
        )
    )
    payload = last_payload(selected)
    if bound:
        bindings = payload["research_bindings"]
        assert bindings["namespace"] == "local_update_research"
        assert bindings["question_ids"] == ["q0"]
        assert bindings["determinations"] == [
            {
                "determination_id": "q0:d0",
                "question_id": "q0",
                "question": "Which permission applies?",
            },
            {
                "determination_id": "q0:d1",
                "question_id": "q0",
                "question": "When does it start?",
            },
        ]
        assert "immutable parent" in bindings["notice"]
        assert payload["outcome_map"]["outcomes"][0]["question_ids"] == ["q1"]
        actual_tools = selected.invoke.call_args.kwargs["tools"]
        validator = Draft202012Validator(parameters(actual_tools))
        assert validator.is_valid(arguments)
        assert not validator.is_valid(update(question="q1"))
        assert not validator.is_valid(update(determination="q1:d1"))
        assert not validator.is_valid(update(determination="q0:d2"))
    else:
        assert "research_bindings" not in payload
        assert need_properties(parameters(tools)) == need_properties(original_schema)
    assert selected.invoke.call_count == 1
    secondary.invoke.assert_not_called()
    assert decision.calls[0].argument_error is None
    receipt = registry.dispatch(decision.calls[0], context)
    assert receipt.status == OutcomeStatus.FOUND
    assert state.question_ids("start_event") == ["q0"]
    assert state.revision == 1
    assert outcomes.export() == before_outcomes
    assert spec.parameters == original_schema


@pytest.mark.parametrize(
    "question,determination,summary",
    [
        ("q1", "q0:d1", "Research needs must bind to original question IDs"),
        (
            "q0",
            "q1:d1",
            "Research determinations must belong to their original question IDs",
        ),
        (
            "q0",
            "q0:d2",
            "Research determinations must belong to their original question IDs",
        ),
    ],
)
def test_invalid_caller_bindings_remain_rejected_without_remapping_or_dropping(
    question: str, determination: str, summary: str
) -> None:
    context, state, outcomes, registry = setup()
    before_state, before_outcomes = state.export(), outcomes.export()
    arguments = update(question, determination)
    response = native_response("update_research", json.dumps(arguments))
    decision = ResearchModel._decision(
        response, registry.definitions(context), return_argument_errors=True
    )
    assert "(enum)" in (decision.calls[0].argument_error or "")
    receipt = registry.dispatch(
        CapabilityCall(name="update_research", arguments=arguments), context
    )
    assert receipt.status == OutcomeStatus.INVALID
    assert receipt.summary == summary
    assert state.export() == before_state
    assert outcomes.export() == before_outcomes


def test_existing_child_checkpoint_keeps_its_exact_local_namespace() -> None:
    context, state, _, registry = setup()
    assert (
        registry.dispatch(
            CapabilityCall(name="update_research", arguments=update()), context
        ).status
        == OutcomeStatus.FOUND
    )
    ledger = context.services["evidence"]
    assert isinstance(ledger, EvidenceLedger)
    saved = state.export()
    restored = ResearchState([TASK], context)
    restored.restore(saved, ledger)
    assert restored.export() == saved
    wrong_questions = {**saved, "questions": ["A different assignment?"]}
    with pytest.raises(ValueError, match="identity or request mismatch"):
        restored.restore(wrong_questions, ledger)
    wrong_ids = copy.deepcopy(saved)
    needs = wrong_ids["needs"]
    assert isinstance(needs, list) and isinstance(needs[0], dict)
    needs[0]["question_ids"] = ["q1"]
    with pytest.raises(ValueError, match="original question IDs"):
        restored.restore(wrong_ids, ledger)
