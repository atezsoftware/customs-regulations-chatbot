import json
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
from jsonschema import Draft202012Validator
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.outcome_map import OutcomeMap, OutcomeUpdate
from onyx.asv3.registry import CapabilityRegistry
from onyx.llm.interfaces import LLM, LLMConfig
from onyx.llm.model_response import (
    ChatCompletionMessageToolCall,
    Choice,
    FunctionCall,
    Message,
    ModelResponse,
)


def research_context() -> tuple[RunContext, OutcomeMap, EvidenceLedger]:
    context = RunContext(
        services={"lean_native_mode": True}, timeout_seconds=float("inf")
    )
    ledger = EvidenceLedger()
    outcomes = OutcomeMap(["First outcome?", "Second outcome?"], context)
    context.services.update(evidence=ledger, outcome_map=outcomes)
    return context, outcomes, ledger


def requested(identity: str = "first", question: str = "q0") -> dict[str, JsonValue]:
    return {
        "outcome_id": identity,
        "question_ids": [question],
        "detail": "The requested concrete result",
    }


def source_registry(
    handler: Callable[[dict[str, JsonValue], RunContext], ToolOutcome],
) -> CapabilityRegistry:
    return CapabilityRegistry(
        [
            ToolSpec(
                name="read_provision",
                description="Read the requested original",
                parameters={
                    "type": "object",
                    "properties": {
                        "source_id": {"type": "string"},
                        "article": {"type": "string"},
                    },
                    "required": ["source_id", "article"],
                    "additionalProperties": False,
                },
                handler=handler,
            )
        ]
    )


def harness(
    context: RunContext, registry: CapabilityRegistry, ledger: EvidenceLedger
) -> Harness:
    return Harness(
        request="First outcome? Second outcome?",
        context=context,
        registry=registry,
        evidence=ledger,
        decide=lambda _view: Decision(answer="Complete"),
    )


def read_call(arguments: dict[str, JsonValue] | None = None) -> CapabilityCall:
    return CapabilityCall(
        name="read_provision",
        arguments={"source_id": "original", "article": "1", **(arguments or {})},
    )


def native_parameters(context: RunContext) -> dict[str, JsonValue]:
    registry = source_registry(
        lambda _arguments, _context: ToolOutcome(
            status=OutcomeStatus.FOUND, summary="Original read"
        )
    )
    definition = registry.definitions(context)[0]
    function = definition["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    return parameters


def test_native_outcome_schema_exposes_actual_required_fields_without_references() -> (
    None
):
    context, _, _ = research_context()
    parameters = native_parameters(context)
    Draft202012Validator.check_schema(parameters)
    serialized = json.dumps(parameters)
    assert '"$ref"' not in serialized
    assert '"$defs"' not in serialized
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    declarations = properties["_outcomes"]
    assert isinstance(declarations, dict)
    item = declarations["items"]
    assert isinstance(item, dict)
    assert set(item["required"]) == {"outcome_id", "question_ids", "detail"}
    assert item["additionalProperties"] is False
    required = parameters["required"]
    assert isinstance(required, list)
    assert "_outcomes" not in required
    assert "_coverage" not in required
    validator = Draft202012Validator(parameters)
    assert validator.is_valid(read_call().arguments)
    assert validator.is_valid(read_call({"_outcomes": [], "_coverage": {}}).arguments)
    assert validator.is_valid(read_call({"_outcomes": [requested()]}).arguments)
    invalid = requested()
    invalid.pop("detail")
    assert not validator.is_valid(read_call({"_outcomes": [invalid]}).arguments)


def test_native_coverage_schema_checks_source_ranges_and_resolution_contract() -> None:
    context, _, _ = research_context()
    parameters = native_parameters(context)
    validator = Draft202012Validator(parameters)
    coverage: dict[str, JsonValue] = {
        "conditions": [
            {
                "condition_id": "approval",
                "outcome_ids": ["first"],
                "detail": "The result depends on an authenticated document.",
                "witnesses": [{"citation": 1, "start_char": 0, "end_char": 60}],
            }
        ],
        "resolutions": [
            {
                "outcome_id": "first",
                "status": "conditional",
                "condition_ids": ["approval"],
                "evidence_numbers": [1],
            }
        ],
    }
    assert validator.is_valid(read_call({"_coverage": coverage}).arguments)
    assert (
        OutcomeUpdate.model_validate(coverage).conditions[0].condition_id == "approval"
    )
    encoded = json.dumps(coverage)
    for old, new in (
        ('"end_char": 60', '"end_char": 0'),
        ('"citation": 1', '"citation": 0'),
        ('"start_char": 0', '"start_char": -1'),
        ('"status": "conditional"', '"status": "complete"'),
        (
            '"detail": "The result depends on an authenticated document."',
            '"unexpected": "not a condition field"',
        ),
    ):
        changed = json.loads(encoded.replace(old, new))
        assert not validator.is_valid(read_call({"_coverage": changed}).arguments)


@pytest.mark.parametrize("service", ["outcome_map", "lean_native_mode"])
def test_outcome_metadata_schema_is_not_exposed_when_inactive(service: str) -> None:
    context, _, _ = research_context()
    context.services.pop(service)
    parameters = native_parameters(context)
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    assert "_outcomes" not in properties
    assert "_coverage" not in properties


@pytest.mark.parametrize("through_harness", [False, True])
def test_outcomes_on_first_useful_call_exist_before_source_handler_without_metadata_leaking(
    through_harness: bool,
) -> None:
    context, outcomes, ledger = research_context()
    observed: list[dict[str, JsonValue]] = []

    def handler(arguments: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        assert outcomes.outcome_ids() == ["first"]
        observed.append(arguments)
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="Original read")

    registry = source_registry(handler)
    call = read_call({"_outcomes": [requested()]})
    result = (
        harness(context, registry, ledger)._dispatch([call])[0].outcome
        if through_harness
        else registry.dispatch(call, context)
    )
    assert result.status == OutcomeStatus.FOUND
    assert observed == [{"source_id": "original", "article": "1"}]
    assert outcomes.revision == 1


def test_researcher_resolution_cannot_change_a_sibling_outcome() -> None:
    context, outcomes, ledger = research_context()
    outcomes.update(
        OutcomeUpdate.model_validate(
            {"outcomes": [requested(), requested("second", "q1")]}
        ),
        ledger,
    )
    child = context.independent_child()
    child.services["task_outcome_ids"] = ["first"]
    calls: list[dict[str, JsonValue]] = []

    def handler(arguments: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        calls.append(arguments)
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="Read")

    registry = source_registry(handler)
    before = outcomes.export()
    forbidden = read_call(
        {
            "_coverage": {
                "resolutions": [
                    {
                        "outcome_id": "second",
                        "status": "unresolved",
                        "gap": "The sibling original is unread.",
                    }
                ]
            }
        }
    )
    result = registry.dispatch(forbidden, child)
    assert result.status == OutcomeStatus.INVALID
    assert result.data["invalid_outcome_metadata"] is True
    assert outcomes.export() == before
    assert calls == []
    allowed = read_call(
        {
            "_coverage": {
                "resolutions": [
                    {
                        "outcome_id": "first",
                        "status": "unresolved",
                        "gap": "This assigned original is unread.",
                    }
                ]
            }
        }
    )
    assert registry.dispatch(allowed, child).status == OutcomeStatus.FOUND
    assert len(calls) == 1
    assert outcomes.view()["unassessed_outcome_ids"] == ["second"]


def test_cached_original_read_applies_new_source_conditions_without_another_source_call() -> (
    None
):
    context, outcomes, ledger = research_context()
    text = "Approval requires the applicant's authenticated document."
    calls = 0

    def handler(_arguments: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        nonlocal calls
        calls += 1
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Original read",
            evidence=[EvidenceItem(source_id="original", chunk_id="unit", text=text)],
        )

    runner = harness(context, source_registry(handler), ledger)
    first = runner._dispatch([read_call({"_outcomes": [requested()]})])[0]
    assert first.evidence_ids == [1]
    coverage: dict[str, JsonValue] = {
        "conditions": [
            {
                "condition_id": "proof",
                "outcome_ids": ["first"],
                "detail": "The required proof is authenticated.",
                "witnesses": [{"citation": 1, "start_char": 0, "end_char": len(text)}],
            }
        ],
        "resolutions": [
            {
                "outcome_id": "first",
                "status": "conditional",
                "condition_ids": ["proof"],
                "evidence_numbers": [1],
                "gap": "",
            }
        ],
    }
    repeated = runner._dispatch([read_call({"_coverage": coverage})])[0]
    assert repeated.outcome.data["reused_recorded_read"] is True
    assert repeated.evidence_ids == [1]
    assert calls == 1
    assert outcomes.view()["unassessed_outcome_ids"] == []
    assert outcomes.view()["resolutions"] == coverage["resolutions"]
    assert outcomes.revision == 2


def test_corrected_outcome_metadata_does_not_poison_the_same_actual_source_arguments() -> (
    None
):
    context, outcomes, ledger = research_context()
    calls = 0

    def handler(_arguments: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        nonlocal calls
        calls += 1
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="Original read")

    runner = harness(context, source_registry(handler), ledger)
    invalid = runner._dispatch(
        [read_call({"_outcomes": [requested(question="outside")]})]
    )[0]
    assert invalid.outcome.status == OutcomeStatus.INVALID
    assert invalid.outcome.data["invalid_outcome_metadata"] is True
    assert calls == 0
    assert outcomes.outcome_ids() == []
    corrected = runner._dispatch([read_call({"_outcomes": [requested()]})])[0]
    assert corrected.outcome.status == OutcomeStatus.FOUND
    assert calls == 1
    assert outcomes.outcome_ids() == ["first"]


def test_same_batch_dependent_metadata_is_available_before_parallel_source_io() -> None:
    context, outcomes, ledger = research_context()
    observed: list[str] = []

    def handler(arguments: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        assert outcomes.outcome_ids() == ["first"]
        assert outcomes.view()["unassessed_outcome_ids"] == []
        observed.append(str(arguments["article"]))
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="Original read")

    runner = harness(context, source_registry(handler), ledger)
    declare = read_call({"_outcomes": [requested()]})
    dependent = read_call(
        {
            "article": "2",
            "_coverage": {
                "resolutions": [
                    {
                        "outcome_id": "first",
                        "status": "unresolved",
                        "gap": "The operative exception needs its original.",
                    }
                ]
            },
        }
    )
    results = runner._dispatch([declare, dependent])
    assert [result.outcome.status for result in results] == [
        OutcomeStatus.FOUND,
        OutcomeStatus.FOUND,
    ]
    assert sorted(observed) == ["1", "2"]
    assert outcomes.revision == 2


def test_batch_metadata_is_not_reapplied_after_a_later_call_changes_the_outcome() -> (
    None
):
    context, outcomes, ledger = research_context()
    outcomes.update(OutcomeUpdate.model_validate({"outcomes": [requested()]}), ledger)
    registry = source_registry(
        lambda _arguments, _context: ToolOutcome(
            status=OutcomeStatus.FOUND, summary="Original read"
        )
    )
    assessment = read_call(
        {
            "_coverage": {
                "resolutions": [
                    {
                        "outcome_id": "first",
                        "status": "unresolved",
                        "gap": "The earlier interpretation needs an operative original.",
                    }
                ]
            }
        }
    )
    changed = requested()
    changed["detail"] = "The revised result after applying the full scenario"
    amended = read_call({"article": "2", "_outcomes": [changed]})

    results = harness(context, registry, ledger)._dispatch([assessment, amended])

    assert all(result.outcome.status == OutcomeStatus.FOUND for result in results)
    assert outcomes.view()["unassessed_outcome_ids"] == ["first"]
    assert outcomes.view()["resolutions"] == []


def test_first_native_greeting_finishes_without_outcome_map_work_or_research_model_call() -> (
    None
):
    context, outcomes, ledger = research_context()
    before = outcomes.export()
    selected, cheap = MagicMock(spec=LLM), MagicMock(spec=LLM)
    for llm in (selected, cheap):
        llm.config = LLMConfig(
            model_provider="openai",
            model_name="selected-model",
            temperature=0,
            max_input_tokens=100000,
        )
    selected.invoke.return_value = ModelResponse(
        id="greeting",
        created="0",
        choice=Choice(
            message=Message(
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id="finish",
                        function=FunctionCall(
                            name="submit_answer",
                            arguments=json.dumps(
                                {"answer": "Merhaba!", "basis": "conversation"}
                            ),
                        ),
                    )
                ]
            )
        ),
    )

    def submit(arguments: dict[str, JsonValue], _active: RunContext) -> ToolOutcome:
        assert arguments["basis"] == "conversation"
        context.services["submitted_answer"] = arguments["answer"]
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="Direct greeting")

    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="submit_answer",
                description="Finish a supported answer",
                parameters={
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "basis": {
                            "type": "string",
                            "enum": ["conversation", "scenario", "originals"],
                        },
                    },
                    "required": ["answer", "basis"],
                    "additionalProperties": False,
                },
                handler=submit,
                parallel_safe=False,
            )
        ]
    )
    adapter = ResearchModel(
        selected, context, research_llm=cheap, lean_native_mode=True
    )
    result = Harness(
        request="Merhaba",
        context=context,
        registry=registry,
        evidence=ledger,
        decide=adapter.decide,
    ).run()
    assert result.answer == "Merhaba!"
    assert result.stop_reason == "model_submitted_answer"
    assert selected.invoke.call_count == 1
    cheap.invoke.assert_not_called()
    assert outcomes.export() == before
    assert ledger.citation_mapping() == {}
