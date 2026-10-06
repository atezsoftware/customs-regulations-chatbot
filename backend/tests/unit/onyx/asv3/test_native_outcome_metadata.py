import json
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
from jsonschema import Draft202012Validator
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.legal_source_reviews import LegalSourceReviews
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


@pytest.mark.parametrize(
    "profile,parallel,child,assigned,exposed",
    [
        ("experimental", True, True, None, False),
        ("experimental", True, True, [], False),
        ("experimental", True, True, ["first"], True),
        ("experimental", True, False, [], True),
        ("experimental", False, True, [], True),
        ("normal", True, True, [], True),
        ("deep", True, True, [], True),
        (None, True, True, [], True),
    ],
)
def test_only_unbound_experimental_parallel_children_omit_outcome_schema(
    profile: str | None,
    parallel: bool,
    child: bool,
    assigned: list[str] | None,
    exposed: bool,
) -> None:
    context, _, _ = research_context()
    context.services.update(research_profile=profile, experimental_parallel=parallel)
    if child:
        context = context.independent_child()
    if assigned is not None:
        context.services["task_outcome_ids"] = assigned
    parameters = native_parameters(context)
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    assert ("_outcomes" in properties) is exposed
    assert ("_coverage" in properties) is exposed
    validator = Draft202012Validator(parameters)
    assert validator.is_valid(read_call().arguments)
    assert (
        validator.is_valid(read_call({"_outcomes": [requested()]}).arguments) is exposed
    )
    assert validator.is_valid(read_call({"_coverage": {}}).arguments) is exposed


@pytest.mark.parametrize("assigned", [None, [], ["first"]])
@pytest.mark.parametrize("through_harness", [False, True])
def test_direct_parallel_child_metadata_still_requires_its_actual_assigned_subset(
    assigned: list[str] | None, through_harness: bool
) -> None:
    parent, outcomes, ledger = research_context()
    parent.services.update(research_profile="experimental", experimental_parallel=True)
    child = parent.independent_child()
    if assigned is not None:
        child.services["task_outcome_ids"] = assigned
    observed: list[dict[str, JsonValue]] = []

    def handler(arguments: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        observed.append(arguments)
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="Original read")

    registry = source_registry(handler)
    call = read_call({"_outcomes": [requested("unassigned")], "_coverage": {}})
    before = outcomes.export()
    arguments = json.dumps(call.arguments, sort_keys=True)
    result = (
        harness(child, registry, ledger)._dispatch([call])[0].outcome
        if through_harness
        else registry.dispatch(call, child)
    )
    assert result.status == OutcomeStatus.INVALID
    assert result.data["invalid_outcome_metadata"] is True
    assert result.data["detail"] == (
        "Researcher may update only assigned outcomes"
        if assigned
        else "Researcher outcome metadata needs an assigned subset"
    )
    assert outcomes.export() == before
    assert observed == []
    assert json.dumps(call.arguments, sort_keys=True) == arguments


@pytest.mark.parametrize("binding", ["absent", "empty", "assigned", "root"])
def test_parallel_native_terminal_keeps_full_body_and_source_reviews_in_one_decision(
    binding: str,
) -> None:
    from tests.unit.onyx.asv3.test_native_model_adapter import model, native_action

    parent, outcomes, ledger = research_context()
    parent.services.update(
        research_profile="experimental",
        experimental_parallel=True,
        scenario_request="First outcome? Second outcome?",
    )
    context = parent if binding == "root" else parent.independent_child()
    if binding == "empty":
        context.services["task_outcome_ids"] = []
    elif binding == "assigned":
        outcomes.update(
            OutcomeUpdate.model_validate({"outcomes": [requested()]}), ledger
        )
        context.services["task_outcome_ids"] = ["first"]
    context.services["legal_source_reviews"] = LegalSourceReviews(
        context, "First outcome? Second outcome?"
    )
    body = ("The specific operative original remains unexamined.\n" * 650) + "\n  "
    arguments: dict[str, JsonValue] = {
        "answer": body,
        "_language": "en",
        "_related_source_reviews": [],
    }
    if binding == "assigned":
        arguments["_coverage"] = {
            "resolutions": [
                {
                    "outcome_id": "first",
                    "status": "unresolved",
                    "gap": "The specific operative original remains unexamined.",
                }
            ]
        }
    elif binding == "root":
        arguments["_outcomes"] = [requested()]
    observed: list[dict[str, JsonValue]] = []

    def partial(values: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        observed.append(values)
        return ToolOutcome(status=OutcomeStatus.PARTIAL, summary="Partial submitted")

    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="submit_partial_answer",
                description="Submit supported portions and the remaining gap",
                parameters={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
                handler=partial,
                parallel_safe=False,
            )
        ]
    )
    definition = registry.definitions(context)[0]["function"]
    assert isinstance(definition, dict)
    parameters = definition["parameters"]
    assert isinstance(parameters, dict)
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    assert "_related_source_reviews" in properties
    assert ("_outcomes" in properties) is (binding in {"assigned", "root"})
    assert ("_coverage" in properties) is (binding in {"assigned", "root"})
    Draft202012Validator(parameters).validate(arguments)
    selected, secondary = model(), model()
    selected.invoke.return_value = native_action("submit_partial_answer", arguments)
    adapter = ResearchModel(
        selected, context, research_llm=secondary, lean_native_mode=True
    )
    before = outcomes.export()
    before_revision = outcomes.revision
    result = Harness(
        request="First outcome? Second outcome?",
        context=context,
        registry=registry,
        evidence=ledger,
        decide=adapter.decide,
        partial_submission=lambda: body if observed else None,
    ).run()
    assert result.answer == body
    assert len(body) > 30000
    assert result.status == OutcomeStatus.PARTIAL
    assert result.stop_reason == "model_requested_partial_publication"
    assert observed == [{"answer": body}]
    assert selected.invoke.call_count == 1
    secondary.invoke.assert_not_called()
    assert len(result.receipts) == 1
    assert result.receipts[0].outcome.status == OutcomeStatus.PARTIAL
    if binding in {"absent", "empty"}:
        assert outcomes.export() == before
    else:
        assert outcomes.revision == before_revision + 1


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


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("through_harness", [False, True])
def test_corrected_outcome_metadata_does_not_poison_the_same_actual_source_arguments(
    parallel: bool, through_harness: bool
) -> None:
    context, outcomes, ledger = research_context()
    if parallel:
        context.services.update(
            research_profile="experimental", experimental_parallel=True
        )
    calls = 0

    def handler(_arguments: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        nonlocal calls
        calls += 1
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="Original read")

    registry = source_registry(handler)
    runner = harness(context, registry, ledger)

    def dispatch(call: CapabilityCall) -> ToolOutcome:
        return (
            runner._dispatch([call])[0].outcome
            if through_harness
            else registry.dispatch(call, context)
        )

    invalid_call = read_call({"_outcomes": [requested(question="outside")]})
    Draft202012Validator(native_parameters(context)).validate(invalid_call.arguments)
    invalid = dispatch(invalid_call)
    assert invalid.status == OutcomeStatus.INVALID
    assert invalid.data["invalid_outcome_metadata"] is True
    assert calls == 0
    assert outcomes.outcome_ids() == []
    corrected = dispatch(read_call({"_outcomes": [requested()]}))
    assert corrected.status == OutcomeStatus.FOUND
    assert calls == 1
    assert outcomes.outcome_ids() == ["first"]


def test_experimental_partial_metadata_error_can_be_corrected_without_rewriting_answer() -> (
    None
):
    context, _outcomes, ledger = research_context()
    context.services["research_profile"] = "experimental"
    outcomes = OutcomeMap(
        ["First outcome?", "Second outcome?"],
        context,
        factual_context="A prior request was filed.",
        detailed_fact_errors=True,
    )
    context.services["outcome_map"] = outcomes
    observed: list[dict[str, JsonValue]] = []

    def partial(arguments: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        observed.append(arguments)
        return ToolOutcome(status=OutcomeStatus.PARTIAL, summary="Partial submitted")

    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="submit_partial_answer",
                description="Submit supported portions and the remaining gap",
                parameters={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
                handler=partial,
            )
        ]
    )
    runner = harness(context, registry, ledger)
    answer = "The requested original remains unexamined."
    declaration = {**requested(), "decisive_facts": ["A prior Request was filed."]}
    rejected = runner._dispatch(
        [
            CapabilityCall(
                name="submit_partial_answer",
                arguments={"answer": answer, "_outcomes": [declaration]},
            )
        ]
    )[0]
    assert rejected.outcome.status == OutcomeStatus.INVALID
    assert "_outcomes[0].decisive_facts[0]" in str(rejected.outcome.data["detail"])
    assert observed == [] and outcomes.outcome_ids() == []
    corrected = runner._dispatch(
        [
            CapabilityCall(
                name="submit_partial_answer",
                arguments={
                    "answer": answer,
                    "_outcomes": [
                        {
                            **declaration,
                            "decisive_facts": ["A prior request was filed."],
                        }
                    ],
                },
            )
        ]
    )[0]
    assert corrected.outcome.status == OutcomeStatus.PARTIAL
    assert observed == [{"answer": answer}]
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
