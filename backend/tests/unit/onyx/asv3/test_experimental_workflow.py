"""Experimental source-review enforcement stays isolated and uses existing decisions."""

import json
from typing import cast

import pytest
from jsonschema import Draft202012Validator
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.legal_source_reviews import LegalSourceReviews
from onyx.asv3.llm_adapter import COORDINATOR_SESSION_ACTIONS, ResearchModel
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    OutcomeStatus,
    RunContext,
    RunStopped,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.outcome_map import OutcomeMap
from onyx.asv3.registry import CapabilityRegistry
from onyx.llm.models import ChatCompletionMessage
from onyx.prompts.asv3.coordinator_reference import (
    COORDINATOR_REFERENCE_PROMPT,
    RESEARCHER_REFERENCE_PROMPT,
)
from onyx.prompts.asv3.experimental import (
    EXPERIMENTAL_COORDINATOR_PROMPT,
    EXPERIMENTAL_RESEARCHER_PROMPT,
)
from onyx.prompts.asv3.research import (
    COORDINATOR_PROMPT,
    LEGAL_DEPARTMENT_RESEARCH,
    OUTCOME_COVERAGE_RESEARCH,
    RESEARCHER_PROMPT,
)
from tests.unit.onyx.asv3.test_legal_source_reviews import (
    deliver,
    lead_id,
    navigation,
    review,
    seen,
    setup_reviews,
)
from tests.unit.onyx.asv3.test_native_model_adapter import (
    adaptive_tool_view,
    last_payload,
    model,
    native_action,
)
from tests.unit.onyx.asv3.test_shared_originals import full_record


def experimental_context(
    *, depth: int = 0
) -> tuple[RunContext, EvidenceLedger, LegalSourceReviews]:
    context, ledger, reviews = setup_reviews()
    context.depth = depth
    context.services.update(
        evidence=ledger,
        lean_native_mode=True,
        research_profile="experimental",
        legal_source_reviews=reviews,
        legal_source_navigation=navigation,
    )
    return context, ledger, reviews


def terminal_registry(
    observed: list[dict[str, JsonValue]],
) -> CapabilityRegistry:
    def handler(arguments: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        observed.append(arguments)
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="Accepted answer")

    return CapabilityRegistry(
        [
            ToolSpec(
                name="submit_answer",
                description="Submit the complete supported answer",
                parameters={
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "minLength": 1},
                        "basis": {
                            "type": "string",
                            "enum": ["originals", "conversation"],
                        },
                    },
                    "required": ["answer", "basis"],
                    "additionalProperties": False,
                },
                handler=handler,
            ),
            ToolSpec(
                name="read_provision",
                description="Read an identified original",
                parameters={
                    "type": "object",
                    "properties": {"source_id": {"type": "string"}},
                    "required": ["source_id"],
                    "additionalProperties": False,
                },
                handler=handler,
            ),
        ]
    )


@pytest.mark.parametrize("depth", [0, 1])
@pytest.mark.parametrize("profile", ["normal", "deep"])
def test_existing_prompt_instructions_remain_exact(profile: str, depth: int) -> None:
    context = RunContext(depth=depth, services={"research_profile": profile})
    adapter = ResearchModel(model(), context, lean_native_mode=True)
    if profile == "deep":
        expected = RESEARCHER_PROMPT if depth else COORDINATOR_PROMPT
    else:
        expected = (
            RESEARCHER_REFERENCE_PROMPT
            if depth
            else COORDINATOR_REFERENCE_PROMPT + "\n\n" + COORDINATOR_SESSION_ACTIONS
        )
        expected += (
            "\n\n" + LEGAL_DEPARTMENT_RESEARCH + "\n\n" + OUTCOME_COVERAGE_RESEARCH
        )
    assert adapter._research_instruction() == expected


@pytest.mark.parametrize("depth", [0, 1])
def test_experimental_uses_isolated_prompt_without_deep_prompt(depth: int) -> None:
    context, _, _ = experimental_context(depth=depth)
    actual = ResearchModel(
        model(), context, lean_native_mode=True
    )._research_instruction()
    assert actual == (
        EXPERIMENTAL_RESEARCHER_PROMPT
        if depth
        else EXPERIMENTAL_COORDINATOR_PROMPT + "\n\n" + COORDINATOR_SESSION_ACTIONS
    )
    assert actual != (RESEARCHER_PROMPT if depth else COORDINATOR_PROMPT)


def test_only_experimental_terminal_schema_exposes_reviews() -> None:
    registry = terminal_registry([])
    normal = RunContext(
        services={"lean_native_mode": True, "research_profile": "normal"}
    )
    deep = RunContext(services={"lean_native_mode": True, "research_profile": "deep"})
    experimental, _, _ = experimental_context()
    assert registry.definitions(normal) == registry.definitions(deep)
    assert "_related_source_reviews" not in json.dumps(registry.definitions(normal))
    definitions = registry.definitions(experimental)
    terminal = cast(dict[str, JsonValue], definitions[0]["function"])
    parameters = cast(dict[str, JsonValue], terminal["parameters"])
    Draft202012Validator.check_schema(parameters)
    encoded = json.dumps(parameters)
    assert "_related_source_reviews" in encoded
    assert '"$ref"' not in encoded and '"$defs"' not in encoded
    assert '"title"' not in encoded and '"default"' not in encoded
    assert "_related_source_reviews" not in json.dumps(definitions[1])


def test_experimental_metadata_deduplication_preserves_every_tool_and_source_input() -> (
    None
):
    registry = terminal_registry([])
    read = registry.get("read_provision")
    assert read is not None
    registry.register(read.model_copy(update={"name": "read_source_range"}))
    normal = RunContext(
        services={"lean_native_mode": True, "research_profile": "normal"}
    )
    normal.services["outcome_map"] = OutcomeMap(["Can the rule be applied?"], normal)
    experimental, _, _ = experimental_context()
    experimental.services["outcome_map"] = OutcomeMap(
        ["Can the rule be applied?"], experimental
    )
    baseline = registry.definitions(normal)
    compact = registry.definitions(experimental)
    assert len(baseline) == len(compact) == 3
    for reference, current in zip(baseline, compact, strict=True):
        reference_fn = cast(dict[str, JsonValue], reference["function"])
        current_fn = cast(dict[str, JsonValue], current["function"])
        assert reference_fn["name"] == current_fn["name"]
        reference_parameters = cast(dict[str, JsonValue], reference_fn["parameters"])
        current_parameters = cast(dict[str, JsonValue], current_fn["parameters"])
        reference_properties = cast(
            dict[str, JsonValue], reference_parameters["properties"]
        )
        current_properties = cast(
            dict[str, JsonValue], current_parameters["properties"]
        )
        assert current_parameters["required"] == reference_parameters["required"]
        assert current_parameters["additionalProperties"] is False
        for name, schema in reference_properties.items():
            if name not in {"_outcomes", "_coverage"}:
                assert current_properties[name] == schema
        if current_fn["name"] in {"read_provision", "submit_answer"}:
            assert current_properties["_outcomes"] == reference_properties["_outcomes"]
            assert current_properties["_coverage"] == reference_properties["_coverage"]
        else:
            assert "_outcomes" not in current_properties
            assert "_coverage" not in current_properties


def test_fitting_only_annotates_then_actual_model_delivery_registers_lead() -> None:
    context, ledger, reviews = experimental_context()
    llm = model()
    llm.invoke.return_value = native_action("read_provision", {"source_id": "decision"})
    adapter = ResearchModel(llm, context, lean_native_mode=True)
    current = adaptive_tool_view(original_evidence=[full_record(ledger, 1)])
    prompt, _, _ = adapter._fit_native_decision(current)
    payload = json.loads(cast(str, prompt[-1].content))
    assert (
        payload["related_source_navigation"][0]["candidates"][0]["lead_id"] == lead_id()
    )
    assert reviews.view(context, ledger, {1})["pending_lead_ids"] == []
    assert adapter.decide(current).calls[0].name == "read_provision"
    assert reviews.view(context, ledger, {1})["pending_lead_ids"] == [lead_id()]
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1}
    assert context.services["last_model_call_id"] == adapter.last_call_id
    assert llm.invoke.call_count == 1


def test_navigation_omitted_from_actual_model_context_never_registers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, ledger, reviews = experimental_context()
    llm = model()
    llm.invoke.return_value = native_action("read_provision", {"source_id": "decision"})
    adapter = ResearchModel(llm, context, lean_native_mode=True)

    def cost(
        prompt: list[ChatCompletionMessage], _tools: list[dict[str, JsonValue]]
    ) -> int:
        payload = json.loads(cast(str, prompt[-1].content))
        return 999999 if "related_source_navigation" in payload else 1

    monkeypatch.setattr(adapter, "_input_cost", cost)
    adapter.decide(adaptive_tool_view(original_evidence=[full_record(ledger, 1)]))
    payload = last_payload(llm)
    assert "related_source_navigation" not in payload
    assert payload["original_evidence"] == [full_record(ledger, 1)]
    assert reviews.view(context, ledger, {1})["pending_lead_ids"] == []
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1}


def test_failed_provider_call_does_not_register_serialized_navigation() -> None:
    context, ledger, reviews = experimental_context()
    llm = model()
    llm.invoke.side_effect = RunStopped("Explicit cancellation")
    adapter = ResearchModel(llm, context, lean_native_mode=True)
    with pytest.raises(RunStopped, match="cancellation"):
        adapter.decide(adaptive_tool_view(original_evidence=[full_record(ledger, 1)]))
    assert reviews.view(context, ledger, {1})["pending_lead_ids"] == []


@pytest.mark.parametrize("candidate_delivered", [False, True])
def test_selected_candidate_keeps_review_pending_until_its_assessment(
    candidate_delivered: bool,
) -> None:
    context, ledger, reviews = experimental_context(depth=1)
    selected, cheap = model(), model()
    selected.invoke.return_value = native_action(
        "submit_answer", {"answer": "Ordinary rule [1].", "basis": "originals"}
    )
    registry = terminal_registry([])
    current = adaptive_tool_view(
        original_evidence=[
            full_record(ledger, n) for n in ([1, 2] if candidate_delivered else [1])
        ],
    ).model_copy(update={"tools": registry.definitions(context)})
    adapter = ResearchModel(
        selected, context, research_llm=cheap, lean_native_mode=True
    )
    decision = adapter.decide(current)
    assert decision.calls[0].name == "submit_answer"
    assert selected.invoke.call_count == 1
    cheap.invoke.assert_not_called()
    state = reviews.view(
        context, ledger, ledger.completely_delivered(adapter.last_call_id or "")
    )
    assert state["pending_lead_ids"] == [lead_id()]
    rows = cast(list[dict[str, JsonValue]], state["reviews"])
    assert rows[0]["status"] == "pending"
    assert rows[0]["available_original_citations"] == (
        [2] if candidate_delivered else []
    )


def test_selected_review_applies_only_when_its_terminal_action_is_dispatched() -> None:
    context, ledger, reviews = experimental_context(depth=1)
    selected, cheap = model(), model()
    arguments: dict[str, JsonValue] = {
        "answer": "The limited holding affects the ordinary rule [1] [2].",
        "basis": "originals",
        "_related_source_reviews": [review()],
    }
    cheap.invoke.return_value = native_action("submit_answer", arguments)
    selected.invoke.return_value = native_action("submit_answer", arguments)
    registry = terminal_registry([])
    current = adaptive_tool_view(
        original_evidence=[full_record(ledger, 1), full_record(ledger, 2)],
    ).model_copy(update={"tools": registry.definitions(context)})
    adapter = ResearchModel(
        selected, context, research_llm=cheap, lean_native_mode=True
    )
    decision = adapter.decide(current)
    assert decision.calls[0].name == "submit_answer"
    assert selected.invoke.call_count == 1
    cheap.invoke.assert_not_called()
    payload = last_payload(selected)
    assert "candidate_related_source_reviews" not in payload
    assert "draft_to_repair" not in payload
    assert payload["original_evidence"] == current.original_evidence
    assert reviews.view(context, ledger, {1, 2})["pending_lead_ids"] == [lead_id()]
    assert registry.dispatch(decision.calls[0], context).status == OutcomeStatus.FOUND
    assert reviews.view(context, ledger, {1, 2})["pending_lead_ids"] == []


@pytest.mark.parametrize("profile", ["normal", "deep", "experimental"])
def test_first_greeting_uses_one_selected_decision_without_research(
    profile: str,
) -> None:
    context, ledger, reviews = experimental_context()
    context.services["research_profile"] = profile
    if profile != "experimental":
        context.services.pop("legal_source_reviews")
    selected, cheap = model(), model()
    selected.invoke.return_value = native_action(
        "submit_answer", {"answer": "Merhaba!", "basis": "conversation"}
    )
    registry = terminal_registry([])
    adapter = ResearchModel(
        selected, context, research_llm=cheap, lean_native_mode=True
    )
    decision = adapter.decide(
        adaptive_tool_view().model_copy(update={"tools": registry.definitions(context)})
    )
    assert decision.calls[0].arguments["answer"] == "Merhaba!"
    assert selected.invoke.call_count == 1
    cheap.invoke.assert_not_called()
    assert (
        reviews.publication_gap("Merhaba!", adapter.last_call_id or "", context, ledger)
        is None
    )


@pytest.mark.parametrize(
    "failure", ["range", "wrong_source", "undelivered", "wrong_owner"]
)
def test_terminal_review_rejects_invalid_provenance_before_handler(
    failure: str,
) -> None:
    context, ledger, reviews = experimental_context()
    seen(context, ledger, reviews)
    deliver(ledger, "terminal", [1, 2])
    context.services["last_model_call_id"] = "terminal"
    row = review()
    if failure == "range":
        row["witnesses"] = [{"citation": 2, "start_char": 0, "end_char": 500}]
    elif failure == "wrong_source":
        row["witnesses"] = [{"citation": 1, "start_char": 0, "end_char": 5}]
    elif failure == "undelivered":
        deliver(ledger, "only-law", [1])
        context.services["last_model_call_id"] = "only-law"
    else:
        context = context.child()
        context.services["task_id"] = "different-worker"
    observed: list[dict[str, JsonValue]] = []
    result = terminal_registry(observed).dispatch(
        CapabilityCall(
            name="submit_answer",
            arguments={
                "answer": "Legal rule [1].",
                "basis": "originals",
                "_related_source_reviews": [row],
            },
        ),
        context,
    )
    assert result.status == OutcomeStatus.INVALID
    assert result.data["invalid_related_source_review"] is True
    assert observed == []
    coordinator = RunContext(run_id=context.run_id, scope=context.scope)
    assert reviews.view(coordinator, ledger, {1, 2})["pending_lead_ids"] == [lead_id()]


def test_terminal_strips_valid_review_metadata_but_retains_assessment() -> None:
    context, ledger, reviews = experimental_context()
    seen(context, ledger, reviews)
    deliver(ledger, "terminal", [1, 2])
    context.services["last_model_call_id"] = "terminal"
    observed: list[dict[str, JsonValue]] = []
    registry = terminal_registry(observed)
    result = registry.dispatch(
        CapabilityCall(
            name="submit_answer",
            arguments={
                "answer": "Limited holding [2].",
                "basis": "originals",
                "_related_source_reviews": [review()],
            },
        ),
        context,
    )
    assert result.status == OutcomeStatus.FOUND
    assert observed == [{"answer": "Limited holding [2].", "basis": "originals"}]
    assert (
        reviews.publication_gap("Limited holding [2].", "terminal", context, ledger)
        is None
    )


@pytest.mark.parametrize("profile", ["normal", "deep", "experimental"])
@pytest.mark.parametrize("dependency", ["related_review", "governing_original"])
def test_repeated_pending_review_does_not_create_experimental_research_cutoff(
    profile: str,
    dependency: str,
) -> None:
    gap = ToolOutcome(
        status=OutcomeStatus.PARTIAL,
        summary="Read the actual source effect",
        data=(
            {
                "pending_related_source_review": True,
                "unread_related_sources": [{"source_id": "decision"}],
            }
            if dependency == "related_review"
            else {"retained_authority_requirements": [{"requirement_id": "original"}]}
        ),
    )
    count = 0

    def guard(_answer: str) -> ToolOutcome | None:
        nonlocal count
        count += 1
        return gap if count <= 4 else None

    result = Harness(
        request="Can the rule be applied?",
        context=RunContext(
            services={"research_profile": profile}, timeout_seconds=float("inf")
        ),
        registry=CapabilityRegistry(),
        decide=lambda _view: Decision(answer="Limited rule [1]."),
        draft_guard=guard,
    ).run()
    if profile == "experimental":
        assert count == 5
        assert result.status == OutcomeStatus.FOUND
        assert result.stop_reason == "verified_draft"
    else:
        assert count == 3
        assert result.status == OutcomeStatus.PARTIAL
        assert result.stop_reason == "repeated_publication_gap"


def test_checkpoint_pending_reviews_survive_into_next_actual_decision() -> None:
    context, ledger, reviews = experimental_context()
    seen(context, ledger, reviews)
    restored_ledger = EvidenceLedger()
    resumed = RunContext(run_id=context.run_id, scope=context.scope)
    restored_ledger.restore(ledger.export(), resumed)
    restored = LegalSourceReviews(resumed, "Can the rule be applied?")
    restored.restore(
        reviews.export(), resumed, "Can the rule be applied?", restored_ledger
    )
    resumed.services.update(
        research_profile="experimental",
        lean_native_mode=True,
        evidence=restored_ledger,
        legal_source_reviews=restored,
    )
    llm = model()
    llm.invoke.return_value = native_action("read_provision", {"source_id": "decision"})
    ResearchModel(llm, resumed, lean_native_mode=True).decide(
        adaptive_tool_view(original_evidence=[full_record(restored_ledger, 1)])
    )
    assert last_payload(llm)["related_source_reviews"]["pending_lead_ids"] == [
        lead_id()
    ]
    restored.apply(
        [], cast(str, resumed.services["last_model_call_id"]), resumed, restored_ledger
    )
    assert (
        restored.publication_gap(
            "Ordinary rule [1].",
            cast(str, resumed.services["last_model_call_id"]),
            resumed,
            restored_ledger,
        )
        is not None
    )
