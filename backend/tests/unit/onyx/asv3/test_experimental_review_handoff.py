"""Experimental decisions retain the selected model and its source-review action."""

import json
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import OutcomeStatus
from onyx.llm.model_response import Choice, Message, ModelResponse
from onyx.llm.models import ChatCompletionMessage, ToolChoiceOptions
from tests.unit.onyx.asv3.test_experimental_workflow import (
    experimental_context,
    terminal_registry,
)
from tests.unit.onyx.asv3.test_legal_source_reviews import deliver, review, seen
from tests.unit.onyx.asv3.test_native_model_adapter import (
    adaptive_tool_view,
    last_payload,
    model,
    native_action,
)
from tests.unit.onyx.asv3.test_shared_originals import full_record


@pytest.mark.parametrize("selected_action", ["submit_answer", "read_provision"])
@pytest.mark.parametrize("provider", ["openai", "vertex_ai"])
def test_pending_review_uses_selected_model_and_keeps_research_available(
    selected_action: str,
    provider: str,
) -> None:
    context, ledger, reviews = experimental_context(depth=1)
    seen(context, ledger, reviews)
    selected, cheap = model(), model()
    selected.config = selected.config.model_copy(update={"model_provider": provider})
    arguments: dict[str, JsonValue] = {
        "answer": "The limited holding affects this rule [1] [2].",
        "basis": "originals",
        "_related_source_reviews": [review()],
    }
    cheap.invoke.return_value = native_action("submit_answer", arguments)
    selected.invoke.return_value = native_action(
        selected_action,
        arguments if selected_action == "submit_answer" else {"source_id": "decision"},
    )
    registry = terminal_registry([])
    current = adaptive_tool_view(
        original_evidence=[full_record(ledger, 1), full_record(ledger, 2)]
    ).model_copy(update={"tools": registry.definitions(context)})
    adapter = ResearchModel(
        selected, context, research_llm=cheap, lean_native_mode=True
    )
    decision = adapter.decide(current)
    assert selected.invoke.call_count == 1
    cheap.invoke.assert_not_called()
    assert selected.invoke.call_args.kwargs["tool_choice"] is (
        ToolChoiceOptions.AUTO
        if provider == "vertex_ai"
        else ToolChoiceOptions.REQUIRED
    )
    assert selected.invoke.call_args.kwargs["tools"] == current.tools
    assert "candidate_related_source_reviews" not in last_payload(selected)
    assert reviews.view(context, ledger, {1, 2})["pending_lead_ids"]
    assert decision.calls[0].name == selected_action
    if selected_action == "submit_answer":
        assert (
            registry.dispatch(decision.calls[0], context).status == OutcomeStatus.FOUND
        )
        assert reviews.view(context, ledger, {1, 2})["pending_lead_ids"] == []
        assert (
            reviews.publication_gap(
                cast(str, arguments["answer"]),
                adapter.last_call_id or "",
                context,
                ledger,
            )
            is None
        )


@pytest.mark.parametrize("profile", ["normal", "deep", "experimental"])
def test_closed_or_inactive_review_handoff_retains_auto_tool_choice(
    profile: str,
) -> None:
    context, ledger, reviews = experimental_context(depth=1)
    context.services["research_profile"] = profile
    if profile == "experimental":
        seen(context, ledger, reviews)
        deliver(ledger, "prior-review", [1, 2])
        reviews.apply([review()], "prior-review", context, ledger)
    else:
        context.services.pop("legal_source_reviews")
    selected, cheap = model(), model()
    cheap.invoke.return_value = native_action(
        "submit_answer", {"answer": "Already examined rule [1].", "basis": "originals"}
    )
    registry = terminal_registry([])
    current = adaptive_tool_view(
        original_evidence=[full_record(ledger, 1), full_record(ledger, 2)]
    ).model_copy(update={"tools": registry.definitions(context)})
    adapter = ResearchModel(
        selected, context, research_llm=cheap, lean_native_mode=True
    )
    assert adapter.decide(current).answer == "Rule [1]."
    assert selected.invoke.call_args.kwargs["tool_choice"] is ToolChoiceOptions.AUTO
    assert selected.invoke.call_count == 1
    if profile == "experimental":
        cheap.invoke.assert_not_called()
    else:
        assert cheap.invoke.call_count == 1
        assert cheap.invoke.call_args.kwargs["tool_choice"] is ToolChoiceOptions.AUTO


@pytest.mark.parametrize("profile", ["normal", "deep", "experimental"])
def test_first_greeting_does_not_force_a_source_review_action(profile: str) -> None:
    context, _, _ = experimental_context()
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
    assert selected.invoke.call_args.kwargs["tool_choice"] is ToolChoiceOptions.AUTO
    assert selected.invoke.call_count == 1
    cheap.invoke.assert_not_called()


def test_pending_review_research_stays_on_selected_model_without_a_handoff() -> None:
    context, ledger, reviews = experimental_context(depth=1)
    seen(context, ledger, reviews)
    selected, cheap = model(), model()
    selected.invoke.return_value = native_action(
        "read_provision", {"source_id": "decision"}
    )
    registry = terminal_registry([])
    current = adaptive_tool_view(original_evidence=[full_record(ledger, 1)]).model_copy(
        update={"tools": registry.definitions(context)}
    )
    adapter = ResearchModel(
        selected, context, research_llm=cheap, lean_native_mode=True
    )
    assert adapter.decide(current).calls[0].name == "read_provision"
    assert selected.invoke.call_args.kwargs["tool_choice"] is ToolChoiceOptions.REQUIRED
    assert selected.invoke.call_count == 1
    cheap.invoke.assert_not_called()


@pytest.mark.parametrize("depth", [0, 1])
@pytest.mark.parametrize("provider", ["openai", "vertex_ai"])
def test_every_experimental_decision_keeps_the_selected_model(
    depth: int,
    provider: str,
) -> None:
    context, ledger, reviews = experimental_context(depth=depth)
    seen(context, ledger, reviews)
    selected, cheap = model(), model(limit=1000)
    selected.config = selected.config.model_copy(
        update={"model_provider": provider, "model_name": "user-selected-model"}
    )
    selected.invoke.side_effect = [
        native_action("read_provision", {"source_id": "decision"}),
        native_action("read_provision", {"source_id": "decision"}),
        native_action(
            "submit_answer",
            {
                "answer": "The selected model assesses the operative holding [1] [2].",
                "basis": "originals",
                "_related_source_reviews": [review()],
            },
        ),
    ]
    registry = terminal_registry([])
    current = adaptive_tool_view(
        original_evidence=[full_record(ledger, 1), full_record(ledger, 2)]
    ).model_copy(update={"tools": registry.definitions(context)})
    adapter = ResearchModel(
        selected, context, research_llm=cheap, lean_native_mode=True
    )
    assert adapter.research_llm is None
    for _ in range(2):
        assert adapter.decide(current).calls[0].name == "read_provision"
    decision = adapter.decide(current)
    assert decision.calls[0].name == "submit_answer"
    assert selected.invoke.call_count == 3
    cheap.invoke.assert_not_called()
    assert selected.config.model_name == "user-selected-model"
    assert registry.dispatch(decision.calls[0], context).status == OutcomeStatus.FOUND
    assert reviews.view(context, ledger, {1, 2})["pending_lead_ids"] == []


def terminal_envelope(
    *, name: str = "submit_answer", assessment: dict[str, JsonValue] | None = None
) -> dict[str, JsonValue]:
    return {
        "name": name,
        "arguments": {
            "answer": "The selected model identifies the limited holding [1] [2].",
            "basis": "originals",
            "_related_source_reviews": [
                assessment
                or review(effect="Selected model's own operative assessment.")
            ],
        },
    }


def content_response(text: str) -> ModelResponse:
    return ModelResponse(
        id="content-envelope",
        created="0",
        choice=Choice(finish_reason="stop", message=Message(content=text)),
    )


@pytest.mark.parametrize("terminal", ["submit_answer", "submit_partial_answer"])
@pytest.mark.parametrize("provider", ["openai", "vertex_ai"])
def test_selected_strict_terminal_content_closes_review_without_another_model_call(
    terminal: str,
    provider: str,
) -> None:
    context, ledger, reviews = experimental_context(depth=1)
    seen(context, ledger, reviews)
    selected, cheap = model(), model()
    selected.config = selected.config.model_copy(update={"model_provider": provider})
    cheap.invoke.return_value = native_action(
        "submit_answer",
        cast(dict[str, JsonValue], terminal_envelope(assessment=review())["arguments"]),
    )
    envelope = terminal_envelope(name=terminal)
    selected.invoke.return_value = content_response(json.dumps(envelope))
    registry = terminal_registry([])
    if terminal == "submit_partial_answer":
        spec = registry.get("submit_answer")
        assert spec is not None
        registry.register(spec.model_copy(update={"name": terminal}))
    current = adaptive_tool_view(
        original_evidence=[full_record(ledger, 1), full_record(ledger, 2)]
    ).model_copy(update={"tools": registry.definitions(context)})
    adapter = ResearchModel(
        selected, context, research_llm=cheap, lean_native_mode=True
    )
    decision = adapter.decide(current)
    assert decision.answer is None
    assert decision.calls[0].name == terminal
    assert decision.calls[0].arguments == envelope["arguments"]
    assert decision.assistant_message is not None
    assert decision.assistant_message.tool_calls is not None
    assert decision.assistant_message.tool_calls[0].id == decision.calls[0].call_id
    assert selected.invoke.call_count == 1
    cheap.invoke.assert_not_called()
    assert selected.invoke.call_args.kwargs["tool_choice"] is (
        ToolChoiceOptions.AUTO
        if provider == "vertex_ai"
        else ToolChoiceOptions.REQUIRED
    )
    assert "strict JSON" in last_payload(selected)["related_source_terminal_transport"]
    assert reviews.view(context, ledger, {1, 2})["pending_lead_ids"]
    assert registry.dispatch(decision.calls[0], context).status == OutcomeStatus.FOUND
    rows = cast(
        list[dict[str, JsonValue]], reviews.view(context, ledger, {1, 2})["reviews"]
    )
    recorded_review = cast(dict[str, JsonValue], rows[0]["review"])
    assert recorded_review["effect"] == "Selected model's own operative assessment."


@pytest.mark.parametrize(
    "failure",
    [
        "wrong_name",
        "wrong_name_type",
        "unexposed_name",
        "schema",
        "wrong_source",
        "range",
        "missing_review",
        "prose",
        "embedded",
        "fenced",
        "extra_root",
        "duplicate",
        "argument_only",
        "gap_not_disclosed",
    ],
)
def test_selected_terminal_content_never_salvages_invalid_or_unwitnessed_envelopes(
    failure: str,
) -> None:
    context, ledger, reviews = experimental_context(depth=1)
    seen(context, ledger, reviews)
    selected, cheap = model(), model()
    cheap.invoke.return_value = native_action(
        "submit_answer",
        cast(dict[str, JsonValue], terminal_envelope(assessment=review())["arguments"]),
    )
    envelope = terminal_envelope()
    arguments = cast(dict[str, JsonValue], envelope["arguments"])
    if failure == "wrong_name":
        envelope["name"] = "read_provision"
    elif failure == "wrong_name_type":
        envelope["name"] = ["submit_answer"]
    elif failure == "unexposed_name":
        envelope["name"] = "submit_partial_answer"
    elif failure == "schema":
        arguments.pop("basis")
    elif failure in {"wrong_source", "range"}:
        arguments["_related_source_reviews"] = [
            review(
                witnesses=[
                    {
                        "citation": 1 if failure == "wrong_source" else 2,
                        "start_char": 0,
                        "end_char": 500 if failure == "range" else 5,
                    }
                ]
            )
        ]
    elif failure == "missing_review":
        arguments.pop("_related_source_reviews")
    elif failure == "extra_root":
        envelope["comment"] = "Not a complete action envelope"
    elif failure == "argument_only":
        arguments["_related_source_reviews"] = [review(source_role="argument_only")]
    elif failure == "gap_not_disclosed":
        arguments["_related_source_reviews"] = [
            review(
                status="unresolved",
                witnesses=[],
                gap="The exact operative interaction remains unread.",
            )
        ]
    text = json.dumps(envelope)
    if failure in {"prose", "embedded"}:
        text = (
            "Prose before the JSON: "
            + text
            + (" and after it." if failure == "embedded" else "")
        )
    elif failure == "fenced":
        text = "```json\n" + text + "\n```"
    elif failure == "duplicate":
        text = text.replace('{"name":', '{"name":"submit_answer", "name":', 1)
    selected.invoke.return_value = content_response(text)
    registry = terminal_registry([])
    current = adaptive_tool_view(
        original_evidence=[full_record(ledger, 1), full_record(ledger, 2)]
    ).model_copy(update={"tools": registry.definitions(context)})
    adapter = ResearchModel(
        selected, context, research_llm=cheap, lean_native_mode=True
    )
    decision = adapter.decide(current)
    assert decision.calls == [] and decision.answer == text
    assert selected.invoke.call_count == 1
    cheap.invoke.assert_not_called()
    assert reviews.view(context, ledger, {1, 2})["pending_lead_ids"]


@pytest.mark.parametrize("profile", ["normal", "deep"])
def test_terminal_content_fallback_does_not_change_existing_workflows(
    profile: str,
) -> None:
    context, ledger, _ = experimental_context(depth=1)
    context.services["research_profile"] = profile
    context.services.pop("legal_source_reviews")
    selected, cheap = model(), model()
    cheap.invoke.return_value = native_action(
        "submit_answer", {"answer": "Supported rule [1].", "basis": "originals"}
    )
    text = json.dumps(terminal_envelope())
    selected.invoke.return_value = content_response(text)
    adapter = ResearchModel(
        selected, context, research_llm=cheap, lean_native_mode=True
    )
    decision = adapter.decide(
        adaptive_tool_view(
            original_evidence=[full_record(ledger, 1), full_record(ledger, 2)]
        )
    )
    assert decision.answer == text and decision.calls == []
    assert "related_source_terminal_transport" not in last_payload(selected)


def test_experimental_never_uses_secondary_model_terminal_approval() -> None:
    context, ledger, reviews = experimental_context(depth=1)
    seen(context, ledger, reviews)
    selected, cheap = model(), model()
    text = json.dumps(terminal_envelope())
    cheap.invoke.return_value = content_response(text)
    adapter = ResearchModel(
        selected, context, research_llm=cheap, lean_native_mode=True
    )
    decision = adapter.decide(
        adaptive_tool_view(
            original_evidence=[full_record(ledger, 1), full_record(ledger, 2)]
        )
    )
    assert decision.answer == "Rule [1]." and decision.calls == []
    assert reviews.view(context, ledger, {1, 2})["pending_lead_ids"]
    cheap.invoke.assert_not_called()
    assert selected.invoke.call_count == 1


def test_uncited_excluded_source_witness_survives_handoff_capacity_pressure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, ledger, reviews = experimental_context()
    seen(context, ledger, reviews)
    deliver(ledger, "candidate-call", [1, 2, 3])
    candidate = review(status="not_material")
    assert (
        reviews.publication_gap(
            "The ordinary rule applies [1].",
            "candidate-call",
            context,
            ledger,
            raw_reviews=[candidate],
        )
        is None
    )
    selected = model()
    adapter = ResearchModel(selected, context, lean_native_mode=True)

    def cost(prompt: list[ChatCompletionMessage], _tools: object) -> int:
        payload = json.loads(cast(str, prompt[-1].content))
        numbers = {row["citation"] for row in payload.get("original_evidence", [])}
        return 999999 if 3 in numbers else 1

    monkeypatch.setattr(adapter, "_input_cost", cost)
    current = adaptive_tool_view(
        original_evidence=[full_record(ledger, n) for n in (1, 2, 3)]
    ).model_copy(update={"draft_to_repair": "The ordinary rule applies [1]."})
    prompt, _, _ = adapter._fit_native_decision(current, candidate_reviews=[candidate])
    payload = json.loads(cast(str, prompt[-1].content))
    assert payload["original_evidence"] == [
        full_record(ledger, 1),
        full_record(ledger, 2),
    ]
    assert {row["citation"] for row in payload["original_evidence_omitted"]} == {3}
    assert payload["candidate_related_source_reviews"] == [candidate]
    selected.invoke.assert_not_called()
