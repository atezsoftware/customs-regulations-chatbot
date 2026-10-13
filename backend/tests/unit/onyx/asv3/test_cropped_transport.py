"""Cropped packets retain originals and reject oversized physical requests."""

import copy
import json
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.cheap_metadata_transport import expand_cheap_payload
from onyx.asv3.cropped_handoff_transport import expand_cropped_handoff_transport
from onyx.asv3.cropped_transport import (
    CROPPED_INPUT_TOKEN_LIMIT,
    CroppedInputOverflow,
    enforce_cropped_input_budget,
    prepare_cropped_writer_view,
    serialize_cropped_payload,
)
from onyx.asv3.llm_adapter import ResearchModel, _NativeOriginalCatalogue
from onyx.asv3.models import EvidenceItem, RunStopped
from onyx.asv3.research_state import ResearchNeed, ResearchState, ResearchUpdate
from onyx.llm.model_response import ModelResponse, Usage
from onyx.llm.models import ChatCompletionMessage, ToolMessage, UserMessage
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.asv3.test_cheap_metadata_adapter import packet
from tests.unit.onyx.asv3.test_cropped_repair_model import current, fixture
from tests.unit.onyx.asv3.test_native_model_adapter import model, native_response, turn
from tests.unit.onyx.asv3.test_shared_originals import full_record


def test_cropped_codec_uses_actual_message_for_full_original_delivery() -> None:
    context, ledger, encoded = packet()
    context.services.pop("asv3_exp_cheap")
    context.services["asv3_exp_cropped"] = True
    canonical = json.loads(json.dumps(expand_cheap_payload(encoded)))
    before = copy.deepcopy(canonical)
    text = serialize_cropped_payload(canonical, len)
    assert expand_cropped_handoff_transport(json.loads(text)) == before
    assert canonical == before
    selected = model(limit=1_000_000)
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    prompt: list[ChatCompletionMessage] = [UserMessage(content=text)]
    bound = _NativeOriginalCatalogue.bind(prompt, cheap_metadata=True)
    adapter._invoke_once(
        prompt,
        [],
        LLMFlow.ASV3_COORDINATOR,
        max_tokens=100,
        research=False,
        structured=False,
        native_catalogue=bound,
    )
    assert ledger.completely_delivered(adapter.last_call_id or "") == set(range(1, 13))
    assert "cheap_source_review_policy" not in context.services
    assert "cheap_authority_policy" not in context.services


def test_cropped_task_and_observation_metadata_roundtrip_keeps_literal_originals() -> (
    None
):
    _, _, encoded = packet()
    canonical = json.loads(json.dumps(expand_cheap_payload(encoded)))
    shared = copy.deepcopy(canonical["original_metadata_catalogue"])
    canonical["research_tasks"] = [{"retained_source_locators": shared}]
    canonical["repair_research_observations"] = [
        {"result": {"retained_source_locators": copy.deepcopy(shared)}}
    ]
    before = copy.deepcopy(canonical)
    raw = serialize_cropped_payload(canonical, len)
    wire = json.loads(raw)
    header = wire["original_metadata_codec"]
    assert {"research_tasks", "repair_research_observations"}.issubset(
        header["payload_fields"]
    )
    assert expand_cropped_handoff_transport(wire) == before
    assert canonical == before
    assert len(raw) < len(json.dumps(before, ensure_ascii=False))
    for original in cast(list[dict[str, JsonValue]], before["original_evidence"]):
        assert cast(str, original["text"]) in raw


@pytest.mark.parametrize("cropped", [False, True])
def test_physical_call_gate_is_cropped_only(cropped: bool) -> None:
    context, ledger, _ = packet()
    context.services.pop("asv3_exp_cheap")
    context.services["asv3_exp_cropped"] = cropped
    selected = model(limit=1_000_000)
    adapter = ResearchModel(selected, context, lean_native_mode=True, token_counter=len)
    prompt: list[ChatCompletionMessage] = [
        UserMessage(content="x" * CROPPED_INPUT_TOKEN_LIMIT)
    ]
    if cropped:
        with pytest.raises(CroppedInputOverflow):
            adapter._invoke_once(
                prompt,
                [],
                LLMFlow.ASV3_ANSWER_REPAIR,
                max_tokens=100,
                research=False,
                structured=False,
            )
        selected.invoke.assert_not_called()
        assert adapter.last_call_id is None
        assert ledger.completely_delivered("missing") == set()
    else:
        adapter._invoke_once(
            prompt,
            [],
            LLMFlow.ASV3_ANSWER_REPAIR,
            max_tokens=100,
            research=False,
            structured=False,
        )
        selected.invoke.assert_called_once()


def test_tools_and_structured_format_are_counted_before_provider() -> None:
    context, _, _ = packet()
    context.services.pop("asv3_exp_cheap")
    context.services["asv3_exp_cropped"] = True
    selected = model(limit=1_000_000)
    adapter = ResearchModel(selected, context, token_counter=len)
    tools: list[dict[str, JsonValue]] = [
        {
            "type": "function",
            "function": {
                "name": "search_corpus",
                "description": "x" * CROPPED_INPUT_TOKEN_LIMIT,
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    with pytest.raises(CroppedInputOverflow):
        adapter._invoke_once(
            [UserMessage(content="small question")],
            tools,
            LLMFlow.ASV3_VERIFICATION,
            max_tokens=100,
            research=False,
        )
    selected.invoke.assert_not_called()
    with pytest.raises(CroppedInputOverflow) as overflow:
        enforce_cropped_input_budget(
            CROPPED_INPUT_TOKEN_LIMIT - 10, response_format_tokens=11
        )
    assert overflow.value.counted_tokens == CROPPED_INPUT_TOKEN_LIMIT + 1


@pytest.mark.parametrize("invalid", [True, -1, 1.5])
def test_invalid_token_count_cannot_disable_budget(invalid: object) -> None:
    with pytest.raises(ValueError):
        enforce_cropped_input_budget(cast(int, invalid))


def test_full_writer_union_replaces_passages_without_losing_fact_or_source() -> None:
    adapter, ledger, _ = fixture()
    view = current(ledger)
    original = view.model_dump(mode="json")
    fresh = prepare_cropped_writer_view(
        view, adapter.context, adapter._native_original_records(view.turns)
    )
    assert fresh.turns == []
    assert fresh.required_evidence_numbers == [1, 2, 3]
    originals = []
    for number in (1, 2, 3):
        item = ledger.get(number)
        assert item is not None
        originals.append(item.text)
    assert [row["text"] for row in fresh.original_evidence] == originals
    assert fresh.facts == view.facts and fresh.research_state == view.research_state
    assert view.model_dump(mode="json") == original
    view.original_evidence[0]["text"] = "Unrecorded source content"
    with pytest.raises(RunStopped, match="identity or text changed"):
        prepare_cropped_writer_view(view, adapter.context, [])


def test_fresh_writer_restores_all_actual_research_needs_without_stale_omission() -> (
    None
):
    adapter, ledger, _ = fixture()
    state = ResearchState(["Actual question"], adapter.context)
    state.update(
        ResearchUpdate(
            needs=[
                ResearchNeed(
                    need_id=f"need_{index}",
                    question_ids=["q0"],
                    purpose="Actual material condition " * 30,
                    completion_test="Read the exact source and relevant exception "
                    * 20,
                )
                for index in range(20)
            ]
        ),
        ledger,
    )
    adapter.context.services["research_state"] = state
    view = current(ledger).model_copy(
        update={
            "research_state": {
                **state.view(max_chars=1000),
                "cropped_completion_request": {"legal_approval": False},
            }
        }
    )
    assert view.research_state["needs_omitted"] == 20
    fresh = prepare_cropped_writer_view(view, adapter.context, [])
    assert fresh.research_state["needs"] == state.export()["needs"]
    assert fresh.research_state["needs_omitted"] == 0
    assert fresh.research_state["cropped_completion_request"] == {
        "legal_approval": False
    }
    assert view.research_state["needs_omitted"] == 20


def test_host_finish_hook_enters_actual_same_model_writer_before_research() -> None:
    adapter, ledger, selected = fixture()
    worker_item = EvidenceItem(
        source_id="completed-worker-source",
        chunk_id="worker-condition",
        text="A completed delegate obtained this governing exception.",
    )
    [worker_number] = ledger.add([worker_item], adapter.context)
    view = current(ledger).model_copy(
        update={"draft_to_repair": None, "publication_gap": None}
    )
    adapter.context.services["cropped_research_limit"] = lambda _view, _context: {
        "reason": "Observed research threshold",
        "absence_proven": False,
    }
    response = native_response("submit_partial_answer", "{}")
    response.choice.message.content = "The rule still needs a certificate [1]."
    selected.invoke.return_value = response
    decision = adapter.decide(view)
    assert decision.calls[0].name == "submit_partial_answer"
    assert selected.invoke.call_count == 1
    assert not any(
        isinstance(message, ToolMessage)
        for message in selected.invoke.call_args.kwargs["prompt"]
    )
    assert ledger.completely_delivered(adapter.last_call_id or "") == {
        1,
        2,
        3,
        worker_number,
    }
    assert worker_item.text in str(selected.invoke.call_args.kwargs["prompt"])
    assert "Observed research threshold" in str(
        selected.invoke.call_args.kwargs["prompt"]
    )


@pytest.mark.parametrize("cropped", [False, True])
def test_no_finish_request_preserves_ordinary_native_short_path(cropped: bool) -> None:
    adapter, ledger, selected = fixture()
    adapter.context.services["asv3_exp_cropped"] = cropped
    adapter.context.services["cropped_research_limit"] = lambda _view, _context: None
    view = current(ledger).model_copy(
        update={"draft_to_repair": None, "publication_gap": None}
    )
    selected.invoke.return_value = native_response(
        "submit_partial_answer", '{"answer":"Supported rule [1]."}'
    )
    decision = adapter.decide(view)
    assert decision.calls[0].name == "submit_partial_answer"
    assert selected.invoke.call_count == 1
    assert any(
        isinstance(message, ToolMessage)
        for message in selected.invoke.call_args.kwargs["prompt"]
    )
    assert adapter._writing_answer is False


def test_ordinary_overflow_activates_tail_before_fresh_source_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.asv3 import cropped_cost_budget
    from onyx.asv3.cropped_cost_budget import CroppedCostBudget
    from onyx.asv3.cropped_research_limit import CroppedResearchLimit

    monkeypatch.setattr(
        cropped_cost_budget,
        "_rates",
        lambda _model, _provider, _incoming: (2e-6, 1e-5, 0.1e-6, 2.5e-6),
    )
    adapter, ledger, selected = fixture()
    adapter._native_output_capacity = 128_000
    selected.config = selected.config.model_copy(update={"max_input_tokens": 1_000_000})
    view = current(ledger).model_copy(
        update={"draft_to_repair": None, "publication_gap": None}
    )
    controller = CroppedResearchLimit(adapter.context, view.request)
    budget = CroppedCostBudget(adapter.context, view.request)
    adapter.context.services.update(
        cropped_cost_budget=budget,
        cropped_research_controller=controller,
        cropped_research_limit=controller.finish_request,
    )
    original_check = ResearchModel._check_cropped_input

    def check(
        model: ResearchModel,
        prompt: list[ChatCompletionMessage],
        tools: list[dict[str, JsonValue]],
    ) -> dict[str, int]:
        if any(isinstance(message, ToolMessage) for message in prompt):
            raise CroppedInputOverflow(CROPPED_INPUT_TOKEN_LIMIT + 1)
        assert controller.long_tail
        return original_check(model, prompt, tools)

    monkeypatch.setattr(ResearchModel, "_check_cropped_input", check)
    response = native_response("submit_partial_answer", "{}")
    response.choice.message.content = "The certificate remains necessary [1]."
    response.usage = Usage(
        prompt_tokens=100,
        completion_tokens=10,
        total_tokens=110,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    selected.invoke.return_value = response
    decision = adapter.decide(view)
    assert decision.calls[0].name == "submit_partial_answer"
    assert selected.invoke.call_count == 1
    assert selected.invoke.call_args.kwargs["max_tokens"] == 128_000
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1, 2, 3}
    state = budget.export()["state"]
    assert isinstance(state, dict) and state["bounded"] is True
    calls = state["calls"]
    assert isinstance(calls, list) and len(calls) == 1
    assert isinstance(calls[0], dict) and calls[0]["budget_enforced"] is True
    assert controller.progress()["forced_finish_reason"] == (
        "ordinary_input_packet_exceeds_cropped_limit"
    )
    message = selected.invoke.call_args.kwargs["prompt"][-1]
    content = message.model_dump(mode="json")["content"]
    sent = (
        content
        if isinstance(content, str)
        else "".join(part["text"] for part in content if part["type"] == "text")
    )
    payload = expand_cropped_handoff_transport(json.loads(sent))
    research_state = payload["research_state"]
    assert isinstance(research_state, dict)
    completion = research_state["cropped_completion_request"]
    assert isinstance(completion, dict)
    assert completion["new_source_acquisition_closed"] is True
    assert "New source acquisition is closed" in str(completion["instruction"])
    cost_state = payload["cropped_cost_state"]
    assert isinstance(cost_state, dict) and cost_state["research_allowed"] is True


@pytest.mark.parametrize("decisions,original_count", [(7, 243), (8, 202), (9, 202)])
def test_normal_seven_to_nine_decisions_preserve_sources_and_native_output(
    monkeypatch: pytest.MonkeyPatch, decisions: int, original_count: int
) -> None:
    from onyx.asv3 import cropped_cost_budget
    from onyx.asv3.cropped_cost_budget import CroppedCostBudget
    from onyx.asv3.cropped_research_limit import CroppedResearchLimit

    monkeypatch.setattr(
        cropped_cost_budget,
        "_rates",
        lambda _model, _provider, _incoming: (2e-6, 1e-5, 0.1e-6, 2.5e-6),
    )

    def canonical_prompt(
        prompt: list[ChatCompletionMessage],
    ) -> list[dict[str, JsonValue]]:
        def canonical_text(text: str) -> JsonValue:
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                return text
            if isinstance(decoded, dict):
                return expand_cropped_handoff_transport(decoded)
            return text

        messages: list[dict[str, JsonValue]] = []
        for message in prompt:
            fields = cast(dict[str, JsonValue], message.model_dump(mode="json"))
            content = fields.get("content")
            if isinstance(content, str):
                fields["content"] = canonical_text(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        part["text"] = canonical_text(cast(str, part["text"]))
            messages.append(fields)
        return messages

    recorded_controls: list[list[dict[str, object]]] = []
    for cropped in (False, True):
        adapter, ledger, selected = fixture()
        selected.config = selected.config.model_copy(
            update={
                "model_name": "gpt-6.1-sol",
                "temperature": 1,
                "max_input_tokens": 1_000_000,
            }
        )
        adapter._native_output_capacity = 128_000
        adapter.context.services.update(
            asv3_exp_cropped=cropped, provider_compatibility_attempts=1
        )
        view = current(ledger).model_copy(
            update={"draft_to_repair": None, "publication_gap": None}
        )
        view.tools.append(
            {
                "type": "function",
                "function": {
                    "name": "read_provision",
                    "parameters": {
                        "type": "object",
                        "properties": {"article": {"type": "string"}},
                        "required": ["article"],
                    },
                },
            }
        )
        controller = CroppedResearchLimit(adapter.context, view.request)
        # The sizes reproduce the observed 5010/5082 short-path envelopes.
        controller.record_sources(
            EvidenceItem(
                source_id="recorded-family",
                chunk_id=str(index),
                text=f"Complete retained source {index} with its qualifier.",
            )
            for index in range(original_count)
        )
        controller.record_search("first-search")
        controller.record_search("second-search")
        budget = CroppedCostBudget(adapter.context, view.request)
        if cropped:
            adapter.context.services.update(
                cropped_cost_budget=budget,
                cropped_research_controller=controller,
                cropped_research_limit=controller.finish_request,
            )
        for index in range(decisions):
            terminal = index == decisions - 1
            response = native_response(
                "submit_partial_answer" if terminal else "read_provision",
                '{"answer":"The certificate remains necessary [1]."}'
                if terminal
                else '{"article":"168"}',
                call_id=f"native-{index}",
            )
            response.usage = Usage(
                prompt_tokens=100,
                completion_tokens=13_500,
                total_tokens=13_600,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            )
            selected.invoke.return_value = response
            decision = adapter.decide(view)
            assert decision.calls[0].name == (
                "submit_partial_answer" if terminal else "read_provision"
            )
            assert adapter._writing_answer is False
            assert not controller.long_tail
            assert any(
                isinstance(message, ToolMessage)
                for message in selected.invoke.call_args.kwargs["prompt"]
            )
            assert ledger.completely_delivered(adapter.last_call_id or "") == {1, 2, 3}
            view = view.model_copy(
                update={
                    "turns": [
                        *view.turns,
                        turn(f"native-{index}", [full_record(ledger, 1)]),
                    ]
                }
            )
        assert selected.invoke.call_count == decisions
        assert adapter._native_output_limit() == 128_000
        recorded_controls.append(
            [
                {
                    key: canonical_prompt(value)
                    if key == "prompt"
                    else copy.deepcopy(value)
                    for key, value in call.kwargs.items()
                    if key not in {"max_tokens", "prompt"}
                }
                for call in selected.invoke.call_args_list
            ]
        )
        assert all(
            call.kwargs["max_tokens"] == (16_384 if cropped else 128_000)
            for call in selected.invoke.call_args_list
        )
        if cropped:
            state = budget.export()["state"]
            assert isinstance(state, dict)
            assert state["spent_usd"] > 0.90
            assert state["uncertain_usd"] == pytest.approx(decisions * 100 * 0.5e-6)
            assert len(state["calls"]) == decisions
            assert budget.finish_reason is None
            assert controller.progress()["research_model_calls"] == decisions
    assert recorded_controls[0] == recorded_controls[1]


@pytest.mark.parametrize("method", ["invoke", "stream"])
@pytest.mark.parametrize("requested", [None, 256, 512, 128_000])
def test_short_path_search_generations_preserve_original_output_control(
    monkeypatch: pytest.MonkeyPatch, method: str, requested: int | None
) -> None:
    import litellm

    from onyx.asv3 import cropped_cost_budget
    from onyx.asv3.cropped_cost_budget import CroppedCostBudget
    from onyx.asv3.cropped_research_limit import CroppedResearchLimit
    from onyx.asv3.search_adapter import ScopedSearchLLM
    from onyx.llm.model_response import ModelResponseStream, StreamingChoice

    adapter, _, selected = fixture()
    monkeypatch.setattr(
        litellm, "get_model_info", lambda **_kwargs: {"max_output_tokens": 128_000}
    )
    monkeypatch.setattr(
        cropped_cost_budget,
        "_rates",
        lambda _model, _provider, _incoming: (2e-6, 1e-5, 0.1e-6, 2.5e-6),
    )
    budget = CroppedCostBudget(adapter.context, "Actual question")
    controller = CroppedResearchLimit(adapter.context, "Actual question")
    adapter.context.services.update(
        cropped_cost_budget=budget,
        cropped_research_controller=controller,
        cropped_token_counter=len,
    )
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=10,
        total_tokens=110,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    response = native_response(None)
    response.usage = usage
    selected.invoke.return_value = response
    selected.stream.return_value = iter(
        [
            ModelResponseStream(
                id="local",
                created="0",
                choice=StreamingChoice(finish_reason="stop"),
                usage=usage,
            )
        ]
    )
    scoped = ScopedSearchLLM(selected, adapter.context, None)
    prompt: list[ChatCompletionMessage] = [
        UserMessage(content="An exact source and effect")
    ]
    if method == "invoke":
        assert scoped.invoke(prompt, max_tokens=requested) is response
        selected_method = selected.invoke
    else:
        assert list(scoped.stream(prompt, max_tokens=requested))[0].usage == usage
        selected_method = selected.stream
    assert selected_method.call_args.kwargs["max_tokens"] == (requested or 128_000)
    assert selected_method.call_args.kwargs["prompt"] is prompt
    assert not controller.long_tail
    state = budget.export()["state"]
    assert isinstance(state, dict) and len(state["calls"]) == 1
    assert state["spent_usd"] == pytest.approx(0.0003)
    assert state["uncertain_usd"] == pytest.approx(100 * 0.5e-6)


def test_observed_tail_search_generation_keeps_native_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.asv3 import cropped_cost_budget
    from onyx.asv3.cropped_cost_budget import CroppedCostBudget
    from onyx.asv3.cropped_research_limit import CroppedResearchLimit
    from onyx.asv3.search_adapter import ScopedSearchLLM

    adapter, _, selected = fixture()
    monkeypatch.setattr(
        cropped_cost_budget,
        "_rates",
        lambda _model, _provider, _incoming: (2e-6, 1e-5, 0.1e-6, 2.5e-6),
    )
    controller = CroppedResearchLimit(adapter.context, "Actual question")
    controller.record_search("s1")
    controller.record_search("s2")
    controller.record_search("s3")
    for _ in range(9):
        controller.reserve_model("root")
    controller.record_sources(
        EvidenceItem(source_id=f"source-{i}", text=f"Original {i}") for i in range(256)
    )
    assert controller.long_tail and not controller.forced_finish
    adapter.context.services.update(
        cropped_cost_budget=CroppedCostBudget(adapter.context, "Actual question"),
        cropped_research_controller=controller,
        cropped_token_counter=len,
    )
    response = native_response(None)
    response.usage = Usage(
        prompt_tokens=100,
        completion_tokens=10,
        total_tokens=110,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    selected.invoke.return_value = response
    ScopedSearchLLM(selected, adapter.context, None).invoke(
        [UserMessage(content="An exact bounded source lookup")], max_tokens=128_000
    )
    assert selected.invoke.call_args.kwargs["max_tokens"] == 128_000


def test_unpriced_research_hands_off_to_first_writer_and_records_unknown_charge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.asv3 import cropped_cost_budget
    from onyx.asv3.cropped_cost_budget import CroppedCostBudget
    from onyx.asv3.cropped_research_limit import CroppedResearchLimit

    def unavailable(_model: str, _provider: str, _incoming: int) -> tuple[float, ...]:
        raise RunStopped("The local selected-model pricing entry is unavailable")

    monkeypatch.setattr(cropped_cost_budget, "_rates", unavailable)
    adapter, ledger, selected = fixture()
    adapter._native_output_capacity = 128_000
    selected.config = selected.config.model_copy(update={"max_input_tokens": 1_000_000})
    view = current(ledger).model_copy(
        update={"draft_to_repair": None, "publication_gap": None}
    )
    controller = CroppedResearchLimit(adapter.context, view.request)
    budget = CroppedCostBudget(adapter.context, view.request)
    adapter.context.services.update(
        cropped_cost_budget=budget,
        cropped_research_controller=controller,
        cropped_research_limit=controller.finish_request,
    )
    response = native_response(
        "submit_partial_answer", '{"answer":"The certificate remains necessary [1]."}'
    )
    response.usage = Usage(
        prompt_tokens=100,
        completion_tokens=10,
        total_tokens=110,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    selected.invoke.return_value = response
    decision = adapter.decide(view)
    assert decision.calls[0].name == "submit_partial_answer"
    assert selected.invoke.call_count == 1
    assert selected.invoke.call_args.kwargs["max_tokens"] == 128_000
    assert adapter._native_output_limit() == 128_000
    assert adapter._writing_answer is False
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1, 2, 3}
    assert not budget.research_allowed and not controller.long_tail
    state = budget.export()["state"]
    assert isinstance(state, dict) and len(state["calls"]) == 1
    assert len(state["unpriced_call_ids"]) == 1
    calls = state["calls"]
    assert isinstance(calls, list) and isinstance(calls[0], dict)
    assert calls[0]["known_cost_usd"] is None
    assert calls[0]["price_available"] is False


def test_section_adapter_writes_actual_original_receipts_then_body_only_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, ledger, selected = fixture()
    from onyx.asv3.evidence import EvidenceLedger
    from tests.unit.onyx.asv3.test_shared_originals import full_record, original

    view = current(ledger).model_copy(
        update={"draft_to_repair": None, "publication_gap": None}
    )
    ledger = EvidenceLedger()
    ledger.add(
        [
            original(
                str(n),
                f"Distinct complete provision {n}.",
                source=f"statute-{n}",
                metadata={"article_no": str(n)},
            )
            for n in (1, 2, 3)
        ],
        adapter.context,
    )
    adapter.context.services["evidence"] = ledger
    view.turns = []
    view.original_evidence = [full_record(ledger, n) for n in (1, 2, 3)]
    full = prepare_cropped_writer_view(
        view, adapter.context, adapter._native_original_records(view.turns)
    )
    guarded: list[tuple[str, str]] = []

    def guard(body: str, call_id: str, context: object) -> None:
        from onyx.asv3.citation_numbers import extract_citation_numbers

        assert context is adapter.context
        assert set(extract_citation_numbers(body)).issubset(
            ledger.completely_delivered(call_id)
        )
        guarded.append((body, call_id))

    adapter.context.services["cropped_section_publication_gap"] = guard

    def count(
        _self: ResearchModel, prompt: list[ChatCompletionMessage], _tools: object
    ) -> dict[str, int]:
        catalogue = _NativeOriginalCatalogue.bind(prompt, cheap_metadata=True)
        total = 90_000 * len(catalogue.records) + 1000
        return enforce_cropped_input_budget(total)

    monkeypatch.setattr(ResearchModel, "_check_cropped_input", count)
    supplied_sections: list[str] = []

    def respond(**kwargs: object) -> ModelResponse:
        prompt = cast(list[ChatCompletionMessage], kwargs["prompt"])
        if kwargs.get("tools"):
            catalogue = _NativeOriginalCatalogue.bind(prompt, cheap_metadata=True)
            citations = [cast(int, row["citation"]) for row in catalogue.records]
            assert set(citations) in ({1, 2}, {3})
            supplied_sections.append("Clause " + " ".join(f"[{n}]" for n in citations))
            result = native_response("submit_partial_answer", "{}")
            result.choice.message.content = "  " + supplied_sections[-1] + "  "
            return result
        last = prompt[-1].content
        assert isinstance(last, str)
        payload = json.loads(last)
        assert payload["answer"] == "\n\n".join(supplied_sections)
        assert "original_evidence" not in payload
        result = model().invoke.return_value
        result.choice.message.content = json.dumps(
            {
                "review_scope": "section_coverage_and_consistency",
                "assessed_section_ids": [
                    section["plan"]["section_id"] for section in payload["sections"]
                ],
                "assessed_determination_ids": [
                    row["determination_id"] for row in payload["determinations"]
                ],
                "complete": True,
                "consistent": True,
                "missing": [],
                "contradictions": [],
            }
        )
        return result

    selected.invoke.side_effect = respond
    result = adapter._assemble_cropped_sections(full)
    assert result.calls[0].name == "submit_partial_answer"
    assert result.calls[0].arguments["answer"] == "\n\n".join(supplied_sections)
    assert selected.invoke.call_count == 3
    assert len({call_id for _, call_id in guarded}) == 2
    assert ledger.completely_delivered(adapter.last_call_id or "") == set()


def test_research_dollar_refusal_goes_to_funded_same_model_writer_without_paid_research(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.asv3 import cropped_cost_budget
    from onyx.asv3.cropped_cost_budget import CroppedCostBudget
    from onyx.asv3.cropped_research_limit import CroppedResearchLimit

    adapter, ledger, selected = fixture()
    view = current(ledger).model_copy(
        update={"draft_to_repair": None, "publication_gap": None}
    )
    monkeypatch.setattr(
        cropped_cost_budget,
        "_rates",
        lambda _model, _provider, _incoming: (1e-6, 1e-6, 0.1e-6, 2.5e-6),
    )
    budget = CroppedCostBudget(
        adapter.context,
        view.request,
        research_usd=0.00001,
        repair_usd=0.000005,
        total_llm_usd=0.5,
    )
    controller = CroppedResearchLimit(adapter.context, view.request)
    adapter.context.services.update(
        cropped_cost_budget=budget, cropped_research_controller=controller
    )
    controller.force_finish("Only the outlier path enforces its generation allowance")
    response = native_response("submit_partial_answer", "{}")
    response.choice.message.content = "The certificate remains necessary [1]."
    response.usage = Usage(
        prompt_tokens=100,
        completion_tokens=10,
        total_tokens=110,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    selected.invoke.return_value = response
    result = adapter.decide(view)
    assert result.calls[0].name == "submit_partial_answer"
    assert selected.invoke.call_count == 1
    assert controller.progress()["research_model_calls"] == 0
    assert controller.progress()["final_model_calls"] == 1
    state = budget.export()["state"]
    assert isinstance(state, dict) and len(state["calls"]) == 1
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1, 2, 3}


def test_each_retry_reserves_and_completes_its_own_physical_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.asv3 import cropped_cost_budget, llm_adapter
    from onyx.asv3.cropped_cost_budget import CroppedCostBudget
    from onyx.asv3.cropped_research_limit import CroppedResearchLimit

    adapter, _, selected = fixture()
    monkeypatch.setattr(
        cropped_cost_budget,
        "_rates",
        lambda _model, _provider, _incoming: (1e-6, 1e-6, 0.1e-6, 2.5e-6),
    )
    monkeypatch.setattr(llm_adapter, "is_retryable_provider_error", lambda _error: True)
    monkeypatch.setattr(llm_adapter, "provider_retry_delay", lambda _error, _attempt: 0)
    budget = CroppedCostBudget(adapter.context, "Question")
    controller = CroppedResearchLimit(adapter.context, "Question")
    adapter.context.services.update(
        cropped_cost_budget=budget,
        cropped_research_controller=controller,
        provider_max_attempts=2,
    )
    response = selected.invoke.return_value
    response.usage = Usage(
        prompt_tokens=100,
        completion_tokens=10,
        total_tokens=110,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    selected.invoke.side_effect = [RuntimeError("Retryable provider failure"), response]
    adapter._invoke(
        [UserMessage(content="Question")],
        [],
        LLMFlow.ASV3_COORDINATOR,
        max_tokens=100,
        research=False,
        structured=False,
    )
    state = budget.export()["state"]
    assert isinstance(state, dict) and len(state["calls"]) == 2
    assert state["uncertain_usd"] > 0
    assert controller.progress()["research_model_calls"] == 2
    assert selected.invoke.call_count == 2
