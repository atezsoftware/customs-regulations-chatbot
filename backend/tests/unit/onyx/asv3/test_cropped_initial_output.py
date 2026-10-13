"""Initial output sizing must retain native completion and source ownership."""

import copy
import json
from typing import Literal, cast
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, JsonValue

from onyx.asv3 import cropped_cost_budget as cost_module
from onyx.asv3.cropped_cost_budget import CroppedCostBudget
from onyx.asv3.cropped_research_limit import CroppedResearchLimit
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import (
    HarnessView,
    OutcomeStatus,
    RunContext,
    SharedBudget,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.research_state import ResearchUpdate
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.llm.model_response import Choice, Message, ModelResponse, Usage
from onyx.llm.models import (
    AssistantMessage,
    ChatCompletionMessage,
    TextContentPart,
    UserMessage,
)
from onyx.llm.multi_llm import LitellmLLM
from onyx.llm.native_responses_stream import (
    CroppedResponseTerminalCapture,
    CroppedResponseTerminalError,
)
from onyx.tracing.flows import LLMFlow
from onyx.tracing.framework.create import trace
from onyx.tracing.llm_utils import record_llm_request_params
from tests.unit.onyx.asv3.test_native_model_adapter import (
    model,
    native_response,
    original,
    view,
)


def adapter_fixture(
    *, depth: int = 0, cropped: bool = True, cheap: bool = False
) -> tuple[ResearchModel, EvidenceLedger, MagicMock, HarnessView]:
    context = RunContext(depth=depth, budget=SharedBudget(unlimited_execution=True))
    ledger = EvidenceLedger()
    records = [
        original(ledger, context, "The duty requires a signed certificate."),
        original(ledger, context, "The later exception requires authority consent."),
    ]
    context.services.update(
        evidence=ledger,
        research_profile="normal",
        asv3_workflow_variant=ASV3_TUNED_VARIANT,
        asv3_exp_cropped=cropped,
        asv3_exp_cheap=cheap,
    )
    selected = model(limit=1_000_000)
    adapter = ResearchModel(selected, context, lean_native_mode=True, token_counter=len)
    adapter._native_output_capacity = 128_000
    current = view(original_evidence=records).model_copy(
        update={
            "facts": ["The certificate has not been signed."],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_provision",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "source_id": {"type": "string"},
                                "article": {"type": "string"},
                            },
                            "required": ["source_id", "article"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
        }
    )
    return adapter, ledger, selected, current


@pytest.mark.parametrize("depth", [0, 1])
def test_initial_cropped_native_decision_has_soft_limit_and_full_capacity(
    depth: int,
) -> None:
    adapter, ledger, selected, current = adapter_fixture(depth=depth)
    selected.invoke.return_value = native_response(
        "read_provision", '{"source_id":"law","article":"8"}'
    )
    before = current.model_dump(mode="json")
    decision = adapter.decide(current)
    assert selected.invoke.call_args.kwargs["max_tokens"] == 16_384
    assert adapter._native_output_limit() == 128_000
    assert decision.calls[0].arguments == {"source_id": "law", "article": "8"}
    assert current.model_dump(mode="json") == before
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1, 2}


@pytest.mark.parametrize(
    "depth,task_id,independent,cropped,cheap,delegated",
    [
        (0, None, False, True, False, False),
        (1, "host-owned-worker", False, True, False, True),
        (1, None, False, True, False, False),
        (1, "root", False, True, False, False),
        (1, "host-owned-worker", True, True, False, False),
        (1, "host-owned-worker", False, False, False, False),
        (1, "host-owned-worker", False, True, True, False),
    ],
)
def test_actual_adapter_invocation_registers_only_scoped_delegated_research(
    depth: int,
    task_id: str | None,
    independent: bool,
    cropped: bool,
    cheap: bool,
    delegated: bool,
) -> None:
    adapter, ledger, selected, current = adapter_fixture(
        depth=depth, cropped=cropped, cheap=cheap
    )
    adapter.context.services.update(task_id=task_id, independent_question=independent)
    controller = CroppedResearchLimit(adapter.context, current.request)
    adapter.context.services["cropped_research_controller"] = controller
    selected.invoke.return_value = native_response(
        "read_provision", '{"source_id":"law","article":"8"}'
    )
    before = current.model_dump(mode="json")
    if cropped and cheap:
        with pytest.raises(ValueError, match="isolated Cropped workflow"):
            adapter.decide(current)
        selected.invoke.assert_not_called()
        assert current.model_dump(mode="json") == before
        return
    decision = adapter.decide(current)

    assert controller.has_delegated_research is delegated
    assert decision.calls[0].arguments == {"source_id": "law", "article": "8"}
    assert current.model_dump(mode="json") == before
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1, 2}


def test_delegated_eighth_call_keeps_ordinary_research_without_forced_finish() -> None:
    adapter, ledger, selected, current = adapter_fixture()
    controller = CroppedResearchLimit(adapter.context, current.request)
    adapter.context.services["cropped_research_controller"] = controller
    adapter.context.services["cropped_research_limit"] = controller.finish_request
    worker = adapter.context.child()
    worker.services["task_id"] = "host-owned-worker"
    controller.reserve_model("host-owned-worker", context=worker)
    for _ in range(7):
        controller.reserve_model("root", context=adapter.context)
    selected.invoke.return_value = native_response(
        "read_provision", '{"source_id":"law","article":"8"}'
    )
    before = current.model_dump(mode="json")

    decision = adapter.decide(current)

    assert not controller.budget_limited and not controller.long_tail
    assert not controller.forced_finish
    assert controller.progress()["research_model_calls"] == 9
    assert controller.progress()["final_model_calls"] == 0
    assert selected.invoke.call_args.kwargs["max_tokens"] == 16_384
    assert decision.calls[0].arguments == {"source_id": "law", "article": "8"}
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1, 2}
    assert current.model_dump(mode="json") == before
    assert not adapter._writing_answer
    assert controller.admit_tool("resolve_source", {"query": "needed rule"}).allowed


def test_soft_limit_changes_only_output_allowance_in_same_native_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, ledger, selected, current = adapter_fixture()
    selected.invoke.return_value = native_response(
        "read_provision", '{"source_id":"law","article":"8"}'
    )
    prompt, tools, output = adapter._fit_native_decision(current)
    original_limit = adapter._cropped_initial_decision_output
    monkeypatch.setattr(
        adapter, "_cropped_initial_decision_output", lambda output, _flow: output
    )
    adapter._invoke_decision(
        current, copy.deepcopy(prompt), copy.deepcopy(tools), output
    )
    baseline = copy.deepcopy(selected.invoke.call_args.kwargs)
    monkeypatch.setattr(adapter, "_cropped_initial_decision_output", original_limit)
    decision = adapter._invoke_decision(
        current, copy.deepcopy(prompt), copy.deepcopy(tools), output
    )
    capped = copy.deepcopy(selected.invoke.call_args.kwargs)
    assert baseline.pop("max_tokens") == 128_000
    assert capped.pop("max_tokens") == 16_384
    assert capped == baseline
    assert decision.calls[0].arguments == {"source_id": "law", "article": "8"}
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1, 2}


@pytest.mark.parametrize("cheap", [False, True])
def test_existing_experimental_and_cheap_initial_output_are_unchanged(
    cheap: bool,
) -> None:
    adapter, _, selected, current = adapter_fixture(cropped=False, cheap=cheap)
    selected.invoke.return_value = native_response(
        "read_provision", '{"source_id":"law","article":"8"}'
    )
    adapter.decide(current)
    assert selected.invoke.call_args.kwargs["max_tokens"] == 128_000


@pytest.mark.parametrize(
    "flow,writing_answer",
    [
        (LLMFlow.ASV3_FINAL, False),
        (LLMFlow.ASV3_VERIFICATION, False),
        (LLMFlow.ASV3_ANSWER_REPAIR, False),
        (LLMFlow.ASV3_TARGETED_RESEARCH, False),
        (LLMFlow.ASV3_COORDINATOR, True),
        (LLMFlow.ASV3_RESEARCHER, True),
    ],
)
def test_writer_sections_verifiers_and_focus_calls_keep_requested_capacity(
    flow: LLMFlow, writing_answer: bool
) -> None:
    adapter, _, selected, current = adapter_fixture()
    adapter._writing_answer = writing_answer
    selected.invoke.return_value = native_response(
        "read_provision", '{"source_id":"law","article":"8"}'
    )
    prompt, tools, output = adapter._fit_native_decision(current)
    adapter._invoke_decision(current, prompt, tools, output, flow_override=flow)
    assert selected.invoke.call_args.kwargs["max_tokens"] == 128_000
    assert adapter._native_output_limit() == 128_000


def test_an_already_smaller_requested_allowance_is_not_increased() -> None:
    adapter, _, _, _ = adapter_fixture()
    assert (
        adapter._cropped_initial_decision_output(2048, LLMFlow.ASV3_RESEARCHER) == 2048
    )


def test_real_structured_verification_keeps_its_output_request() -> None:
    class Assessment(BaseModel):
        status: Literal["supported"]

    adapter, _, selected, _ = adapter_fixture()
    selected.invoke.return_value = ModelResponse(
        id="assessment",
        created="0",
        choice=Choice(
            finish_reason="stop", message=Message(content='{"status":"supported"}')
        ),
    )
    text = adapter.invoke_text(
        "Assess the supplied body without editing it.",
        '{"body":"The conditional duty [1]."}',
        LLMFlow.ASV3_VERIFICATION,
        max_tokens=24_000,
        response_model_override=Assessment,
    )
    assert json.loads(text) == {"status": "supported"}
    assert selected.invoke.call_args.kwargs["max_tokens"] == 24_000


def test_truncated_native_batch_executes_only_the_complete_regeneration() -> None:
    adapter, ledger, selected, current = adapter_fixture()
    executed: list[tuple[str, dict[str, JsonValue]]] = []
    findings: list[dict[str, JsonValue]] = [
        {
            "finding_id": f"finding_{index}",
            "need_id": "certificate",
            "statement": (
                f"Material qualification {index}: "
                + "Şart gerçekleşmedikçe hüküm uygulanmaz; istisna ve bildirim koşulları ayrıca korunur. "
                * 22
            ),
            "kind": "rule",
            "witnesses": [{"citation": 1, "start_char": 0, "end_char": 43}],
        }
        for index in range(24)
    ]
    update: dict[str, JsonValue] = {"findings": cast(list[JsonValue], findings)}
    ResearchUpdate.model_validate(update)
    complete = native_response("update_research", json.dumps(update), call_id="large")
    complete.choice.finish_reason = "stop"
    valid = native_response(
        "read_provision", '{"source_id":"law","article":"8"}', call_id="read"
    )
    assert valid.choice.message.tool_calls and complete.choice.message.tool_calls
    complete.choice.message.tool_calls = [
        *valid.choice.message.tool_calls,
        *complete.choice.message.tool_calls,
    ]
    cut = copy.deepcopy(complete)
    cut.choice.finish_reason = "length"
    assert cut.choice.message.tool_calls
    cut.choice.message.tool_calls[
        1
    ].function.arguments = '{"findings":[{"finding_id":"unfinished"'
    final = ModelResponse(
        id="finished",
        created="0",
        choice=Choice(
            finish_reason="stop",
            message=Message(
                content="The signed-certificate condition [1] and consent exception [2] remain material."
            ),
        ),
    )

    def invoke(**_kwargs: object) -> ModelResponse:
        if selected.invoke.call_count == 1:
            assert executed == []
            return cut
        if selected.invoke.call_count == 2:
            assert executed == []
            return complete
        assert len(executed) == 2
        assert dict(executed) == {
            "read_provision": {"source_id": "law", "article": "8"},
            "update_research": update,
        }
        return final

    def read(args: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        executed.append(("read_provision", copy.deepcopy(args)))
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="Recorded bounded read.")

    def record(args: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        executed.append(("update_research", copy.deepcopy(args)))
        return ToolOutcome(
            status=OutcomeStatus.FOUND, summary="Complete supplied conditions retained."
        )

    selected.invoke.side_effect = invoke
    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="read_provision",
                description="Read the named provision.",
                parameters=cast(
                    dict[str, JsonValue],
                    cast(dict[str, JsonValue], current.tools[0]["function"])[
                        "parameters"
                    ],
                ),
                handler=read,
                parallel_safe=False,
            ),
            ToolSpec(
                name="update_research",
                description="Retain source-bound findings.",
                parameters=ResearchUpdate.model_json_schema(),
                handler=record,
                parallel_safe=False,
            ),
        ]
    )
    harness = Harness(
        request=current.request,
        context=adapter.context,
        registry=registry,
        evidence=ledger,
        decide=adapter.decide,
        report_terminal=False,
    )
    result = harness.run()
    assert result.answer == final.choice.message.content
    assert [call.kwargs["max_tokens"] for call in selected.invoke.call_args_list] == [
        16_384,
        128_000,
        16_384,
    ]
    assert len(result.receipts) == 2
    assert {receipt.call.name for receipt in result.receipts} == {
        "read_provision",
        "update_research",
    }
    assert (
        next(
            receipt.call.arguments
            for receipt in result.receipts
            if receipt.call.name == "update_research"
        )
        == update
    )
    assert all(receipt.call.argument_error is None for receipt in result.receipts)
    assert adapter._native_output_limit() == 128_000


@pytest.mark.parametrize("finish", ["length", "stop"])
def test_literal_answer_continuation_retains_exact_prefix_and_source_receipts(
    finish: str,
) -> None:
    adapter, ledger, selected, current = adapter_fixture()
    prefix = "The duty requires a signed certificate; the quotation continues: “"
    suffix = "a signed certificate.” [1]\n\nThe exception still requires authority consent [2]."
    first = ModelResponse(
        id="cut",
        created="0",
        choice=Choice(finish_reason=finish, message=Message(content=prefix)),
        usage=Usage(
            prompt_tokens=100,
            completion_tokens=16_384,
            total_tokens=16_484,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            reasoning_tokens=100,
        ),
    )
    selected.invoke.side_effect = [
        first,
        ModelResponse(
            id="completed",
            created="0",
            choice=Choice(finish_reason="stop", message=Message(content=suffix)),
        ),
    ]
    decision = adapter.decide(current)
    assert decision.answer == prefix + suffix
    assert [call.kwargs["max_tokens"] for call in selected.invoke.call_args_list] == [
        16_384,
        128_000,
    ]
    continuation = selected.invoke.call_args.kwargs
    assert continuation["tools"] is None
    assert any(
        isinstance(message, AssistantMessage) and message.content == prefix
        for message in continuation["prompt"]
    )
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1, 2}
    assert (
        adapter._native_output_limit() == 128_000
        and not adapter.last_response_truncated
    )
    original_prompt = cast(
        list[ChatCompletionMessage], selected.invoke.call_args_list[0].kwargs["prompt"]
    )
    continued_prompt = cast(list[ChatCompletionMessage], continuation["prompt"])
    assert continued_prompt[: len(original_prompt)] == original_prompt


@pytest.mark.parametrize("tool", [False, True])
def test_actual_completed_terminal_at_cap_does_not_request_continuation(
    tool: bool,
) -> None:
    adapter, ledger, selected, current = adapter_fixture()
    response = (
        native_response("read_provision", '{"source_id":"law","article":"8"}')
        if tool
        else ModelResponse(
            id="completed",
            created="0",
            choice=Choice(
                message=Message(
                    content="Signed certificate [1]; authority consent [2]."
                )
            ),
        )
    )
    response.choice.finish_reason = "stop"
    response.usage = Usage(
        prompt_tokens=100,
        completion_tokens=16_384,
        total_tokens=16_484,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    capture = CroppedResponseTerminalCapture(
        event_type="response.completed",
        status="completed",
        response_id="actual-completed-response",
        usage=response.usage,
        usage_status="provider_reported",
    )

    def completed(**_kwargs: object) -> ModelResponse:
        record_llm_request_params(
            {"asv3_exp_cropped_response_terminal": capture.diagnostic()}
        )
        return response

    selected.invoke.side_effect = completed
    with trace("Cropped actual completed at cap"):
        decision = adapter.decide(current)
    selected.invoke.assert_called_once()
    assert selected.invoke.call_args.kwargs["max_tokens"] == 16_384
    assert decision.answer == response.choice.message.content
    assert bool(decision.calls) is tool
    assert adapter.last_finish_reason == "stop" and not adapter.last_response_truncated
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1, 2}


def test_typed_failed_terminal_settles_real_usage_without_source_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, ledger, selected, current = adapter_fixture()
    monkeypatch.setattr(
        cost_module, "_rates", lambda *_: (0.000002, 0.00001, 0.0000001, 0.0000025)
    )
    budget = CroppedCostBudget(adapter.context, current.request)
    adapter.context.services["cropped_cost_budget"] = budget
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=137,
        total_tokens=237,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        reasoning_tokens=21,
    )
    error = CroppedResponseTerminalError(
        CroppedResponseTerminalCapture(
            event_type="response.failed",
            status="failed",
            response_id="actual-failed-response",
            usage=usage,
            usage_status="provider_reported",
        )
    )
    selected.invoke.side_effect = error
    with pytest.raises(CroppedResponseTerminalError, match="terminal failure: failed"):
        adapter.decide(current)
    selected.invoke.assert_called_once()
    state = budget.export()["state"]
    assert isinstance(state, dict)
    assert state["spent_usd"] == pytest.approx(100 * 0.000002 + 137 * 0.00001)
    assert state["uncertain_usd"] == pytest.approx(100 * 0.0000005)
    assert state["unpriced_call_ids"] == []
    calls = cast(list[dict[str, JsonValue]], state["calls"])
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == 16_384
    assert calls[0]["retained_reserve_usd"] == 0
    assert adapter.last_call_id is None
    assert ledger.completely_delivered("actual-failed-response") == set()


def test_explicit_cache_only_first_large_tail_source_writer() -> None:
    adapter, _ledger, _selected, current = adapter_fixture()
    controller = CroppedResearchLimit(adapter.context, current.request)
    adapter.context.services["cropped_research_controller"] = controller
    controller.force_finish("actual long-tail read boundary")
    prompt: list[ChatCompletionMessage] = [UserMessage(content="original text " * 9000)]
    retained = copy.deepcopy(prompt)
    # Research uses the warm implicit cache even with the same large originals.
    assert not adapter._cropped_large_writer_cache(
        prompt, current.tools, LLMFlow.ASV3_COORDINATOR
    )
    adapter._writing_answer = True
    assert adapter._cropped_large_writer_cache(
        prompt, current.tools, LLMFlow.ASV3_FINAL
    )
    assert not adapter._cropped_large_writer_cache(
        prompt, current.tools, LLMFlow.ASV3_FINAL
    )
    assert prompt == retained


@pytest.mark.parametrize("cropped,tail", [(False, True), (True, False)])
def test_normal_writer_keeps_existing_cache(cropped: bool, tail: bool) -> None:
    adapter, _ledger, _selected, current = adapter_fixture(cropped=cropped)
    controller = CroppedResearchLimit(adapter.context, current.request)
    adapter.context.services["cropped_research_controller"] = controller
    if tail:
        controller.force_finish("actual long-tail read boundary")
    adapter._writing_answer = True
    prompt: list[ChatCompletionMessage] = [UserMessage(content="original text " * 9000)]
    assert not adapter._cropped_large_writer_cache(
        prompt, current.tools, LLMFlow.ASV3_FINAL
    )


def test_small_first_writer_does_not_disable_later_writer_cache() -> None:
    adapter, _ledger, _selected, current = adapter_fixture()
    controller = CroppedResearchLimit(adapter.context, current.request)
    adapter.context.services["cropped_research_controller"] = controller
    controller.force_finish("actual long-tail read boundary")
    adapter._writing_answer = True
    small: list[ChatCompletionMessage] = [UserMessage(content="short original")]
    large: list[ChatCompletionMessage] = [UserMessage(content="original text " * 9000)]
    assert not adapter._cropped_large_writer_cache(
        small, current.tools, LLMFlow.ASV3_FINAL
    )
    assert not adapter._cropped_large_writer_cache(
        large, current.tools, LLMFlow.ASV3_FINAL
    )


def test_oversized_packet_preserves_first_physical_writer_cache_selection() -> None:
    from onyx.asv3.cropped_transport import CroppedInputOverflow

    adapter, _ledger, selected, current = adapter_fixture()
    controller = CroppedResearchLimit(adapter.context, current.request)
    adapter.context.services["cropped_research_controller"] = controller
    controller.force_finish("actual long-tail read boundary")
    adapter._writing_answer = True
    oversized: list[ChatCompletionMessage] = [UserMessage(content="x" * 240_001)]
    with pytest.raises(CroppedInputOverflow):
        adapter._cropped_large_writer_cache(
            oversized, current.tools, LLMFlow.ASV3_FINAL
        )
    assert not adapter._cropped_source_writer_started
    selected.invoke.assert_not_called()
    fitting: list[ChatCompletionMessage] = [
        UserMessage(content="original text " * 9000)
    ]
    assert adapter._cropped_large_writer_cache(
        fitting, current.tools, LLMFlow.ASV3_FINAL
    )
    assert not adapter._cropped_large_writer_cache(
        fitting, current.tools, LLMFlow.ASV3_FINAL
    )


@pytest.mark.parametrize("native_contract", [False, True])
@pytest.mark.parametrize("hard_tail", [False, True])
def test_first_writer_keeps_capacity_and_only_real_forced_native_contract_reduces_input_reserve(
    monkeypatch: pytest.MonkeyPatch, native_contract: bool, hard_tail: bool
) -> None:
    from tests.unit.onyx.llm.test_cropped_response_terminal_capture import (
        model as native_model,
    )

    adapter, ledger, generic, current = adapter_fixture()
    current.tools = [
        {
            "type": "function",
            "function": {
                "name": "read_evidence",
                "parameters": {
                    "type": "object",
                    "properties": {"citation": {"type": "integer"}},
                    "required": ["citation"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    monkeypatch.setattr(
        cost_module, "_rates", lambda *_: (0.000002, 0.00001, 0.0000001, 0.0000025)
    )
    budget = CroppedCostBudget(adapter.context, current.request)
    prior = budget.prepare(
        "gpt-6.1-sol", "openai", 1000, 1024, research=True, owner="root", enforce=False
    )
    budget.complete(
        prior,
        Usage(
            prompt_tokens=1000,
            completion_tokens=100,
            total_tokens=1100,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            cost=0.9596472,
        ),
    )
    controller = CroppedResearchLimit(adapter.context, current.request)
    adapter.context.services.update(
        cropped_cost_budget=budget, cropped_research_controller=controller
    )
    if hard_tail:
        controller.force_finish("actual long-tail read boundary")
    else:
        worker = adapter.context.child()
        worker.services["task_id"] = "host-owned-worker"
        controller.reserve_model("host-owned-worker", context=worker)
        for _ in range(7):
            controller.reserve_model("root", context=adapter.context)
        assert not controller.long_tail
        assert controller.admit_tool("resolve_source", {"query": "needed rule"}).allowed
    assert controller.budget_limited is hard_tail
    adapter._writing_answer = True
    prompt, tools, output = adapter._fit_native_decision(current)
    retained = copy.deepcopy(prompt)
    monkeypatch.setattr(adapter, "_input_cost", lambda *_args: 129429)
    invoked: list[dict[str, object]] = []

    def invoke(selected: LitellmLLM, **kwargs: object) -> ModelResponse:
        assert selected.cropped_native_cache_writes_disabled is hard_tail
        assert (selected.cropped_native_cache_write_contract is not None) is hard_tail
        invoked.append(kwargs)
        response = native_response("read_evidence", '{"citation":1}')
        response.usage = Usage(
            prompt_tokens=129429,
            completion_tokens=100,
            total_tokens=129529,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        return response

    monkeypatch.setattr(LitellmLLM, "invoke", invoke)
    # An arbitrary provider object claiming the boolean cannot relax the reserve.
    generic.cropped_native_cache_writes_disabled = True
    selected = native_model(capture=False) if native_contract else generic
    if not native_contract:
        response = native_response("read_evidence", '{"citation":1}')
        response.usage = Usage(
            prompt_tokens=129429,
            completion_tokens=100,
            total_tokens=129529,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        generic.invoke.return_value = response

    adapter._invoke_decision(
        current,
        prompt,
        tools,
        output,
        flow_override=LLMFlow.ASV3_FINAL,
        call_llm_override=selected,
    )
    if native_contract:
        assert len(invoked) == 1 and invoked[0]["max_tokens"] == 128000
        expected = retained[:-1] + [
            UserMessage(content=[TextContentPart(text=cast(str, retained[-1].content))])
        ]
        assert invoked[0]["prompt"] == expected and prompt == expected
    else:
        assert invoked == []
        assert generic.invoke.call_args.kwargs["max_tokens"] == 128000
    state = cast(dict[str, JsonValue], budget.export()["state"])
    calls = cast(list[dict[str, JsonValue]], state["calls"])
    disabled = native_contract and hard_tail
    assert calls[-1]["cache_writes_disabled"] is disabled
    assert calls[-1]["input_upper_tokens"] == 146468
    rate = 0.000002 if disabled else 0.0000025
    assert calls[-1]["reserved_usd"] == pytest.approx(146468 * rate + 128000 * 0.00001)
    assert calls[-1]["known_cost_usd"] == pytest.approx(0.259858)
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1, 2}


@pytest.mark.parametrize("cropped", [False, True])
@pytest.mark.parametrize("first_party", [False, True])
@pytest.mark.parametrize("already_guarded", [False, True])
def test_calibrated_counter_is_isolated_to_cropped_native_sol(
    cropped: bool, first_party: bool, already_guarded: bool
) -> None:
    import tiktoken

    context = RunContext(budget=SharedBudget(unlimited_execution=True))
    context.services["asv3_exp_cropped"] = cropped
    selected = LitellmLLM(
        api_key="test-key",
        model_provider="openai",
        model_name="gpt-6.1-sol",
        max_input_tokens=900000,
        api_base=None if first_party else "https://proxy.example/v1",
    )
    if already_guarded:
        selected = selected.with_tool_argument_stream_guard()

    def fallback(text: str) -> int:
        return len(text.encode("utf-8"))

    adapter = ResearchModel(
        selected, context, lean_native_mode=True, token_counter=fallback
    )
    text = 'Gümrük vergisinin tüm şartları ve istisnaları korunur. {"title":"<|endoftext|>"}'
    before = selected.config.model_dump()
    expected = (
        len(tiktoken.get_encoding("o200k_base").encode(text, disallowed_special=()))
        if cropped and first_party
        else fallback(text)
    )
    assert adapter._tokens(text) == expected
    assert selected.config.model_dump() == before
    assert selected._tool_argument_stream_guard is already_guarded
    if not (cropped and first_party):
        assert adapter.token_counter is fallback
