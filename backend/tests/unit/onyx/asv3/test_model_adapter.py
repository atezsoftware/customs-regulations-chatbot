import copy
import threading
from typing import Any
from unittest.mock import MagicMock

from pydantic import JsonValue

from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import RunContext, SharedBudget
from onyx.llm.interfaces import LLM, LLMConfig
from onyx.llm.model_response import Choice, Message, ModelResponse
from onyx.tracing.flows import LLMFlow


def test_real_adapter_limits_shared_provider_calls_across_worker_contexts() -> None:
    llm = MagicMock(spec=LLM)
    llm.config = LLMConfig(
        model_provider="openai",
        model_name="selected-model",
        temperature=0,
        max_input_tokens=100000,
    )
    context = RunContext(budget=SharedBudget(max_inflight_models=2))
    release = threading.Event()
    lock = threading.Lock()
    active = peak = 0
    responses: list[str] = []
    errors: list[Exception] = []

    def invoke(**_kwargs: Any) -> ModelResponse:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 2:
                release.set()
        assert release.wait(3)
        with lock:
            active -= 1
        return ModelResponse(
            id="response",
            created="0",
            choice=Choice(message=Message(content="original selected provider result")),
        )

    llm.invoke.side_effect = invoke

    def run() -> None:
        try:
            model = ResearchModel(llm, context.child())
            assert model.llm is llm
            responses.append(model.invoke_text("read", "task", LLMFlow.ASV3_RESEARCHER))
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=run) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(3)
    assert errors == []
    assert not any(thread.is_alive() for thread in threads)
    assert len(responses) == 6 and llm.invoke.call_count == 6
    assert peak == 2
    assert context.budget.snapshot()["decisions"] == 6
    assert llm.config.model_name == "selected-model"


def scripted_model(limit: int = 100000) -> MagicMock:
    llm = MagicMock(spec=LLM)
    llm.config = LLMConfig(
        model_provider="openai",
        model_name="selected-model",
        temperature=0,
        max_input_tokens=limit,
    )
    return llm


def tool_response(arguments: str, name: str = "read_evidence") -> ModelResponse:
    from onyx.llm.model_response import ChatCompletionMessageToolCall, FunctionCall

    return ModelResponse(
        id="result",
        created="0",
        choice=Choice(
            message=Message(
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id="call-1",
                        function=FunctionCall(name=name, arguments=arguments),
                    )
                ]
            )
        ),
    )


def test_vertex_tool_normalization_does_not_mutate_decision_validation_schema() -> None:
    from litellm.llms.vertex_ai.common_utils import _build_vertex_schema

    from onyx.asv3.models import HarnessView

    llm = scripted_model()
    tools: list[dict[str, JsonValue]] = [
        {
            "type": "function",
            "function": {
                "name": "search_corpus",
                "description": "search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "mode": {"type": "string", "enum": ["keyword", "hybrid"]},
                    },
                    "required": ["query"],
                    "$defs": {"unused": {"type": "string"}},
                    "additionalProperties": False,
                },
            },
        }
    ]
    original = copy.deepcopy(tools)

    def invoke(**kwargs: Any) -> ModelResponse:
        supplied = kwargs["tools"][0]["function"]["parameters"]
        normalized = _build_vertex_schema(supplied)
        assert normalized["properties"]["mode"]["type"] == "string"
        assert "$defs" not in supplied
        return tool_response('{"query":"royalti","mode":"keyword"}', "search_corpus")

    llm.invoke.side_effect = invoke
    decision = ResearchModel(llm, RunContext()).decide(
        HarnessView(
            request="Royalti koşullarını araştır",
            questions=[],
            facts=[],
            receipts=[],
            evidence=[],
            tools=tools,
        )
    )
    assert decision.calls[0].arguments["mode"] == "keyword"
    assert llm.invoke.call_count == 1
    assert tools == original


def original_state() -> tuple[RunContext, Any, list[dict[str, Any]]]:
    from onyx.asv3.evidence import EvidenceLedger
    from onyx.asv3.models import EvidenceItem
    from onyx.asv3.registry import CapabilityRegistry, build_core_specs

    ledger = EvidenceLedger()
    context = RunContext(services={"evidence": ledger})
    ledger.add(
        [
            EvidenceItem(
                source_id="law",
                chunk_id="143",
                text="The original complete operative text.",
            )
        ],
        context,
    )
    registry = CapabilityRegistry()
    for spec in build_core_specs(registry, ledger, lambda: {}):
        registry.register(spec)
    return context, ledger, registry.definitions(context)


def test_tool_json_and_schema_get_one_selected_model_repair_before_dispatch() -> None:
    from onyx.asv3.models import HarnessView

    context, ledger, tools = original_state()
    llm = scripted_model()
    llm.invoke.side_effect = [
        tool_response('{"citation":"1"}'),
        tool_response('{"citation":1}'),
    ]
    model = ResearchModel(llm, context)
    view = HarnessView(
        request="Read the complete operative provision",
        questions=[],
        facts=[],
        receipts=[],
        evidence=[],
        tools=tools,
    )
    before = ledger.export()
    decision = model.decide(view)
    assert len(decision.calls) == 1 and decision.calls[0].arguments == {"citation": 1}
    assert llm.invoke.call_count == 2
    assert context.budget.snapshot()["decisions"] == 1
    assert ledger.export() == before
    correction = (
        llm.invoke.call_args_list[1].kwargs["prompt"][1].content[0].text
        if isinstance(llm.invoke.call_args_list[1].kwargs["prompt"][1].content, list)
        else llm.invoke.call_args_list[1].kwargs["prompt"][1].content
    )
    assert "format_repair" in correction and "Do not invent" in correction
    assert llm.config.model_name == "selected-model"


def test_repeated_malformed_tool_call_fails_after_one_repair_and_keeps_sources() -> (
    None
):
    import pytest

    from onyx.asv3.models import HarnessView

    context, ledger, tools = original_state()
    llm = scripted_model()
    llm.invoke.side_effect = [
        tool_response('{"citation":'),
        tool_response('{"citation":'),
    ]
    model = ResearchModel(llm, context)
    before = ledger.export()
    with pytest.raises(ValueError):
        model.decide(
            HarnessView(
                request="read",
                questions=[],
                facts=[],
                receipts=[],
                evidence=[],
                tools=tools,
            )
        )
    assert llm.invoke.call_count == 2
    assert ledger.export() == before
    assert context.budget.model_slots.acquire(blocking=False)
    context.budget.model_slots.release()


def test_language_json_repair_uses_original_request_and_same_selected_model() -> None:
    from onyx.asv3.llm_adapter import parse_json_object

    llm = scripted_model()
    llm.invoke.side_effect = [
        ModelResponse(
            id="bad", created="0", choice=Choice(message=Message(content="```{broken"))
        ),
        ModelResponse(
            id="valid",
            created="0",
            choice=Choice(
                message=Message(content='{"language":"tr","notifications":{}}')
            ),
        ),
    ]
    context = RunContext()
    model = ResearchModel(llm, context)
    result = model.invoke_text(
        "Identify response language and return JSON",
        "Yanıt Türkçe olsun",
        LLMFlow.ASV3_LANGUAGE,
    )
    assert parse_json_object(result)["language"] == "tr"
    assert llm.invoke.call_count == 2
    assert context.budget.snapshot()["decisions"] == 2
    assert llm.invoke.call_args.kwargs["prompt"][1].content == "Yanıt Türkçe olsun"


def test_strict_json_rejects_duplicate_keys_nonfinite_values_and_bad_fences() -> None:
    import pytest

    from onyx.asv3.llm_adapter import parse_json_object

    for text in ('{"citation":1,"citation":2}', '{"value":NaN}', "```{}", "[1]"):
        with pytest.raises(ValueError):
            parse_json_object(text)


def test_pathological_context_fits_selected_limit_and_originals_remain_reopenable() -> (
    None
):
    import json

    from onyx.asv3.models import EvidenceItem, HarnessView

    context, ledger, tools = original_state()
    original = "FULL_ORIGINAL_MARKER" + "Text " * 30000
    ledger.add(
        [EvidenceItem(source_id="another-law", chunk_id="article", text=original)],
        context,
    )
    for n in range(20):
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": f"large_schema_{n}",
                    "description": "Description " * 5000,
                    "parameters": {"type": "object"},
                },
            }
        )
    llm = scripted_model(40000)
    llm.invoke.return_value = ModelResponse(
        id="valid",
        created="0",
        choice=Choice(message=Message(content="Read source evidence next.")),
    )
    history = "EARLY_HISTORY_CONSTRAINT " + "history " * 2000
    model = ResearchModel(llm, context, token_counter=len, history=history)
    view = HarnessView(
        request="Cover all three alternatives.",
        questions=["free repair", "paid repair", "replacement"],
        facts=["no standard exchange permission"],
        receipts=[],
        evidence=[
            {"citation": 1, "source_id": "law", "text": "first source"},
            {"citation": 2, "source_id": "another-law", "text": original},
        ],
        tools=tools,
    )
    model.decide(view)
    request = llm.invoke.call_args.kwargs
    assert (
        model._input_cost(request["prompt"], request["tools"]) + request["max_tokens"]
        <= llm.config.max_input_tokens
    )
    payload = json.loads(request["prompt"][1].content[0].text)
    assert (
        payload["request"] == view.request
        and payload["questions"] == view.questions
        and payload["facts"] == view.facts
    )
    assert payload["conversation"] == history
    assert payload["evidence"][1]["truncated"] is True
    assert payload["capability_context"]["reopen"] == "discover_tools"
    assert {tool["function"]["name"] for tool in request["tools"]} >= {
        "read_evidence",
        "discover_tools",
    }
    original_item = ledger.get(2)
    assert original_item is not None and original_item.text == original
    assert view.evidence[1]["text"] == original


def test_irreducible_question_and_complete_cited_evidence_fail_before_provider() -> (
    None
):
    import json

    import pytest

    from onyx.asv3.models import RunStopped

    context, ledger, tools = original_state()
    llm = scripted_model(12000)
    model = ResearchModel(llm, context)
    from onyx.asv3.harness import Harness
    from onyx.asv3.registry import CapabilityRegistry

    question = "QUESTION " * 10000
    harness = Harness(
        request=question,
        context=context,
        registry=CapabilityRegistry(),
        decide=model.decide,
    )
    view = harness.view()
    assert view.request == question
    with pytest.raises(RunStopped, match="scenario"):
        model.decide(view)
    text = "Complete law " * 10000
    with pytest.raises(RunStopped, match="Complete cited evidence"):
        model.invoke_text(
            "Verify the cited rule",
            json.dumps(
                {
                    "claim": "Rule [1]",
                    "scenario": "facts",
                    "evidence": json.dumps([{"citation": 1, "text": text}]),
                }
            ),
            LLMFlow.ASV3_VERIFICATION,
        )
    assert llm.invoke.call_count == 0
    assert ledger.get(1) is not None


def test_final_review_removes_only_uncited_supplemental_context() -> None:
    import json

    llm = scripted_model(12000)
    llm.invoke.return_value = ModelResponse(
        id="valid",
        created="0",
        choice=Choice(message=Message(content='{"status":"supported"}')),
    )
    model = ResearchModel(llm, RunContext())
    original = "Complete operative paragraph with all exceptions."
    model.invoke_text(
        "Verify original cited text",
        json.dumps(
            {
                "claim": "Rule [1]",
                "scenario": "facts",
                "evidence": json.dumps(
                    [
                        {"citation": 1, "text": original, "truncated": False},
                        {"citation": 2, "text": "Uncited " * 10000},
                    ]
                ),
            }
        ),
        LLMFlow.ASV3_VERIFICATION,
    )
    payload = json.loads(llm.invoke.call_args.kwargs["prompt"][1].content)
    assert json.loads(payload["evidence"]) == [
        {"citation": 1, "text": original, "truncated": False}
    ]
    assert payload["supplemental_evidence_omitted"] is True


def test_schema_repair_cannot_drop_an_already_valid_parallel_action() -> None:
    import pytest

    from onyx.asv3.models import HarnessView

    context, _ledger, tools = original_state()
    llm = scripted_model()
    first = tool_response('{"citation":1}')
    invalid = tool_response('{"citation":"2"}')
    assert invalid.choice.message.tool_calls and first.choice.message.tool_calls
    invalid.choice.message.tool_calls[0].id = "call-2"
    first.choice.message.tool_calls += invalid.choice.message.tool_calls
    repair = tool_response('{"citation":2}')
    assert repair.choice.message.tool_calls
    repair.choice.message.tool_calls[0].id = "call-2"
    llm.invoke.side_effect = [first, repair]
    model = ResearchModel(llm, context)
    with pytest.raises(ValueError, match="action count"):
        model.decide(
            HarnessView(
                request="Read both",
                questions=[],
                facts=[],
                receipts=[],
                evidence=[],
                tools=tools,
            )
        )
    assert llm.invoke.call_count == 2


def test_selected_tokenizer_preserves_nonascii_cited_law_that_fits_actual_limit() -> (
    None
):
    import json

    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    llm = scripted_model(12000)
    llm.invoke.return_value = ModelResponse(
        id="valid",
        created="0",
        choice=Choice(message=Message(content='{"status":"supported"}')),
    )
    model = ResearchModel(
        llm, RunContext(), token_counter=lambda text: len(encoding.encode(text))
    )
    original = "Tamir işlemi ücretsizdir. Üretim hatası kanıtlanır. " * 300
    assert len(original.encode("utf-8")) > llm.config.max_input_tokens
    model.invoke_text(
        "Verify original cited text",
        json.dumps(
            {
                "claim": "Tamir koşulları [1]",
                "scenario": "Ücretsiz tamir",
                "evidence": json.dumps(
                    [{"citation": 1, "text": original, "truncated": False}],
                    ensure_ascii=False,
                ),
            },
            ensure_ascii=False,
        ),
        LLMFlow.ASV3_VERIFICATION,
    )
    request = llm.invoke.call_args.kwargs
    assert (
        model._input_cost(request["prompt"], request["tools"] or [])
        + request["max_tokens"]
        <= llm.config.max_input_tokens
    )
    payload = json.loads(request["prompt"][1].content)
    assert json.loads(payload["evidence"])[0]["text"] == original
    assert llm.invoke.call_count == 1


def test_invalid_token_counter_fails_before_provider_and_unsupported_falls_back() -> (
    None
):
    import pytest

    llm = scripted_model()
    for value in (-1, True):
        model = ResearchModel(llm, RunContext(), token_counter=lambda _text: value)
        with pytest.raises(ValueError, match="invalid count"):
            model.invoke_text("instruction", "question", LLMFlow.ASV3_FINAL)
    assert llm.invoke.call_count == 0

    def unsupported(_text: str) -> int:
        raise NotImplementedError("No provider tokenizer")

    model = ResearchModel(llm, RunContext(), token_counter=unsupported)
    assert model._tokens("Ücretsiz") == len("Ücretsiz".encode("utf-8"))
