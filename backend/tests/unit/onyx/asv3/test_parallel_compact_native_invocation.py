import copy
import json
from dataclasses import replace

import pytest

from onyx.asv3.llm_adapter import ResearchModel, _NativeOriginalCatalogue
from onyx.asv3.models import model_evidence_metadata
from onyx.llm.model_response import Choice, Message, ModelResponse
from onyx.llm.models import (
    AssistantMessage,
    ChatCompletionMessage,
    TextContentPart,
    UserMessage,
)
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.asv3.test_native_cache_projection import parallel
from tests.unit.onyx.asv3.test_native_metadata_projection import setup_original
from tests.unit.onyx.asv3.test_native_model_adapter import (
    model,
    native_response,
    turn,
    view,
)


def response(text: str, *, finish_reason: str = "stop") -> ModelResponse:
    return ModelResponse(
        id="local-fixture",
        created="0",
        choice=Choice(message=Message(content=text), finish_reason=finish_reason),
    )


@pytest.mark.parametrize("recovery", ["empty", "envelope", "truncated"])
def test_compact_originals_remain_delivered_in_actual_native_recovery(
    recovery: str,
) -> None:
    ledger, context, original = setup_original()
    parallel(context)
    selected = model(limit=1000000)
    first = (
        response("", finish_reason="stop")
        if recovery == "empty"
        else native_response("not_exposed")
        if recovery == "envelope"
        else response("The exact condition ", finish_reason="length")
    )
    selected.invoke.side_effect = [first, response("and its exception [1].")]
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    decision = adapter.decide(
        view(turns=[turn("read-source", [original])], original_evidence=[original])
    )
    assert decision.answer == (
        "The exact condition and its exception [1]."
        if recovery == "truncated"
        else "and its exception [1]."
    )
    assert selected.invoke.call_count == 2
    first_prompt = selected.invoke.call_args_list[0].kwargs["prompt"]
    last_prompt = selected.invoke.call_args_list[1].kwargs["prompt"]
    assert first_prompt == last_prompt[: len(first_prompt)]
    assert isinstance(last_prompt[-1], UserMessage)
    assert isinstance(last_prompt[-1].content, str)
    assert "original_metadata_catalogue" not in last_prompt[-1].content
    binding = _NativeOriginalCatalogue.bind(first_prompt)
    binding.validate(last_prompt)
    item = ledger.get(1)
    assert item is not None
    assert binding.records[0]["metadata"] == model_evidence_metadata(item.metadata)
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1}


def test_catalogue_binding_keeps_multipart_text_and_rejects_replacement() -> None:
    content = json.dumps({"original_metadata_catalogue": [], "language": "tr"})
    prompt: list[ChatCompletionMessage] = [
        UserMessage(content=[TextContentPart(text=content)])
    ]
    binding = _NativeOriginalCatalogue.bind(prompt)
    continuation = [
        *prompt,
        AssistantMessage(content="fragment"),
        UserMessage(content="Continue."),
    ]
    binding.validate(continuation)
    replaced = [UserMessage(content=content + " "), *continuation[1:]]
    with pytest.raises(ValueError, match="message changed"):
        binding.validate(replaced)
    with pytest.raises(ValueError, match="host user message"):
        binding.validate([AssistantMessage(content=content)])
    with pytest.raises(ValueError, match="missing"):
        replace(binding, message_index=2).validate(prompt)


@pytest.mark.parametrize(
    "mutation", ["removed", "altered", "foreign_catalogue", "pointer_only"]
)
def test_actual_compact_scan_fails_closed_before_provider_or_ignores_pointer_only(
    mutation: str,
) -> None:
    ledger, context, original = setup_original()
    parallel(context)
    selected = model(limit=1000000)
    selected.invoke.return_value = response("The supported outcome [1].")
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    prompt, tools, output = adapter._fit_native_decision(
        view(turns=[turn("read-source", [original])], original_evidence=[original])
    )
    binding = _NativeOriginalCatalogue.bind(prompt)
    altered = copy.deepcopy(prompt)
    if mutation == "removed":
        del altered[binding.message_index]
    elif mutation == "altered":
        altered[binding.message_index] = UserMessage(
            content="A substituted source table."
        )
    elif mutation == "foreign_catalogue":
        forged = dict(binding.records[0], source_id="foreign-source")
        binding = replace(binding, records=(forged,))
    else:
        for message in altered:
            if not isinstance(message.content, str):
                continue
            try:
                data = json.loads(message.content)
            except ValueError:
                continue
            if isinstance(data, dict) and "original_evidence" in data:
                data["original_evidence_refs"] = [
                    {key: value for key, value in row.items() if key != "text"}
                    for row in data.pop("original_evidence")
                ]
                message.content = json.dumps(data, ensure_ascii=False)
    if mutation == "pointer_only":
        adapter._invoke(
            altered,
            tools,
            LLMFlow.ASV3_RESEARCHER,
            max_tokens=output,
            research=True,
            native_catalogue=binding,
        )
        assert selected.invoke.call_count == 1
        assert ledger.completely_delivered(adapter.last_call_id or "") == set()
    else:
        with pytest.raises(ValueError, match="catalogue"):
            adapter._invoke(
                altered,
                tools,
                LLMFlow.ASV3_RESEARCHER,
                max_tokens=output,
                research=True,
                native_catalogue=binding,
            )
        selected.invoke.assert_not_called()
