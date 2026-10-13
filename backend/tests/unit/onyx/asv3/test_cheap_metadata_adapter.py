"""Compressed metadata must bind actual originals before a provider call."""

import copy
import json
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.cheap_metadata_transport import encode_cheap_metadata
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import (
    ResearchModel,
    SourceRequirementInventory,
    _NativeOriginalCatalogue,
)
from onyx.asv3.models import RunContext, RunStopped, SharedBudget
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.llm.model_response import Choice, Message, ModelResponse, Usage
from onyx.llm.models import ChatCompletionMessage, UserMessage
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.asv3.test_native_model_adapter import model
from tests.unit.onyx.asv3.test_shared_originals import full_record, original


def packet() -> tuple[RunContext, EvidenceLedger, dict[str, JsonValue]]:
    context = RunContext(budget=SharedBudget(unlimited_execution=True))
    context.services.update(
        asv3_exp_cheap=True,
        asv3_workflow_variant=ASV3_TUNED_VARIANT,
        research_profile="normal",
    )
    ledger = EvidenceLedger()
    ledger.add(
        [
            original(
                f"chunk-{number}",
                f"Material condition {number} AND its exception.",
                metadata={"title": "Long recurring source title " * 20},
            )
            for number in range(12)
        ],
        context,
    )
    context.services["evidence"] = ledger
    records = [full_record(ledger, number) for number in ledger.citation_numbers()]
    fields = ("citation", "source_id", "chunk_id", "text_hash", "metadata")
    literals = []
    for record in records:
        row = {key: value for key, value in record.items() if key not in fields[1:]}
        row["identity_ref"] = record["citation"]
        text = cast(str, record["text"])
        row.update(start_char=0, end_char=len(text), source_text_length=len(text))
        literals.append(row)
    payload: dict[str, JsonValue] = {
        "scenario": "Determine the conditions and exceptions.",
        "original_metadata_catalogue": cast(
            JsonValue, [{key: record[key] for key in fields} for record in records]
        ),
        "original_evidence": cast(JsonValue, literals),
        "required_evidence_numbers": cast(JsonValue, ledger.citation_numbers()),
    }
    return context, ledger, encode_cheap_metadata(payload, len)


def structured(text: str) -> ModelResponse:
    return ModelResponse(
        id="test", created="0", choice=Choice(message=Message(content=text))
    )


def test_encoded_message_is_only_accepted_in_explicit_cheap_binding() -> None:
    _, _, payload = packet()
    assert "original_metadata_codec" in payload
    prompt: list[ChatCompletionMessage] = [
        UserMessage(content=json.dumps(payload, ensure_ascii=False))
    ]
    with pytest.raises(ValueError):
        _NativeOriginalCatalogue.bind(prompt)
    bound = _NativeOriginalCatalogue.bind(prompt, cheap_metadata=True)
    assert len(bound.records) == 12
    bound.validate(prompt)
    prompt[0] = UserMessage(content=json.dumps({**payload, "scenario": "Changed"}))
    with pytest.raises(ValueError, match="message changed"):
        bound.validate(prompt)


@pytest.mark.parametrize("format_retry", [False, True])
def test_independent_call_records_all_literals_including_format_retry(
    format_retry: bool,
) -> None:
    context, ledger, payload = packet()
    selected = model(limit=1_000_000)
    valid = structured(
        json.dumps(
            {"examined_citations": ledger.citation_numbers(), "requirements": []}
        )
    )
    selected.invoke.side_effect = [structured("{}"), valid] if format_retry else [valid]
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    answer = adapter.invoke_text(
        "Inventory complete sources without a draft.",
        json.dumps(payload, ensure_ascii=False),
        LLMFlow.ASV3_SOURCE_INVENTORY,
        response_model_override=SourceRequirementInventory,
    )
    assert SourceRequirementInventory.model_validate_json(answer).requirements == []
    assert selected.invoke.call_count == (2 if format_retry else 1)
    assert ledger.completely_delivered(adapter.last_call_id or "") == set(range(1, 13))
    assert (
        ledger.delivery_flow(adapter.last_call_id or "")
        == LLMFlow.ASV3_SOURCE_INVENTORY.value
    )
    if format_retry:
        assert len(selected.invoke.call_args.kwargs["prompt"]) == 3


def test_context_overflow_cannot_prune_complete_condition_originals() -> None:
    context, ledger, payload = packet()
    selected = model(limit=100)
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    before = copy.deepcopy(payload)
    with pytest.raises(RunStopped):
        adapter.invoke_text(
            "Inventory complete sources.",
            json.dumps(payload),
            LLMFlow.ASV3_SOURCE_INVENTORY,
            response_model_override=SourceRequirementInventory,
        )
    assert payload == before
    assert adapter.last_call_id is None
    assert ledger.completely_delivered("absent") == set()
    selected.invoke.assert_not_called()


@pytest.mark.parametrize("cheap", [False, True])
def test_stop_normalization_cannot_repeat_the_same_exhausted_cheap_output_cap(
    cheap: bool,
) -> None:
    context, ledger, payload = packet()
    context.services["asv3_exp_cheap"] = cheap
    selected = model(limit=1_000_000)
    exhausted = structured('{"examined_citations": [')
    exhausted.choice.finish_reason = "stop"
    exhausted.usage = Usage(
        completion_tokens=20,
        prompt_tokens=100,
        total_tokens=120,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    valid = structured(
        json.dumps(
            {"examined_citations": ledger.citation_numbers(), "requirements": []}
        )
    )
    selected.invoke.side_effect = [exhausted, valid]
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    # The protected path uses its ordinary literal payload, without Cheap's codec.
    if not cheap:
        from onyx.asv3.cheap_metadata_transport import expand_cheap_payload

        payload = expand_cheap_payload(payload)
        payload.pop("original_metadata_catalogue")
    adapter.invoke_text(
        "Inventory complete sources.",
        json.dumps(payload, ensure_ascii=False),
        LLMFlow.ASV3_SOURCE_INVENTORY,
        max_tokens=20,
        response_model_override=SourceRequirementInventory,
    )
    assert [call.kwargs["max_tokens"] for call in selected.invoke.call_args_list] == [
        20,
        40 if cheap else 20,
    ]
    assert adapter.last_response_truncated is False
