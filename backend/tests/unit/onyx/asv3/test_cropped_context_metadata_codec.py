"""Cropped context tables preserve actual originals and unresolved tool state."""

import copy
import hashlib
import json
from typing import cast
from unittest.mock import patch

import pytest
import tiktoken
from pydantic import JsonValue

import onyx.asv3.cheap_metadata_transport as codec
from onyx.asv3.cropped_handoff_transport import expand_cropped_handoff_transport
from onyx.asv3.cropped_transport import serialize_cropped_payload
from onyx.asv3.llm_adapter import ResearchModel, _NativeOriginalCatalogue
from onyx.llm.models import ChatCompletionMessage, UserMessage
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.asv3.test_cheap_metadata_adapter import packet
from tests.unit.onyx.asv3.test_native_model_adapter import model

ENCODER = tiktoken.get_encoding("o200k_base")
OLD_EXTRA_FIELDS = ("research_tasks", "repair_research_observations")


def tokens(text: str) -> int:
    return len(ENCODER.encode(text, disallowed_special=()))


def full_payload() -> dict[str, JsonValue]:
    _, _, encoded = packet()
    payload = json.loads(json.dumps(codec.expand_cheap_payload(encoded)))
    status: dict[str, JsonValue] = {
        "call_id": "actual_failed_lookup",
        "method": "read_named_provision",
        "arguments": {"article_no": "15", "paragraph": "3"},
        "status": "not_found",
        "not_read": True,
        "absence_proven": False,
        "has_more": True,
        "next_position": 47,
        "error": "Actual lookup remains unresolved. " * 30,
        "unknown_contract_field": {
            "value_ref": {"layout_ref": [0, ["literal data"]]},
            "null": None,
            "empty": {},
            "false": False,
            "integer": 1,
            "float": 1.0,
        },
    }
    gap: dict[str, JsonValue] = {
        "instruction": "Retain supported conditions; disclose unresolved scope.",
        "cropped_evidence_checks": [status],
        "absence_proven": False,
    }
    payload["publication_gap"] = gap
    payload["cropped_completion"] = {
        "full_question": "Preserve exact original conditions and pending lookup.",
        "facts": ["Case date remains unknown."],
        "publication_gap": copy.deepcopy(gap),
        "pending_operations": [copy.deepcopy(status)],
    }
    payload["session_research"] = {
        "status": "retained_navigation_not_current_original_delivery",
        "requests": ["The full prior user scenario remains available."],
        "failed_lookup": copy.deepcopy(status),
        "source_gaps": [copy.deepcopy(status)],
        "prior_citation_map": [
            {"old": index, "current": index + 10, "unread": True} for index in range(30)
        ],
    }
    return payload


def test_optional_context_fields_exact_roundtrip_and_actual_literal_delivery() -> None:
    context, ledger, _ = packet()
    context.services.pop("asv3_exp_cheap")
    context.services["asv3_exp_cropped"] = True
    original = full_payload()
    before = copy.deepcopy(original)
    ordinary = codec.serialize_cheap_payload(original, tokens)
    text = serialize_cropped_payload(original, tokens)
    wire = json.loads(text)
    restored = expand_cropped_handoff_transport(wire)
    assert original == before == restored
    for field in ("cropped_completion", "session_research", "publication_gap"):
        assert json.dumps(restored[field], ensure_ascii=False) == json.dumps(
            original[field], ensure_ascii=False
        )
    assert tokens(text) < tokens(ordinary)
    assert {"cropped_completion", "session_research"}.issubset(
        wire["original_metadata_codec"]["payload_fields"]
    )
    originals = cast(list[dict[str, JsonValue]], original["original_evidence"])
    for row in originals:
        literal = cast(str, row["text"])
        assert literal in text
        item = ledger.get(cast(int, row["citation"]))
        assert item is not None
        assert hashlib.sha256(item.text.encode()).hexdigest() == item.text_hash
    adapter = ResearchModel(
        model(limit=1_000_000), context, lean_native_mode=True, token_counter=tokens
    )
    prompt: list[ChatCompletionMessage] = [UserMessage(content=text)]
    bound = _NativeOriginalCatalogue.bind(prompt, cheap_metadata=True)
    adapter._invoke_once(
        prompt,
        [],
        LLMFlow.ASV3_ANSWER_REPAIR,
        max_tokens=100,
        research=False,
        structured=False,
        native_catalogue=bound,
    )
    assert ledger.completely_delivered(adapter.last_call_id or "") == set(range(1, 13))


def test_default_cheap_wire_and_existing_extra_fields_are_byte_identical() -> None:
    payload = full_payload()
    payload["research_tasks"] = [{"status": "working", "next_position": 47}]
    for extra in ((), OLD_EXTRA_FIELDS):
        current = codec.serialize_cheap_payload(payload, tokens, extra_fields=extra)
        with patch.object(codec, "_EXTRA_PAYLOAD_FIELDS", OLD_EXTRA_FIELDS):
            baseline = codec.serialize_cheap_payload(
                payload, tokens, extra_fields=extra
            )
        assert current == baseline
        assert (
            json.loads(current)["cropped_completion"] == payload["cropped_completion"]
        )
        assert json.loads(current)["session_research"] == payload["session_research"]


def test_small_or_catalogue_free_context_is_not_rewritten() -> None:
    payload: dict[str, JsonValue] = {
        "request": "q",
        "cropped_completion": {"publication_gap": None},
        "session_research": {"status": "unknown", "next_position": None},
    }
    original = copy.deepcopy(payload)
    text = serialize_cropped_payload(payload, tokens)
    assert json.loads(text) == original == payload
    assert "original_metadata_codec" not in text


@pytest.mark.parametrize("damage", ["unknown_field", "invalid_value", "unused_value"])
def test_context_table_tampering_fails_before_native_binding(damage: str) -> None:
    wire = json.loads(serialize_cropped_payload(full_payload(), tokens))
    if damage == "unknown_field":
        wire["original_metadata_codec"]["payload_fields"].append("unapproved_context")
    elif damage == "invalid_value":
        wire["session_research"] = {"value_ref": True}
    else:
        wire["original_metadata_values"].append("Unused hidden value")
    with pytest.raises(ValueError):
        _NativeOriginalCatalogue.bind(
            [UserMessage(content=json.dumps(wire, ensure_ascii=False))],
            cheap_metadata=True,
        )
