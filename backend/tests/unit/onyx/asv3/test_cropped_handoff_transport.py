"""Actual handoff objects retain source delivery, hashes and typed observations."""

import copy
import json
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.cheap_metadata_transport import (
    encode_cheap_metadata,
    expand_cheap_payload,
    serialize_cheap_payload,
)
from onyx.asv3.cropped_handoff_transport import (
    canonical_cropped_handoff_payload,
    encode_cropped_handoff_transport,
    expand_cropped_handoff_transport,
)
from onyx.asv3.cropped_transport import serialize_cropped_payload
from onyx.asv3.cropped_worker_handoff import build_worker_handoff, expand_worker_handoff
from onyx.asv3.llm_adapter import ResearchModel, _NativeOriginalCatalogue
from onyx.asv3.source_metadata_transport import share_source_metadata
from onyx.llm.models import ChatCompletionMessage, UserMessage
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.asv3.test_cheap_metadata_adapter import packet
from tests.unit.onyx.asv3.test_cropped_worker_handoff import build
from tests.unit.onyx.asv3.test_cropped_worker_handoff import payload as worker_payload
from tests.unit.onyx.asv3.test_native_model_adapter import model
from tests.unit.onyx.asv3.test_shared_originals import full_record

HEADER = "cropped_canonical_fields_transport"
REFERENCE = "canonical_fields_ref"


def handoff_packet() -> dict[str, JsonValue]:
    handoff, _, ledger, *_ = build()
    record = full_record(ledger, 1)
    return {
        "request": "Read every condition and preserve the unresolved exception.",
        "facts": ["Known user fact; no source absence was established."],
        "original_metadata_catalogue": [
            {
                key: record[key]
                for key in (
                    "citation",
                    "source_id",
                    "chunk_id",
                    "text_hash",
                    "metadata",
                )
            }
        ],
        "original_evidence": [record],
        "research_tasks": [
            {
                "task": "Exact child scope",
                "outcome": {"data": {"cropped_worker_handoff": handoff}},
            }
        ],
        "repair_research_observations": [
            {"record": copy.deepcopy(record), "not_read": True, "absence_proven": False}
        ],
    }


def first_reference(node: JsonValue) -> dict[str, JsonValue]:
    if isinstance(node, dict):
        if REFERENCE in node:
            return node
        children = node.values()
    elif isinstance(node, list):
        children = node
    else:
        raise LookupError
    for child in children:
        try:
            return first_reference(child)
        except LookupError:
            continue
    raise LookupError


def test_real_worker_handoff_hashes_and_actual_state_survive_both_codecs() -> None:
    handoff, context, ledger, *_ = build()
    payload = handoff_packet()
    original = copy.deepcopy(payload)
    encoded = encode_cropped_handoff_transport(payload)
    assert HEADER in encoded
    wire = encode_cheap_metadata(
        encoded, len, extra_fields=("research_tasks", "repair_research_observations")
    )
    decoded = expand_cropped_handoff_transport(wire)
    assert decoded == original
    assert payload == original
    tasks = cast(list[dict[str, JsonValue]], decoded["research_tasks"])
    actual = cast(
        dict[str, JsonValue],
        cast(
            dict[str, JsonValue],
            cast(dict[str, JsonValue], tasks[0]["outcome"])["data"],
        )["cropped_worker_handoff"],
    )
    assert actual == handoff
    assert expand_worker_handoff(
        actual, ledger=ledger, context=context
    ) == expand_worker_handoff(handoff, ledger=ledger, context=context)
    assert decoded["original_evidence"] == payload["original_evidence"]


@pytest.mark.parametrize("compact", [False, True])
def test_native_record_json_string_reconstructs_with_its_original_key_order(
    compact: bool,
) -> None:
    context, ledger, result, turns, state, memory, draft = worker_payload()
    record = full_record(ledger, 1)
    literal = json.dumps(
        {"not_read": False, "original_evidence": [record], "has_more": True},
        ensure_ascii=False,
        separators=(",", ":") if compact else None,
    )
    turns[0].results[0].content = literal
    handoff = build_worker_handoff(
        context=context,
        result=result,
        ledger=ledger,
        turns=turns,
        research_state=state,
        working_memory=memory,
        draft=draft,
    )
    payload = handoff_packet()
    payload["research_tasks"] = [{"handoff": handoff}]
    transported = encode_cropped_handoff_transport(payload)
    decoded = expand_cropped_handoff_transport(
        encode_cheap_metadata(
            transported,
            len,
            extra_fields=("research_tasks", "repair_research_observations"),
        )
    )
    tasks = cast(list[dict[str, JsonValue]], decoded["research_tasks"])
    actual = cast(dict[str, JsonValue], tasks[0]["handoff"])
    assert json.dumps(actual, ensure_ascii=False) == json.dumps(
        handoff, ensure_ascii=False
    )
    restored = expand_worker_handoff(actual, ledger=ledger, context=context)
    assert restored == expand_worker_handoff(handoff, ledger=ledger, context=context)
    restored_turn = cast(list[dict[str, JsonValue]], restored["turns"])[0]
    restored_result = cast(list[dict[str, JsonValue]], restored_turn["results"])[0]
    assert restored_result["content"] == literal


def test_equal_metadata_with_different_nested_key_order_stays_literal() -> None:
    payload = handoff_packet()
    row = cast(list[dict[str, JsonValue]], payload["original_metadata_catalogue"])[0]
    metadata = cast(dict[str, JsonValue], row["metadata"])
    metadata["ordered"] = {"first": 1, "second": 2}
    different = copy.deepcopy(row)
    cast(dict[str, JsonValue], different["metadata"])["ordered"] = {
        "second": 2,
        "first": 1,
    }
    payload["research_tasks"] = [{"row": different}]
    before = json.dumps(payload["research_tasks"], ensure_ascii=False)
    encoded = encode_cropped_handoff_transport(payload)
    reference = first_reference(encoded["research_tasks"])
    assert "metadata" in reference
    decoded = expand_cropped_handoff_transport(encoded)
    assert json.dumps(decoded["research_tasks"], ensure_ascii=False) == before


def test_actual_native_delivery_still_requires_the_inline_originals() -> None:
    context, ledger, encoded = packet()
    context.services.pop("asv3_exp_cheap")
    context.services["asv3_exp_cropped"] = True
    payload = cast(
        dict[str, JsonValue], json.loads(json.dumps(expand_cheap_payload(encoded)))
    )
    payload["research_tasks"] = [
        {"locators": copy.deepcopy(payload["original_metadata_catalogue"])}
    ]
    # Exercise actual same-message decoding even if a small fixture prefers old wire.
    transported = encode_cheap_metadata(
        encode_cropped_handoff_transport(payload), len, extra_fields=("research_tasks",)
    )
    raw = json.dumps(transported, ensure_ascii=False)
    assert expand_cropped_handoff_transport(transported) == payload
    prompt: list[ChatCompletionMessage] = [UserMessage(content=raw)]
    bound = _NativeOriginalCatalogue.bind(prompt, cheap_metadata=True)
    adapter = ResearchModel(model(limit=1_000_000), context, lean_native_mode=True)
    adapter._invoke_once(
        prompt,
        [],
        LLMFlow.ASV3_FINAL,
        max_tokens=100,
        research=False,
        structured=False,
        native_catalogue=bound,
    )
    assert ledger.completely_delivered(adapter.last_call_id or "") == set(range(1, 13))


@pytest.mark.parametrize(
    "value", [True, 1, 1.0, None, [False, 0, "0"], {"unknown": [None, False]}]
)
def test_unknown_metadata_types_and_mismatched_values_are_preserved(
    value: JsonValue,
) -> None:
    payload = handoff_packet()
    rows = cast(list[dict[str, JsonValue]], payload["original_metadata_catalogue"])
    cast(dict[str, JsonValue], rows[0]["metadata"])["unknown_field"] = value
    payload["research_tasks"] = [{"identity": copy.deepcopy(rows[0])}]
    before = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    decoded = expand_cropped_handoff_transport(
        encode_cropped_handoff_transport(payload)
    )
    assert json.dumps(decoded, ensure_ascii=False, sort_keys=True) == before


@pytest.mark.parametrize("actual", [1, 1.0])
def test_python_numeric_equality_does_not_substitute_a_different_json_type(
    actual: JsonValue,
) -> None:
    payload = handoff_packet()
    rows = cast(list[dict[str, JsonValue]], payload["original_metadata_catalogue"])
    cast(dict[str, JsonValue], rows[0]["metadata"])["typed"] = True
    different = copy.deepcopy(rows[0])
    cast(dict[str, JsonValue], different["metadata"])["typed"] = actual
    payload["research_tasks"] = [{"row": different}]
    before = json.dumps(payload, sort_keys=True)
    decoded = expand_cropped_handoff_transport(
        encode_cropped_handoff_transport(payload)
    )
    assert json.dumps(decoded, sort_keys=True) == before


@pytest.mark.parametrize(
    "tamper",
    [
        "citation",
        "bool_index",
        "layout_index",
        "collision",
        "unused_layout",
        "duplicate_layout",
        "unknown_field",
        "fields",
        "instruction",
        "hash",
        "catalogue",
        "literal",
        "outside_scope",
        "orphan",
        "field_order",
        "field_order_hash",
    ],
)
def test_tampered_same_message_references_fail_closed(tamper: str) -> None:
    encoded = encode_cropped_handoff_transport(handoff_packet())
    header = cast(dict[str, JsonValue], encoded[HEADER])
    reference = first_reference(encoded["research_tasks"])
    if tamper == "citation":
        reference[REFERENCE] = [999, 0]
    elif tamper == "bool_index":
        reference[REFERENCE] = [True, False]
    elif tamper == "layout_index":
        reference[REFERENCE] = [1, 999]
    elif tamper == "collision":
        reference["citation"] = 1
    elif tamper == "unused_layout":
        cast(list[JsonValue], header["layouts"]).append(
            {"fields": ["source_id", "chunk_id"], "order": ["source_id", "chunk_id"]}
        )
    elif tamper == "duplicate_layout":
        cast(list[JsonValue], header["layouts"]).append(
            copy.deepcopy(cast(list[JsonValue], header["layouts"])[0])
        )
    elif tamper == "unknown_field":
        header["unexpected"] = None
    elif tamper == "fields":
        header["fields"] = ["original_evidence"]
    elif tamper == "instruction":
        header["instruction"] = "The model may use hidden sources."
    elif tamper == "hash":
        header["payload_hash"] = "0" * 64
    elif tamper == "catalogue":
        cast(list[dict[str, JsonValue]], encoded["original_metadata_catalogue"])[0][
            "source_id"
        ] = "changed-source"
    elif tamper == "literal":
        cast(list[dict[str, JsonValue]], encoded["original_evidence"])[0]["text"] = (
            "changed original"
        )
    elif tamper == "outside_scope":
        encoded["facts"] = [copy.deepcopy(reference)]
    elif tamper == "orphan":
        encoded.pop(HEADER)
    elif tamper == "field_order":
        layout = cast(list[dict[str, JsonValue]], header["layouts"])[0]
        layout["order"] = list(reversed(cast(list[JsonValue], layout["order"])))
    elif tamper == "field_order_hash":
        header["field_order_hash"] = "0" * 64
    with pytest.raises(ValueError):
        expand_cropped_handoff_transport(encoded)


def test_default_cheap_bytes_and_small_packet_cropped_bytes_are_unchanged() -> None:
    _, _, wire = packet()
    payload = cast(
        dict[str, JsonValue], json.loads(json.dumps(expand_cheap_payload(wire)))
    )
    before = copy.deepcopy(payload)
    expected = serialize_cheap_payload(
        payload, len, extra_fields=("research_tasks", "repair_research_observations")
    )
    assert serialize_cropped_payload(payload, len) == expected
    assert payload == before
    assert serialize_cheap_payload(payload, len) == serialize_cheap_payload(before, len)


def test_cropped_serializer_chooses_only_a_smaller_exact_candidate() -> None:
    _, _, wire = packet()
    payload = cast(
        dict[str, JsonValue], json.loads(json.dumps(expand_cheap_payload(wire)))
    )
    payload["research_tasks"] = [
        {"locators": copy.deepcopy(payload["original_metadata_catalogue"])}
        for _ in range(12)
    ]
    legacy = serialize_cheap_payload(
        payload, len, extra_fields=("research_tasks", "repair_research_observations")
    )
    selected = serialize_cropped_payload(payload, len)
    assert len(selected) <= len(legacy)
    assert expand_cropped_handoff_transport(json.loads(selected)) == payload


def test_reserved_marker_collision_cannot_change_existing_unknown_data() -> None:
    payload = handoff_packet()
    payload["research_tasks"] = [{REFERENCE: "existing unknown field"}]
    with pytest.raises(ValueError):
        encode_cropped_handoff_transport(payload)


def test_existing_shared_source_metadata_is_resolved_from_this_message() -> None:
    _, _, wire = packet()
    payload = cast(
        dict[str, JsonValue], json.loads(json.dumps(expand_cheap_payload(wire)))
    )
    rows = cast(list[dict[str, JsonValue]], payload["original_metadata_catalogue"])
    payload["research_tasks"] = [{"actual_source_rows": copy.deepcopy(rows)}]
    shared, sources = share_source_metadata(rows)
    payload["original_metadata_catalogue"] = cast(list[JsonValue], shared)
    payload["original_source_metadata"] = sources
    expected = canonical_cropped_handoff_payload(payload)
    transported = encode_cropped_handoff_transport(payload)
    assert expand_cropped_handoff_transport(transported) == expected
    assert (
        expand_cropped_handoff_transport(
            json.loads(serialize_cropped_payload(payload, len))
        )
        == expected
    )
    assert payload["original_source_metadata"] == sources


def test_null_chunk_and_decoded_metadata_have_no_shared_mutable_alias() -> None:
    payload = handoff_packet()
    row = cast(list[dict[str, JsonValue]], payload["original_metadata_catalogue"])[0]
    row["chunk_id"] = None
    payload["research_tasks"] = [{"source": copy.deepcopy(row)}]
    decoded = expand_cropped_handoff_transport(
        encode_cropped_handoff_transport(payload)
    )
    tasks = cast(list[dict[str, JsonValue]], decoded["research_tasks"])
    source = cast(dict[str, JsonValue], tasks[0]["source"])
    assert source["chunk_id"] is None
    cast(dict[str, JsonValue], source["metadata"])["article_no"] = "changed child"
    original = cast(list[dict[str, JsonValue]], decoded["original_metadata_catalogue"])[
        0
    ]
    assert cast(dict[str, JsonValue], original["metadata"])["article_no"] == "8"
