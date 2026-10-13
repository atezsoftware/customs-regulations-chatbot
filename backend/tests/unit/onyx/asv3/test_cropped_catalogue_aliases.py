"""Full inline sources and exact worker JSON survive the optional alias grammar."""

import copy
import hashlib
import json
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.cheap_metadata_transport import (
    expand_cheap_payload,
    serialize_cheap_payload,
)
from onyx.asv3.cropped_catalogue_aliases import (
    CATALOGUE_ALIAS_HEADER,
    encode_cropped_catalogue_aliases,
    expand_cropped_catalogue_aliases,
)
from onyx.asv3.cropped_handoff_transport import (
    encode_cropped_handoff_transport,
    expand_cropped_handoff_transport,
)
from onyx.asv3.cropped_transport import serialize_cropped_payload
from onyx.asv3.cropped_worker_handoff import build_worker_handoff, expand_worker_handoff
from onyx.asv3.llm_adapter import ResearchModel, _NativeOriginalCatalogue
from onyx.llm.models import ChatCompletionMessage, UserMessage
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.asv3.test_cheap_metadata_adapter import packet
from tests.unit.onyx.asv3.test_cropped_worker_handoff import payload as worker_payload
from tests.unit.onyx.asv3.test_native_model_adapter import model
from tests.unit.onyx.asv3.test_shared_originals import full_record, original, recorded

SUBTREE = "canonical_metadata_subtree_ref"
BINDING = "canonical_full_original_ref"


def alias_packet() -> dict[str, JsonValue]:
    ledger = recorded(
        [original("known-chunk", "Original rule and exact exception. " * 40)]
    )
    record = full_record(ledger, 1)
    record.update(start_char=0, end_char=len(cast(str, record["text"])))
    metadata: dict[str, JsonValue] = {
        "long_name": "Canonical source identity and operative exception. " * 5,
        "nested": [{"flag": False, "count": 1, "decimal": 1.0}, {"null": None}],
    }
    different: dict[str, JsonValue] = {
        "nested": [{"decimal": 1.0, "count": 1, "flag": False}, {"null": None}],
        "long_name": metadata["long_name"],
    }
    return {
        "request": "Preserve the actual conditions and unresolved source effect.",
        "original_metadata_catalogue": [
            {
                **{
                    key: record[key]
                    for key in ("citation", "source_id", "chunk_id", "text_hash")
                },
                "metadata": metadata,
            }
        ],
        "original_evidence": [record],
        "research_tasks": [
            {
                "metadata": different,
                "binding": {
                    **{
                        key: record[key]
                        for key in ("citation", "source_id", "chunk_id", "text_hash")
                    },
                    "metadata_hash": "a" * 64,
                    "start_char": 0,
                    "end_char": len(cast(str, record["text"])),
                    "range_hash": record["text_hash"],
                },
                "unread_status": {"not_found": True, "absence_proven": False},
            }
        ],
    }


def reference(value: JsonValue, key: str) -> dict[str, JsonValue]:
    if isinstance(value, dict):
        if key in value:
            return value
        children = value.values()
    elif isinstance(value, list):
        children = value
    else:
        raise LookupError(key)
    for child in children:
        try:
            return reference(child, key)
        except LookupError:
            pass
    raise LookupError(key)


def test_recursive_order_types_and_opaque_hash_roundtrip_through_both_codecs() -> None:
    payload = alias_packet()
    before = json.dumps(payload, ensure_ascii=False)
    encoded = encode_cropped_catalogue_aliases(payload)
    assert CATALOGUE_ALIAS_HEADER in encoded
    assert BINDING in reference(encoded["research_tasks"], BINDING)
    transported = serialize_cheap_payload(
        encode_cropped_handoff_transport(encoded),
        len,
        extra_fields=("research_tasks", "repair_research_observations"),
    )
    decoded = expand_cropped_handoff_transport(json.loads(transported))
    assert json.dumps(decoded, ensure_ascii=False) == before
    assert json.dumps(payload, ensure_ascii=False) == before
    assert decoded["original_evidence"] == payload["original_evidence"]
    assert "a" * 64 in json.dumps(decoded["research_tasks"])


@pytest.mark.parametrize("compact", [False, True])
def test_actual_worker_json_string_hashes_and_source_binding_remain_valid(
    compact: bool,
) -> None:
    context, ledger, result, turns, state, memory, draft = worker_payload()
    record = full_record(ledger, 1)
    cast(dict[str, JsonValue], record["metadata"])["unknown_ordered_metadata"] = {
        "first": "Unknown fields and scalar types remain literal. " * 5,
        "second": {"flag": False, "integer": 1, "decimal": 1.0},
    }
    changed = copy.deepcopy(record)
    metadata = cast(dict[str, JsonValue], changed["metadata"])
    changed["metadata"] = dict(reversed(list(metadata.items())))
    literal = json.dumps(
        {"has_more": True, "original_evidence": [changed], "absence_proven": False},
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
    payload: dict[str, JsonValue] = {
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
        "research_tasks": [{"handoff": handoff}],
    }
    encoded = encode_cropped_catalogue_aliases(payload)
    assert CATALOGUE_ALIAS_HEADER in encoded
    decoded = expand_cropped_handoff_transport(
        json.loads(
            serialize_cheap_payload(
                encode_cropped_handoff_transport(encoded),
                len,
                extra_fields=("research_tasks",),
            )
        )
    )
    actual = cast(
        dict[str, JsonValue],
        cast(list[dict[str, JsonValue]], decoded["research_tasks"])[0]["handoff"],
    )
    assert json.dumps(actual, ensure_ascii=False) == json.dumps(
        handoff, ensure_ascii=False
    )
    expanded = expand_worker_handoff(actual, ledger=ledger, context=context)
    turn = cast(list[dict[str, JsonValue]], expanded["turns"])[0]
    restored = cast(list[dict[str, JsonValue]], turn["results"])[0]["content"]
    assert restored == literal
    assert (
        hashlib.sha256(cast(str, restored).encode()).hexdigest()
        == hashlib.sha256(literal.encode()).hexdigest()
    )


@pytest.mark.parametrize("actual", [True, 1, 1.0, None, False])
def test_equal_python_numbers_do_not_change_json_scalar_type(actual: JsonValue) -> None:
    payload = alias_packet()
    rows = cast(list[dict[str, JsonValue]], payload["original_metadata_catalogue"])
    cast(dict[str, JsonValue], rows[0]["metadata"])["typed"] = True
    tasks = cast(list[dict[str, JsonValue]], payload["research_tasks"])
    cast(dict[str, JsonValue], tasks[0]["metadata"])["typed"] = actual
    before = json.dumps(payload, ensure_ascii=False)
    assert (
        json.dumps(
            expand_cropped_catalogue_aliases(encode_cropped_catalogue_aliases(payload)),
            ensure_ascii=False,
        )
        == before
    )


@pytest.mark.parametrize(
    "tamper",
    [
        "citation",
        "bool_reference",
        "path",
        "unused_order",
        "duplicate_order",
        "opaque_hash",
        "body",
        "citable",
        "truncated",
        "bool_start",
        "bool_end",
        "bool_identity",
        "extra_body",
        "missing_body",
        "identity_ref_conflict",
        "order",
        "field_scope",
        "orphan",
    ],
)
def test_tampered_aliases_or_full_body_witnesses_fail_closed(tamper: str) -> None:
    encoded = encode_cropped_catalogue_aliases(alias_packet())
    header = cast(dict[str, JsonValue], encoded[CATALOGUE_ALIAS_HEADER])
    binding = reference(encoded["research_tasks"], BINDING)
    subtree = reference(encoded["research_tasks"], SUBTREE)
    body = cast(list[dict[str, JsonValue]], encoded["original_evidence"])[0]
    if tamper == "citation":
        cast(list[JsonValue], binding[BINDING])[0] = 999
    elif tamper == "bool_reference":
        cast(list[JsonValue], binding[BINDING])[0] = True
    elif tamper == "path":
        cast(list[JsonValue], header["paths"])[0] = ["unknown"]
    elif tamper == "unused_order":
        cast(list[JsonValue], header["orders"]).append(
            {"keys": ["unavailable"], "children": {}}
        )
    elif tamper == "duplicate_order":
        cast(list[JsonValue], header["orders"]).append(
            copy.deepcopy(cast(list[JsonValue], header["orders"])[0])
        )
    elif tamper == "opaque_hash":
        cast(list[JsonValue], binding[BINDING])[1] = "b" * 64
    elif tamper == "body":
        body["text"] = cast(str, body["text"]) + " changed"
    elif tamper in {"citable", "truncated"}:
        body[tamper] = tamper == "truncated"
    elif tamper in {"bool_start", "bool_end", "bool_identity"}:
        body[
            {
                "bool_start": "start_char",
                "bool_end": "end_char",
                "bool_identity": "identity_ref",
            }[tamper]
        ] = tamper != "bool_start"
    elif tamper == "extra_body":
        extra = copy.deepcopy(body)
        extra["citation"] = 2
        cast(list[JsonValue], encoded["original_evidence"]).append(extra)
    elif tamper == "missing_body":
        encoded["original_evidence"] = []
    elif tamper == "identity_ref_conflict":
        body["identity_ref"] = 1
        body["source_id"] = "foreign-source"
    elif tamper == "order":
        index = cast(list[JsonValue], subtree[SUBTREE])[2]
        order = cast(
            dict[str, JsonValue],
            cast(list[JsonValue], header["orders"])[cast(int, index)],
        )
        order["keys"] = list(reversed(cast(list[JsonValue], order["keys"])))
    elif tamper == "field_scope":
        header["fields"] = ["original_evidence"]
    elif tamper == "orphan":
        del encoded[CATALOGUE_ALIAS_HEADER]
    with pytest.raises(ValueError):
        expand_cropped_catalogue_aliases(encoded)


def test_incomplete_inline_original_does_not_enable_a_full_binding_alias() -> None:
    payload = alias_packet()
    body = cast(list[dict[str, JsonValue]], payload["original_evidence"])[0]
    body["truncated"] = True
    encoded = encode_cropped_catalogue_aliases(payload)
    with pytest.raises(LookupError):
        reference(encoded["research_tasks"], BINDING)
    assert expand_cropped_catalogue_aliases(encoded) == payload


def test_smaller_only_selection_keeps_previous_cheap_wire_on_equal_counts() -> None:
    payload = alias_packet()

    def counter(_: str) -> int:
        return 1

    legacy = serialize_cheap_payload(
        payload,
        counter,
        extra_fields=("research_tasks", "repair_research_observations"),
    )
    assert serialize_cropped_payload(payload, counter) == legacy


def test_actual_native_delivery_decodes_aliases_and_requires_every_inline_body() -> (
    None
):
    context, ledger, encoded = packet()
    context.services.pop("asv3_exp_cheap")
    context.services["asv3_exp_cropped"] = True
    payload = expand_cheap_payload(encoded)
    rows = cast(list[dict[str, JsonValue]], payload["original_metadata_catalogue"])
    payload["research_tasks"] = [
        {"metadata": copy.deepcopy(row["metadata"])} for row in rows
    ]
    alias_only = encode_cropped_handoff_transport(
        encode_cropped_catalogue_aliases(payload)
    )
    assert CATALOGUE_ALIAS_HEADER in alias_only
    assert "cropped_canonical_fields_transport" not in alias_only
    transported = serialize_cheap_payload(
        alias_only,
        len,
        extra_fields=("research_tasks",),
    )
    prompt: list[ChatCompletionMessage] = [UserMessage(content=transported)]
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
    assert (
        expand_cropped_handoff_transport(json.loads(transported))["original_evidence"]
        == payload["original_evidence"]
    )


@pytest.mark.parametrize("tamper", ["hash", "order", "bool_reference"])
def test_alias_only_native_binder_rejects_tampering_before_provider_dispatch(
    tamper: str,
) -> None:
    context, _, source = packet()
    context.services.pop("asv3_exp_cheap")
    context.services["asv3_exp_cropped"] = True
    payload = expand_cheap_payload(source)
    rows = cast(list[dict[str, JsonValue]], payload["original_metadata_catalogue"])
    payload["research_tasks"] = [
        {"metadata": copy.deepcopy(row["metadata"])} for row in rows
    ]
    transported = encode_cropped_handoff_transport(
        encode_cropped_catalogue_aliases(payload)
    )
    assert CATALOGUE_ALIAS_HEADER in transported
    assert "cropped_canonical_fields_transport" not in transported
    valid_wire = serialize_cheap_payload(
        transported, len, extra_fields=("research_tasks",)
    )
    valid = _NativeOriginalCatalogue.bind(
        [UserMessage(content=valid_wire)], cheap_metadata=True
    )
    assert len(valid.records) == 12
    assert (
        expand_cropped_handoff_transport(json.loads(valid_wire))["original_evidence"]
        == payload["original_evidence"]
    )
    header = cast(dict[str, JsonValue], transported[CATALOGUE_ALIAS_HEADER])
    if tamper == "hash":
        header["payload_hash"] = "0" * 64
    elif tamper == "order":
        order = cast(dict[str, JsonValue], cast(list[JsonValue], header["orders"])[0])
        order["keys"] = list(reversed(cast(list[JsonValue], order["keys"])))
    else:
        subtree = reference(transported["research_tasks"], SUBTREE)
        cast(list[JsonValue], subtree[SUBTREE])[0] = True
    bad_wire = serialize_cheap_payload(
        transported, len, extra_fields=("research_tasks",)
    )
    provider = model(limit=1_000_000)
    adapter = ResearchModel(provider, context, lean_native_mode=True)
    with pytest.raises(ValueError):
        invalid = _NativeOriginalCatalogue.bind(
            [UserMessage(content=bad_wire)], cheap_metadata=True
        )
        adapter._invoke_once(
            [UserMessage(content=bad_wire)],
            [],
            LLMFlow.ASV3_FINAL,
            max_tokens=100,
            research=False,
            structured=False,
            native_catalogue=invalid,
        )
    provider.invoke.assert_not_called()
