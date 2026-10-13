"""Explicit message-local handoff factoring preserves source and observation bytes."""

import copy
import json
from collections.abc import Sequence
from typing import cast

import pytest
import tiktoken
from pydantic import JsonValue

from onyx.asv3.cheap_metadata_transport import (
    _encode_cheap_metadata,
    encode_cheap_metadata,
    expand_cheap_payload,
    serialize_cheap_payload,
)
from onyx.asv3.cropped_research_limit import CroppedResearchLimit
from onyx.asv3.cropped_worker_handoff import (
    WORKER_HANDOFF_REASON,
    build_worker_handoff,
    expand_worker_handoff,
)
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import _NativeOriginalCatalogue
from onyx.asv3.models import (
    CapabilityCall,
    EvidenceItem,
    HarnessResult,
    OriginalEvidenceRead,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolReceipt,
)
from onyx.asv3.native_cache_projection import decode_compact_originals
from onyx.llm.models import UserMessage

EXTRA_FIELDS = ("research_tasks", "repair_research_observations")
ENCODER = tiktoken.get_encoding("o200k_base")


def tokens(text: str) -> int:
    return len(ENCODER.encode(text, disallowed_special=()))


def fixture() -> tuple[dict[str, JsonValue], RunContext, EvidenceLedger]:
    root = RunContext(run_id="same-run", scope={"owner": "same-user", "corpus": 15})
    ledger = EvidenceLedger()
    controller = CroppedResearchLimit(root, "Exact question and user supplied facts")
    root.services.update(
        asv3_exp_cropped=True, evidence=ledger, cropped_research_controller=controller
    )
    controller.force_finish("Recorded long tail")
    child = root.child()
    child.services["task_id"] = "same-worker"
    text = 'MADDE 8 — Koşul VE istisna korunur. "Özgün metin". 東京\n\t  ' * 40
    numbers = ledger.add(
        [
            EvidenceItem(
                source_id="same-source-with-recorded-authority-ffffffffffffffffffffffff",
                chunk_id=f"same-chunk-with-immutable-source-bounds-{index}",
                text=f"{text}Canonical paragraph {index}.",
                metadata={"article_no": "8", "title": "Official source " * 30},
            )
            for index in range(40)
        ],
        child,
    )
    number = numbers[0]
    item = ledger.get(number)
    assert item is not None
    text = item.text
    receipt = ToolReceipt(
        call=CapabilityCall(
            name="read_provision",
            call_id="physical-read",
            arguments={"source_id": item.source_id, "article": "8"},
        ),
        outcome=ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="Actual continuation not read",
            data={"exact_text": text, "has_more": True, "absence_proven": False},
            original_reads=[
                OriginalEvidenceRead(
                    citation=number,
                    text_hash=item.text_hash,
                    start_char=0,
                    end_char=len(text),
                )
            ],
        ),
        elapsed_seconds=0.2,
        evidence_ids=[number],
    )
    handoff = build_worker_handoff(
        context=child,
        result=HarnessResult(
            answer=None,
            status=OutcomeStatus.PARTIAL,
            receipts=[receipt],
            questions=["Exact subquestion"],
            facts=["Provided qualification and unresolved fact. " * 300],
            stop_reason=WORKER_HANDOFF_REASON,
        ),
        ledger=ledger,
        turns=[],
        research_state={
            "unknown_fields": [None, True, False, 1, 1.0, "1", [], {}],
            "literal_collision": {"value_ref": 0, "layout_ref": [1, []]},
        },
        working_memory={"query": "Actual exact source", "not_read": True},
        draft=None,
    )
    originals = cast(
        list[dict[str, JsonValue]],
        json.loads(ledger.serialize_records(numbers, max_chars=None)),
    )
    task: dict[str, JsonValue] = {
        "task_id": "same-worker",
        "status": "completed",
        "outcome": {"status": "partial", "cropped_worker_handoff": handoff},
        "source_identity_index": [
            {key: row[key] for key in ("source_id", "chunk_id", "text_hash")}
            for row in originals
        ],
    }
    payload: dict[str, JsonValue] = {
        "original_metadata_catalogue": [
            {
                key: row[key]
                for key in (
                    "citation",
                    "source_id",
                    "chunk_id",
                    "text_hash",
                    "metadata",
                )
            }
            for row in originals
        ],
        "original_evidence": [
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key not in {"source_id", "chunk_id", "text_hash", "metadata"}
                },
                "identity_ref": row["citation"],
                "start_char": 0,
                "end_char": len(cast(str, row["text"])),
            }
            for row in originals
        ],
        "research_tasks": [task],
        "repair_research_observations": [
            {"name": "wait_researcher", "result": {"tasks": [copy.deepcopy(task)]}}
        ],
    }
    return payload, child, ledger


def test_declared_handoff_fields_preserve_actual_payload_and_native_source_delivery() -> (
    None
):
    payload, child, ledger = fixture()
    saved = copy.deepcopy(payload)
    baseline = serialize_cheap_payload(payload, tokens)
    wire = serialize_cheap_payload(payload, tokens, extra_fields=EXTRA_FIELDS)
    assert tokens(wire) < tokens(baseline)
    encoded = json.loads(wire)
    restored = expand_cheap_payload(encoded)
    assert restored == payload == saved
    task = cast(list[dict[str, JsonValue]], restored["research_tasks"])[0]
    handoff = cast(
        dict[str, JsonValue],
        cast(dict[str, JsonValue], task["outcome"])["cropped_worker_handoff"],
    )
    assert expand_worker_handoff(handoff, ledger=ledger, context=child)[
        "working_memory"
    ] == {
        "query": "Actual exact source",
        "not_read": True,
    }
    assert handoff["publication_approved"] is False
    bound = _NativeOriginalCatalogue.bind(
        [UserMessage(content=wire)], cheap_metadata=True
    )
    actual = decode_compact_originals([restored], bound.records, ledger)
    assert len(actual) == 40
    item = ledger.get(1)
    delivered_text = actual[0]["text"]
    assert isinstance(delivered_text, str)
    assert item is not None and delivered_text.encode() == item.text.encode()
    assert json.dumps(item.text, ensure_ascii=False) in wire


def test_default_cheap_transport_does_not_encode_new_fields() -> None:
    payload, _, _ = fixture()
    ordinary = encode_cheap_metadata(payload, tokens)
    assert ordinary == encode_cheap_metadata(payload, tokens, extra_fields=())
    assert serialize_cheap_payload(payload, tokens) == serialize_cheap_payload(
        payload, tokens, extra_fields=()
    )
    assert ordinary["research_tasks"] == payload["research_tasks"]
    assert (
        ordinary["repair_research_observations"]
        == payload["repair_research_observations"]
    )
    header = cast(dict[str, JsonValue], ordinary["original_metadata_codec"])
    assert set(EXTRA_FIELDS).isdisjoint(cast(list[str], header["payload_fields"]))
    upgraded = encode_cheap_metadata(ordinary, tokens, extra_fields=EXTRA_FIELDS)
    assert expand_cheap_payload(upgraded) == payload
    assert tokens(json.dumps(upgraded, ensure_ascii=False)) < tokens(
        json.dumps(ordinary, ensure_ascii=False)
    )
    assert (
        encode_cheap_metadata(upgraded, tokens, extra_fields=EXTRA_FIELDS) == upgraded
    )


def test_absent_extra_fields_keep_the_version_one_wire() -> None:
    payload, _, _ = fixture()
    for field in EXTRA_FIELDS:
        payload.pop(field)
    ordinary = serialize_cheap_payload(payload, tokens)
    assert (
        serialize_cheap_payload(payload, tokens, extra_fields=EXTRA_FIELDS) == ordinary
    )
    assert json.loads(ordinary)["original_metadata_codec"]["version"] == 1


def test_v2_identity_references_preserve_real_canonical_sources_and_literals() -> None:
    payload, _, ledger = fixture()
    # Validate the v2 candidate separately from the public smallest-wire chooser.
    coded = _encode_cheap_metadata(
        payload, tokens, extra_fields=EXTRA_FIELDS, identity_refs=True
    )
    header = cast(dict[str, JsonValue], coded["original_metadata_codec"])
    assert header["version"] == 2
    rows = cast(list[list[JsonValue]], coded["original_metadata_catalogue"])
    assert isinstance(rows[0][2], dict) and isinstance(rows[0][3], dict)
    sources = cast(list[list[JsonValue]], coded["original_metadata_sources"])
    assert isinstance(sources[0][0], dict)
    restored = expand_cheap_payload(coded)
    assert restored == payload
    wire = json.dumps(coded, ensure_ascii=False, separators=(",", ":"))
    bound = _NativeOriginalCatalogue.bind(
        [UserMessage(content=wire)], cheap_metadata=True
    )
    assert len(decode_compact_originals([restored], bound.records, ledger)) == 40
    tampered = copy.deepcopy(coded)
    values = cast(list[JsonValue], tampered["original_metadata_values"])
    digest = cast(dict[str, JsonValue], rows[0][3])
    reference = cast(int, digest["value_ref"])
    values[reference] = "a different string hash still needs actual ledger validation"
    changed = expand_cheap_payload(tampered)
    with pytest.raises(ValueError, match="canonical"):
        decode_compact_originals(
            [changed],
            cast(list[dict[str, JsonValue]], changed["original_metadata_catalogue"]),
            ledger,
        )


@pytest.mark.parametrize(
    "damage",
    [
        "bool_ref",
        "outside_ref",
        "null_hash",
        "container_hash",
        "null_source",
        "layout_identity",
        "v1_header",
        "v2_without_extra",
    ],
)
def test_v2_identity_node_types_and_version_are_strict(damage: str) -> None:
    payload, _, _ = fixture()
    coded = _encode_cheap_metadata(
        payload, tokens, extra_fields=EXTRA_FIELDS, identity_refs=True
    )
    header = cast(dict[str, JsonValue], coded["original_metadata_codec"])
    assert header["version"] == 2
    rows = cast(list[list[JsonValue]], coded["original_metadata_catalogue"])
    sources = cast(list[list[JsonValue]], coded["original_metadata_sources"])
    values = cast(list[JsonValue], coded["original_metadata_values"])
    if damage == "bool_ref":
        rows[0][3] = {"value_ref": True}
    elif damage == "outside_ref":
        rows[0][3] = {"value_ref": len(values)}
    elif damage == "null_hash":
        rows[0][3] = None
    elif damage == "container_hash":
        values[cast(int, cast(dict[str, JsonValue], rows[0][3])["value_ref"])] = []
    elif damage == "null_source":
        values[cast(int, cast(dict[str, JsonValue], sources[0][0])["value_ref"])] = None
    elif damage == "layout_identity":
        rows[0][2] = {"layout_ref": [0, []]}
    elif damage == "v1_header":
        version_one = encode_cheap_metadata(payload, tokens)
        header["version"] = 1
        coded["original_metadata_transport_note"] = version_one[
            "original_metadata_transport_note"
        ]
    else:
        header["payload_fields"] = ["original_evidence"]
    with pytest.raises(ValueError):
        expand_cheap_payload(coded)


@pytest.mark.parametrize(
    "fields", [("request",), ("original_evidence",), EXTRA_FIELDS * 2, "research_tasks"]
)
def test_unknown_or_duplicate_extra_fields_are_rejected(fields: Sequence[str]) -> None:
    payload, _, _ = fixture()
    with pytest.raises(ValueError, match="extra payload fields"):
        serialize_cheap_payload(payload, tokens, extra_fields=fields)


@pytest.mark.parametrize(
    "damage", ["missing", "duplicate", "unknown", "unused_value", "invalid_ref"]
)
def test_extra_field_tampering_fails_closed(damage: str) -> None:
    payload, _, _ = fixture()
    coded = encode_cheap_metadata(payload, tokens, extra_fields=EXTRA_FIELDS)
    header = cast(dict[str, JsonValue], coded["original_metadata_codec"])
    fields = cast(list[str], header["payload_fields"])
    if damage == "missing":
        coded.pop("research_tasks")
    elif damage == "duplicate":
        fields.append("research_tasks")
    elif damage == "unknown":
        fields.append("invented_approval")
    elif damage == "unused_value":
        cast(list[JsonValue], coded["original_metadata_values"]).append(
            "Unused fabricated approval"
        )
    else:
        coded["research_tasks"] = {"value_ref": -1}
    with pytest.raises(ValueError):
        expand_cheap_payload(coded)
