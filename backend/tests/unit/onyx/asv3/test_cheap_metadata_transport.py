import copy
import gzip
import hashlib
import json
from pathlib import Path
from typing import cast

import pytest
import tiktoken
from pydantic import JsonValue

from onyx.asv3.cheap_metadata_transport import (
    encode_cheap_metadata,
    expand_cheap_metadata,
    expand_cheap_payload,
    serialize_cheap_payload,
)
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, RunContext
from onyx.asv3.native_cache_projection import decode_compact_originals
from onyx.asv3.source_metadata_transport import (
    expand_source_metadata,
    share_source_metadata,
)

ENCODER = tiktoken.get_encoding("o200k_base")


def tokens(text: str) -> int:
    return len(ENCODER.encode(text, disallowed_special=()))


def payload() -> dict[str, JsonValue]:
    rows: list[JsonValue] = []
    heading = (
        "Verified provision, material exceptions and implementation conditions. " * 7
    )
    for index in range(40):
        metadata: dict[str, JsonValue] = {
            "title": "Original official document " * 14,
            "heading_path": [heading, f"Provision {index}"],
            "unknown_metadata": {
                "position": index,
                "typed_values": [None, True, False, 1, 1.0, "1", [], {}],
                "layout_ref": [0, ["literal canonical field"]],
                "value_ref": {"value_ref": True},
            },
        }
        if index % 2:
            metadata["absent_or_null"] = None
        rows.append(
            {
                "citation": index + 1,
                "source_id": f"source-{index % 3}",
                "chunk_id": None if index == 0 else f"chunk-{index}",
                "text_hash": hashlib.sha256(str(index).encode()).hexdigest(),
                "metadata": metadata,
            }
        )
    return {
        "request": "Conditions, exceptions and unresolved facts must remain visible.",
        "original_metadata_catalogue": rows,
        "original_evidence": [
            {"text": "Literal source text. 東京\n\t  ", "citation": 1}
        ],
        "original_evidence_ranges": [
            {"citation": index + 1, "start_char": 0, "end_char": index + 20}
            for index in range(40)
        ],
        "delivered_provisions": [
            {
                "source_id": f"source-{index % 3}",
                "original_citations": [index + 1],
                "scope": {"unknown_fact": None, "conditions": heading},
            }
            for index in range(40)
        ],
    }


def encoded() -> dict[str, JsonValue]:
    result = encode_cheap_metadata(payload(), tokens)
    assert "original_metadata_codec" in result
    return result


def test_full_roundtrip_preserves_types_unknown_keys_order_and_literal_collisions() -> (
    None
):
    original = payload()
    saved = copy.deepcopy(original)
    coded = encode_cheap_metadata(original, tokens)
    assert tokens(json.dumps(coded, ensure_ascii=False)) < tokens(
        json.dumps(original, ensure_ascii=False)
    )
    restored = expand_cheap_payload(coded)
    assert restored == original == saved
    assert json.dumps(restored, sort_keys=True) == json.dumps(original, sort_keys=True)
    assert restored["original_evidence"] == original["original_evidence"]
    metadata = cast(
        list[dict[str, JsonValue]], restored["original_metadata_catalogue"]
    )[1]["metadata"]
    assert isinstance(metadata, dict)
    assert list(metadata) == list(
        cast(
            dict[str, JsonValue],
            cast(list[dict[str, JsonValue]], original["original_metadata_catalogue"])[
                1
            ]["metadata"],
        )
    )
    assert expand_cheap_metadata(coded) == original["original_metadata_catalogue"]
    coded["request"] = "Independent copy"
    assert original == saved


def test_small_payload_keeps_ordinary_bytes_and_does_not_mutate() -> None:
    original: dict[str, JsonValue] = {"request": "q", "original_metadata_catalogue": []}
    saved = json.dumps(original, ensure_ascii=False)
    coded = encode_cheap_metadata(original, tokens)
    assert coded is not original
    assert json.dumps(coded, ensure_ascii=False) == saved
    assert expand_cheap_payload(coded) == original
    ordinary: dict[str, JsonValue] = {"request": "No catalogue"}
    assert expand_cheap_metadata(ordinary) == []


def test_flat_and_existing_shared_inputs_decode_to_the_same_objects() -> None:
    original = payload()
    rows, common = share_source_metadata(
        cast(list[dict[str, JsonValue]], original["original_metadata_catalogue"])
    )
    shared = {
        **copy.deepcopy(original),
        "original_metadata_catalogue": cast(JsonValue, rows),
        "original_source_metadata": common,
    }
    coded = encode_cheap_metadata(shared, tokens)
    assert expand_cheap_metadata(coded) == expand_source_metadata(shared)
    flat = encode_cheap_metadata(original, tokens)
    assert expand_cheap_metadata(flat) == expand_cheap_metadata(coded)
    assert encode_cheap_metadata(coded, tokens) == coded


@pytest.mark.parametrize(
    "damage",
    [
        "version",
        "bool_version",
        "catalogue_fields",
        "payload_field",
        "duplicate_payload_field",
        "missing_payload_field",
        "missing_table",
        "row_arity",
        "duplicate_citation",
        "bool_citation",
        "bool_source_ref",
        "source_ref_range",
        "source_duplicate",
        "unused_source",
        "layout_duplicate_key",
        "unused_layout",
        "duplicate_layout",
        "layout_arity",
        "unused_value",
        "duplicate_value",
        "overlap",
        "legacy_table",
        "grammar",
    ],
)
def test_tampered_tables_and_references_fail_closed(damage: str) -> None:
    coded = encoded()
    header = cast(dict[str, JsonValue], coded["original_metadata_codec"])
    rows = cast(list[list[JsonValue]], coded["original_metadata_catalogue"])
    sources = cast(list[list[JsonValue]], coded["original_metadata_sources"])
    layouts = cast(list[list[str]], coded["original_metadata_layouts"])
    values = cast(list[JsonValue], coded["original_metadata_values"])
    if damage == "version":
        header["version"] = 2
    elif damage == "bool_version":
        header["version"] = True
    elif damage == "catalogue_fields":
        header["catalogue_fields"] = list(
            reversed(cast(list[JsonValue], header["catalogue_fields"]))
        )
    elif damage == "payload_field":
        header["payload_fields"] = ["request"]
    elif damage == "duplicate_payload_field":
        header["payload_fields"] = ["delivered_provisions", "delivered_provisions"]
    elif damage == "missing_payload_field":
        coded.pop("delivered_provisions")
    elif damage == "missing_table":
        coded.pop("original_metadata_layouts")
    elif damage == "row_arity":
        rows[0].pop()
    elif damage == "duplicate_citation":
        rows[1][0] = rows[0][0]
    elif damage == "bool_citation":
        rows[0][0] = True
    elif damage == "bool_source_ref":
        rows[0][1] = True
    elif damage == "source_ref_range":
        rows[0][1] = len(sources)
    elif damage == "source_duplicate":
        sources[1][0] = sources[0][0]
    elif damage == "unused_source":
        sources.append(["unused source", {}])
    elif damage == "layout_duplicate_key":
        layouts[0].append(layouts[0][0])
    elif damage == "unused_layout":
        layouts.append(["unused new key"])
    elif damage == "duplicate_layout":
        layouts.append(copy.deepcopy(layouts[0]))
    elif damage == "layout_arity":
        node = cast(dict[str, JsonValue], rows[0][4])
        pair = cast(list[JsonValue], node["layout_ref"])
        cast(list[JsonValue], pair[1]).append("extra")
    elif damage == "unused_value":
        values.append("unused distinct literal")
    elif damage == "duplicate_value":
        values.append(copy.deepcopy(values[0]))
    elif damage == "overlap":
        sources[0][1] = copy.deepcopy(rows[0][4])
    elif damage == "legacy_table":
        coded["original_source_metadata"] = {}
    elif damage == "grammar":
        coded["original_metadata_transport_note"] = "Wrong grammar"
    with pytest.raises(ValueError):
        expand_cheap_payload(coded)


def test_unversioned_transport_and_extra_identity_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="versioned"):
        expand_cheap_payload({"original_metadata_values": []})
    original = payload()
    rows = cast(list[dict[str, JsonValue]], original["original_metadata_catalogue"])
    rows[0]["extra_identity_claim"] = True
    with pytest.raises(ValueError, match="exact canonical"):
        encode_cheap_metadata(original, tokens)


@pytest.mark.parametrize("compact_serialization", [False, True])
def test_actual_literal_records_still_pass_the_native_ledger_guard(
    compact_serialization: bool,
) -> None:
    ledger = EvidenceLedger()
    context = RunContext(scope={"document_set_id": 15})
    numbers = ledger.add(
        [
            EvidenceItem(
                source_id="owned-source",
                chunk_id=f"chunk-{index}",
                text=f"Original clause {index}; izin VE koşul korunur. 東京\n\t  ",
                metadata={
                    "title": "Verified official source " * 20,
                    "heading_path": [
                        "Exact conditions and operative exceptions " * 10,
                        f"Article {index}",
                    ],
                    "paragraph_no": str(index),
                },
            )
            for index in range(40)
        ],
        context,
    )
    originals = cast(
        list[dict[str, JsonValue]],
        json.loads(ledger.serialize_records(numbers, max_chars=None)),
    )
    catalogue = [
        {
            key: row[key]
            for key in ("citation", "source_id", "chunk_id", "text_hash", "metadata")
        }
        for row in originals
    ]
    literal = [
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
    ]
    original: dict[str, JsonValue] = {
        "original_metadata_catalogue": cast(JsonValue, catalogue),
        "original_evidence": cast(JsonValue, literal),
    }
    coded = (
        json.loads(serialize_cheap_payload(original, tokens))
        if compact_serialization
        else encode_cheap_metadata(original, tokens)
    )
    assert "original_metadata_codec" in coded
    restored = expand_cheap_payload(coded)
    decoded = decode_compact_originals([restored], expand_cheap_metadata(coded), ledger)
    assert decoded == [
        {**row, "start_char": 0, "end_char": len(cast(str, row["text"]))}
        for row in originals
    ]
    tampered = copy.deepcopy(coded)
    sources = cast(list[list[JsonValue]], tampered["original_metadata_sources"])
    sources[0][0] = "another canonical source"
    with pytest.raises(ValueError, match="canonical"):
        decode_compact_originals([restored], expand_cheap_metadata(tampered), ledger)


def test_captured_5077_semantic_support_metadata_remains_exact() -> None:
    path = Path(__file__).parent / "fixtures/cheap_repair_history.json.gz"
    raw = path.read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "f7234675c1faea170ebc09921642b69a4dcd9777b84c0016f42dc64c4c1c2ba7"
    )
    history = json.loads(gzip.decompress(raw))
    run = next(row for row in history["runs"] if row["message_id"] == 5077)
    catalogue = [
        {
            key: row[key]
            for key in ("citation", "source_id", "chunk_id", "text_hash", "metadata")
        }
        for row in run["originals"]
    ]
    original: dict[str, JsonValue] = {"original_metadata_catalogue": catalogue}
    coded = encode_cheap_metadata(original, tokens)
    restored = expand_cheap_metadata(coded)
    assert restored == catalogue
    assert set(history["known_5077_final_support"]).issubset(
        {row["citation"] for row in restored}
    )


def test_original_text_is_inline_even_when_the_same_string_is_in_the_value_pool() -> (
    None
):
    original = payload()
    catalogue = cast(
        list[dict[str, JsonValue]], original["original_metadata_catalogue"]
    )
    metadata = cast(dict[str, JsonValue], catalogue[0]["metadata"])
    heading = cast(list[str], metadata["heading_path"])[0]
    original["original_evidence"] = [
        {"citation": 1, "text": heading},
        {"citation": 2, "text": heading},
    ]
    coded = encode_cheap_metadata(original, tokens)
    encoded_records = cast(list[JsonValue], coded["original_evidence"])
    # The literal occurs in both actual record nodes, independent of table contents.
    for record in encoded_records:
        assert json.dumps(heading, ensure_ascii=False) in json.dumps(
            record, ensure_ascii=False
        )
    assert (
        expand_cheap_payload(coded)["original_evidence"]
        == original["original_evidence"]
    )


def test_compact_serialization_preserves_source_bytes_types_and_entire_payload() -> (
    None
):
    original = payload()
    saved = copy.deepcopy(original)
    literal = '  Ön koşul VE istisna; "text": [1, null]. 東京\n\t  '
    original["original_evidence"] = [{"citation": 1, "text": literal}]
    saved = copy.deepcopy(original)
    coded = encode_cheap_metadata(original, tokens)
    serialized = serialize_cheap_payload(original, tokens)
    assert tokens(serialized) < tokens(json.dumps(coded, ensure_ascii=False))
    assert json.loads(serialized) == coded
    assert expand_cheap_payload(json.loads(serialized)) == original == saved
    restored = cast(
        list[dict[str, JsonValue]],
        expand_cheap_payload(json.loads(serialized))["original_evidence"],
    )[0]["text"]
    assert isinstance(restored, str)
    assert restored.encode() == literal.encode()
    assert json.dumps(literal, ensure_ascii=False) in serialized


@pytest.mark.parametrize("compact_penalty", [0, 1])
def test_serialization_keeps_default_separators_when_counter_does_not_improve(
    compact_penalty: int,
) -> None:
    original: dict[str, JsonValue] = {
        "request": "q",
        "facts": [None, True, False, 1, 1.0, "  "],
    }
    ordinary = json.dumps(original, ensure_ascii=False)

    def counter(text: str) -> int:
        return 10 if text == ordinary else 10 + compact_penalty

    assert serialize_cheap_payload(original, counter) == ordinary


def test_captured_repeated_condition_list_preserves_gap_quotes_and_actual_originals() -> (
    None
):
    path = Path(__file__).parent / "fixtures/cheap_metadata_condition_repeat.json.gz"
    raw = path.read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "67e862147a6a4cbe6b4c47de95ded2dbebb5dc9361f7daf2fd15e9cb4a6ce7d0"
    )
    fixture = json.loads(gzip.decompress(raw))
    original = fixture["payload"]
    assert (
        fixture["provenance"]["canonical_payload_sha256"]
        == hashlib.sha256(
            json.dumps(
                original, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    )
    assert len(original["original_evidence"]) == 157
    assert len(original["retained_source_conditions"]) == 26
    assert (
        original["publication_gap"]["data"]["retained_source_conditions"]
        == original["retained_source_conditions"]
    )
    coded = encode_cheap_metadata(original, tokens)
    reference = coded["retained_source_conditions"]
    assert isinstance(reference, dict) and set(reference) == {"value_ref"}
    pool = cast(list[JsonValue], coded["original_metadata_values"])
    assert (
        pool[cast(int, reference["value_ref"])]
        == original["retained_source_conditions"]
    )
    serialized = serialize_cheap_payload(coded, tokens)
    restored = expand_cheap_payload(json.loads(serialized))
    assert restored == original
    assert restored["publication_gap"] == original["publication_gap"]
    quotes = [
        item["requirement"]["witness"]["source_quote"]
        for item in original["retained_source_conditions"]
    ]
    restored_quotes = []
    for item in cast(
        list[dict[str, JsonValue]], restored["retained_source_conditions"]
    ):
        requirement = cast(dict[str, JsonValue], item["requirement"])
        witness = cast(dict[str, JsonValue], requirement["witness"])
        restored_quotes.append(witness["source_quote"])
    assert quotes == restored_quotes
    for row, node in zip(
        original["original_evidence"],
        cast(list[JsonValue], coded["original_evidence"]),
        strict=True,
    ):
        assert json.dumps(row["text"], ensure_ascii=False) in json.dumps(
            node, ensure_ascii=False
        )
    ledger = EvidenceLedger()
    context = RunContext(scope={"captured_condition_repeat": 4998})
    catalogue = {
        row["citation"]: row for row in original["original_metadata_catalogue"]
    }
    for row in original["original_evidence"]:
        identity = catalogue[row["citation"]]
        numbers = ledger.add(
            [
                EvidenceItem(
                    source_id=identity["source_id"],
                    chunk_id=identity["chunk_id"],
                    text=row["text"],
                    metadata=identity["metadata"],
                )
            ],
            context,
        )
        assert numbers == [row["citation"]]
    decoded = decode_compact_originals(
        [restored], expand_cheap_metadata(json.loads(serialized)), ledger
    )
    assert len(decoded) == 157
    for row in decoded:
        identity = catalogue[row["citation"]]
        assert row["metadata"] == identity["metadata"]
        for field in ("source_id", "chunk_id", "text_hash"):
            assert row[field] == identity[field]
        assert (
            hashlib.sha256(cast(str, row["text"]).encode()).hexdigest()
            == identity["text_hash"]
        )
    damaged = copy.deepcopy(coded)
    cast(list[JsonValue], damaged["original_metadata_values"]).append(
        "Unreferenced condition text"
    )
    with pytest.raises(ValueError, match="unused"):
        expand_cheap_payload(damaged)
