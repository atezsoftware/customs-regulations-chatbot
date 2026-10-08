"""Navigation sharing preserves every original identity and exact JSON value."""

import hashlib
import json
import random
from typing import cast
from unittest.mock import Mock

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, RunContext
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import SearchDoc
from onyx.legal_composite.engine import (
    LegalCompositeEngine,
    SourceAcquirer,
    _compact_original_catalogue,
)
from onyx.legal_composite.models import WorkflowPolicy
from onyx.llm.utils import check_number_of_tokens

CATALOGUE_KEYS = {
    "title",
    "article_no",
    "paragraph_no",
    "clause_label",
    "heading_path",
}


def serialized(value: JsonValue) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def decode_catalogue(wire: JsonValue) -> list[dict[str, JsonValue]]:
    """An independent consumer of the documented wire contract, after JSON transport."""
    wire = json.loads(serialized(wire))
    if isinstance(wire, list):
        assert all(isinstance(row, dict) for row in wire)
        return cast(list[dict[str, JsonValue]], wire)
    assert isinstance(wire, dict)
    assert wire["codec"] == "shared_metadata_v1"
    assert wire["row_fields"] == [
        "citation",
        "source_id_ref",
        "chunk_id",
        "metadata_ref",
    ]
    values, metadata, rows = wire["values"], wire["metadata"], wire["rows"]
    assert isinstance(values, list)
    assert isinstance(metadata, list)
    assert isinstance(rows, list)
    result: list[dict[str, JsonValue]] = []
    for row in rows:
        assert isinstance(row, list) and len(row) == 4
        citation, source_ref, chunk, metadata_ref = row
        assert type(source_ref) is int and 0 <= source_ref < len(values)
        assert type(metadata_ref) is int and 0 <= metadata_ref < len(metadata)
        fields = metadata[metadata_ref]
        assert isinstance(fields, dict)
        decoded_metadata: dict[str, JsonValue] = {}
        for key, value_ref in fields.items():
            assert type(value_ref) is int and 0 <= value_ref < len(values)
            decoded_metadata[key] = values[value_ref]
        result.append(
            {
                "citation": citation,
                "source_id": values[source_ref],
                "chunk_id": chunk,
                "metadata": decoded_metadata,
            }
        )
    return result


@pytest.mark.parametrize(
    "field_value",
    [
        True,
        1,
        1.0,
        False,
        0,
        -0.0,
        None,
        "",
        [],
        {},
        ["İşlem", "I\u0307şlem", "條款", "🚢"],
        {"source_id_ref": 0, "$string": 2, "nested": [None, "ş"]},
    ],
)
def test_shared_values_round_trip_json_types_unicode_and_reference_like_objects(
    field_value: JsonValue,
) -> None:
    rows: list[dict[str, JsonValue]] = [
        {
            "citation": number,
            "source_id": "a7fa397a-9743-5dc7-842a-8a0e9889d05c",
            "chunk_id": None if number == 1 else f"bölüm-🚢-{number}",
            "metadata": {
                "title": "Şartlar ve istisnalar — kaynak başlığı " * 8,
                "article_no": field_value,
                "heading_path": ["İlk bölüm", "İkinci bölüm"],
            },
        }
        for number in range(1, 9)
    ]
    before = serialized(cast(JsonValue, rows))
    wire = _compact_original_catalogue(rows)
    assert isinstance(wire, dict)
    assert serialized(cast(JsonValue, decode_catalogue(wire))) == before
    assert serialized(cast(JsonValue, rows)) == before
    assert len(cast(list[JsonValue], wire["metadata"])) == 1


def test_type_distinct_values_missing_fields_and_order_cannot_alias() -> None:
    metadata: list[dict[str, JsonValue]] = [
        {"title": "Ortak kaynak — koşul inceleme başlığı " * 12, "article_no": value}
        for value in (True, 1, 1.0, None, [], {}, "", -0.0, 0)
    ]
    metadata.extend(
        [
            {"title": metadata[0]["title"]},
            {"article_no": 1, "title": metadata[0]["title"]},
            {
                "title": metadata[0]["title"],
                "heading_path": {"İ": "bir", "ş": "iki"},
            },
            {
                "title": metadata[0]["title"],
                "heading_path": {"ş": "iki", "İ": "bir"},
            },
        ]
    )
    rows: list[dict[str, JsonValue]] = [
        {
            "citation": number + 1,
            "source_id": "same-source",
            "chunk_id": f"chunk-{number}",
            "metadata": fields,
        }
        for number, fields in enumerate(metadata)
    ]
    wire = _compact_original_catalogue(rows)
    assert isinstance(wire, dict)
    decoded = decode_catalogue(wire)
    assert serialized(cast(JsonValue, decoded)) == serialized(cast(JsonValue, rows))
    assert len(cast(list[JsonValue], wire["metadata"])) == len(metadata)
    values = cast(list[JsonValue], wire["values"])
    signatures = [serialized(value) for value in values]
    assert all(signatures.count(value) == 1 for value in ("true", "1", "1.0"))


def test_seeded_mixed_catalogue_round_trip_preserves_every_row_and_value() -> None:
    random_source = random.Random(20261008)
    candidates: list[JsonValue] = [
        True,
        1,
        1.0,
        None,
        [],
        {},
        "",
        "İşlem / İşlem\nنص\t條款 🚢",
        ["Kısım", {"s": 0}, "bölüm"],
        {"path": ["Şart", None, 1.0]},
    ]
    for _ in range(80):
        rows: list[dict[str, JsonValue]] = []
        for number in range(1, random_source.randint(1, 64)):
            keys = random_source.sample(
                sorted(CATALOGUE_KEYS), k=random_source.randrange(6)
            )
            metadata = {key: random_source.choice(candidates) for key in keys}
            rows.append(
                {
                    "citation": number * 3,
                    "source_id": random_source.choice(
                        ["source-İ", "source-條款", "source-🚢"]
                    ),
                    "chunk_id": random_source.choice([None, "", f"chunk-{number}"]),
                    "metadata": metadata,
                }
            )
        random_source.shuffle(rows)
        before = serialized(cast(JsonValue, rows))
        wire = _compact_original_catalogue(rows)
        assert serialized(cast(JsonValue, decode_catalogue(wire))) == before
        assert serialized(cast(JsonValue, rows)) == before
        assert len(json.dumps(wire, ensure_ascii=False)) <= len(
            json.dumps(rows, ensure_ascii=False)
        )


@pytest.mark.parametrize("empty", [True, False])
def test_small_catalogue_uses_exact_original_list_when_pooling_would_expand_it(
    empty: bool,
) -> None:
    rows: list[dict[str, JsonValue]] = (
        []
        if empty
        else [{"citation": 1, "source_id": "s", "chunk_id": "c", "metadata": {}}]
    )
    assert _compact_original_catalogue(rows) is rows


def mixed_engine() -> LegalCompositeEngine:
    ledger = EvidenceLedger()
    items: list[EvidenceItem] = []
    for index in range(63):
        source_index = index % 3
        article = index // 9 + 1
        source_id = f"00000000-0000-4000-8000-{source_index + 1:012d}"
        title = (
            "İşlemlerin Koşulları, Kapsamı ve İstisnalarının İncelenmesine "
            f"İlişkin Özgün Kaynak {source_index + 1}"
        )
        text = (
            f"Özgün kaynak {source_index + 1}, hüküm {index + 1}. "
            "Bir işlemin kapsamı, koşulu ve istisnası birlikte incelenir. "
            "Bu sentetik metin navigasyon içeriği veya gerçek bir hukuk kuralı değildir. "
        ) * 3
        if index == 0:
            text += "Bu uzun özgün bölüm bütüne sığmıyorsa katalogda kalır. " * 1_000
        items.append(
            EvidenceItem(
                source_id=source_id,
                chunk_id=f"00000000-0000-4000-9000-{index + 1:012d}",
                text=text,
                metadata={
                    "title": title,
                    "article_no": str(article),
                    "paragraph_no": index // 3 % 3 + 1,
                    "clause_label": None if index % 2 else "ç",
                    "heading_path": [
                        title,
                        "Birinci Kısım — Genel Esaslar",
                        "İkinci Bölüm — Koşulların Uygulanması",
                        f"Madde {article} — Kapsam ve İstisnalar",
                    ],
                },
                search_doc=SearchDoc(
                    document_id=source_id,
                    chunk_ind=index,
                    semantic_identifier=title,
                    blurb=text,
                    source_type=DocumentSource.FILE,
                    boost=0,
                    hidden=False,
                    metadata={},
                    match_highlights=[],
                ),
            )
        )
    ledger.add(items, RunContext())
    acquirer = Mock(spec=SourceAcquirer)
    acquirer.definitions.return_value = [
        {"name": "read_evidence", "description": "Read one recorded complete original."}
    ]
    engine = LegalCompositeEngine(
        gateway=Mock(),
        acquirer=acquirer,
        ledger=ledger,
        policy=WorkflowPolicy(),
        check_active=lambda: None,
        research_available=lambda: True,
    )
    engine.receipts = [
        {
            "need_ids": ["scope", "conditions"],
            "status": "found",
            "citations": [2, 13, 24],
            "summary": "Üç özgün kaynak kaydedildi; kapsam ayrıca incelenmelidir.",
            "data": {
                "navigation": ["Koşullar", "İstisnalar", "Sonuçlar"],
                "nested": {"source_id_ref": 0, "values": [True, 1, 1.0, None]},
            },
        }
    ]
    return engine


def unshared_catalogue(ledger: EvidenceLedger) -> list[dict[str, JsonValue]]:
    return [
        {
            "citation": row["citation"],
            "source_id": row["source_id"],
            "chunk_id": row["chunk_id"],
            "metadata": {
                key: value for key, value in fields.items() if key in CATALOGUE_KEYS
            },
        }
        for row in ledger.provision_metadata()
        if isinstance(fields := row["metadata"], dict)
    ]


def test_mixed_63_catalogue_preserves_omitted_originals_receipts_and_final_phase() -> (
    None
):
    engine = mixed_engine()
    before = engine.ledger.provision_metadata()
    receipt_json = serialized(cast(JsonValue, engine.receipts))
    expected_catalogue = unshared_catalogue(engine.ledger)
    research = engine._payload("Kaynak kapsamını incele.", "Önceki konuşma: ş/İ")
    final = engine._payload(
        "Kaynak kapsamını incele.", "Önceki konuşma: ş/İ", source_phase=False
    )
    decoded = decode_catalogue(research["original_catalogue"])
    assert serialized(cast(JsonValue, decoded)) == serialized(
        cast(JsonValue, expected_catalogue)
    )
    assert len(decoded) == 63
    assert research["original_evidence"] == final["original_evidence"]
    records = cast(list[dict[str, JsonValue]], research["original_evidence"])
    delivered = {cast(int, row["citation"]) for row in records}
    omitted = set(cast(list[int], research["omitted_original_ids"]))
    assert omitted and 1 in omitted
    assert omitted == set(engine.ledger.citation_numbers()) - delivered
    assert {row["citation"] for row in decoded} == delivered | omitted
    for row in records:
        item = engine.ledger.get(cast(int, row["citation"]))
        assert item is not None
        assert row["source_id"] == item.source_id and row["chunk_id"] == item.chunk_id
        assert row["text"] == item.text
        assert (
            hashlib.sha256(cast(str, row["text"]).encode()).hexdigest()
            == item.text_hash
        )
    assert engine.ledger.provision_metadata() == before
    assert serialized(research["receipts"]) == receipt_json
    assert serialized(cast(JsonValue, engine.receipts)) == receipt_json
    assert final["receipts"] == [
        {
            "need_ids": ["scope", "conditions"],
            "status": "found",
            "citations": [2, 13, 24],
        }
    ]
    assert "original_catalogue" not in final and "tools" not in final


def test_mixed_63_catalogue_reduces_full_source_phase_input_tokens_without_shrinking() -> (
    None
):
    engine = mixed_engine()
    compact = engine._payload("Kaynak kapsamını incele.", "")
    baseline = {**compact, "original_catalogue": unshared_catalogue(engine.ledger)}
    compact_json = json.dumps(compact, ensure_ascii=False)
    baseline_json = json.dumps(baseline, ensure_ascii=False)
    compact_tokens = check_number_of_tokens(compact_json)
    baseline_tokens = check_number_of_tokens(baseline_json)
    raw_catalogue_tokens = check_number_of_tokens(
        json.dumps(baseline["original_catalogue"], ensure_ascii=False)
    )
    compact_catalogue_tokens = check_number_of_tokens(
        json.dumps(compact["original_catalogue"], ensure_ascii=False)
    )
    assert compact_catalogue_tokens < raw_catalogue_tokens * 0.7
    assert compact_tokens < baseline_tokens
    assert {
        key: value for key, value in baseline.items() if key != "original_catalogue"
    } == {key: value for key, value in compact.items() if key != "original_catalogue"}
    print(
        json.dumps(
            {
                "fixture": "synthetic_mixed_63_three_sources_seven_articles",
                "tokenizer": "cl100k_base",
                "baseline_catalogue_tokens": raw_catalogue_tokens,
                "compact_catalogue_tokens": compact_catalogue_tokens,
                "baseline_full_payload_tokens": baseline_tokens,
                "compact_full_payload_tokens": compact_tokens,
                "baseline_full_payload_chars": len(baseline_json),
                "compact_full_payload_chars": len(compact_json),
                "catalogue_rows": 63,
                "original_records_unchanged": True,
                "receipts_unchanged": True,
            },
            sort_keys=True,
        )
    )
