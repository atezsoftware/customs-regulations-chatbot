"""A valid evidence ID cannot turn clause a into proof for clause b."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from onyx.db.document_intake_metadata import validate_metadata_receipt_evidence
from onyx.db.models import RegulatoryChunk
from onyx.regulatory.document_intake import metadata


@pytest.mark.parametrize("index", range(5))
def test_actual_false_selected_evidence_fails_construction_and_stored_receipt(
    index: int,
) -> None:
    path = Path(__file__).parent / "fixtures/selected_clause_conflicts.json"
    case = json.loads(path.read_text())[index]
    chunks = [
        metadata.MetadataChunk(
            chunk_id=row["id"],
            text=row["text"],
            heading_path=row["heading_path"],
            chunk_type=row["chunk_type"],
            structural_metadata={},
        )
        for row in (case["target"], case["source"])
    ]
    selected = next(
        entry
        for entry in metadata.build_metadata_evidence_catalog(chunks)
        if entry.source_chunk_id == case["fact"]["source_chunk_id"]
        and entry.text == case["fact"]["evidence_quote"]
        and entry.scope == "heading"
    )
    response = json.dumps(
        {
            "chunks": [
                {
                    "chunk_id": case["target"]["id"],
                    "fields": [
                        {
                            "field": "clause_label",
                            "value": case["value"],
                            "basis": "explicit",
                            "evidence_id": selected.evidence_id,
                            "reference_evidence_id": "",
                        }
                    ],
                }
            ]
        }
    )
    with pytest.raises(ValueError, match="clause locator contradicts"):
        metadata.validate_metadata_transport_response(
            response, chunks=chunks, target_ids=[case["target"]["id"]]
        )
    with pytest.raises(ValueError, match="clause locator contradicts"):
        validate_metadata_receipt_evidence(
            [
                cast(
                    RegulatoryChunk,
                    SimpleNamespace(
                        id=case["target"]["id"],
                        chunk_type=case["target"]["chunk_type"],
                        chunk_metadata={
                            "clause_label": case["value"],
                            "intake_legal_metadata": {
                                "fields": {"clause_label": case["fact"]}
                            },
                        },
                    ),
                )
            ],
            chunks,
        )


@pytest.mark.parametrize(
    ("value", "quote", "conflict"),
    [
        ("b", "a) Makine veya ilgili ürünler", True),
        ("b", "(d) Makine veya ilgili ürünler", True),
        ("c", "(b) Zorlu iş duruşları", True),
        ("b", "b) Bakanlığın gerekçeli talebine", False),
        ("c", "(c) 7.3 maddesinde atıfta bulunulan", False),
        ("çç", "çç) Mücbir sebep", False),
        ("ğğ", "ğğ) Performans planı", False),
        ("ıı", "II) Sıkışık altyapı", False),
        ("ff", "ff) Tip onayı", False),
        ("jj", "jj) Ulusal Yeterlilik Raporu", False),
        ("z", "z) USB Tip-C fişi", False),
        ("z", "istasyonu olarak adlandırılan...\ncc) Yedek parça", False),
        ("b", "Madde13(a) uyarınca işlem yapılır.", False),
        ("b", "a) İlk hüküm\nb) İkinci hüküm", False),
        ("mm, nn", "mm) Birinci hüküm\nnn) İkinci hüküm", False),
        ("c", "ç) Farklı bir bent", True),
    ],
)
def test_only_unambiguous_selected_formal_prefixes_create_contradictions(
    value: str,
    quote: str,
    conflict: bool,
) -> None:
    assert (
        metadata.selected_clause_evidence_contradicts(value=value, evidence_quote=quote)
        is conflict
    )


def test_own_inherited_heading_on_a_coarse_continuation_does_not_become_later_cc() -> (
    None
):
    body = "istasyonu olarak adlandırılan bağlantının devamı,\ncc) Yedek parça güç kaynağı."
    heading = "z) USB Tip-C fişi: Özellikleri..."
    chunk = metadata.MetadataChunk(
        chunk_id="continued",
        text=body,
        heading_path=[heading],
        chunk_type="clause",
        structural_metadata={},
    )
    result = metadata.LegalMetadataResult(
        chunks=[
            metadata.ChunkLegalMetadata(
                chunk_id=chunk.chunk_id,
                fields=[
                    metadata.LegalMetadataFact(
                        field="clause_label",
                        value="z",
                        source_chunk_id=chunk.chunk_id,
                        evidence_quote=heading,
                        basis="explicit",
                    )
                ],
            )
        ]
    )
    assert (
        metadata.validate_metadata_response(
            result.model_dump_json(), chunks=[chunk], target_ids=[chunk.chunk_id]
        )
        == result
    )
