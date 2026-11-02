"""Article locator values must agree with the selected frozen source evidence."""

import json

import pytest

from onyx.regulatory.document_intake import metadata


def _source(
    identifier: str, heading: str, text: str = "(1) Kaynak hüküm."
) -> metadata.MetadataChunk:
    return metadata.MetadataChunk(
        chunk_id=identifier, text=text, heading_path=[heading], chunk_type="paragraph"
    )


def _response(
    target: str, source: metadata.MetadataChunk, quote: str, value: str
) -> str:
    return metadata.LegalMetadataResult(
        chunks=[
            metadata.ChunkLegalMetadata(
                chunk_id=target,
                fields=[
                    metadata.LegalMetadataFact(
                        field="article_no",
                        value=value,
                        source_chunk_id=source.chunk_id,
                        evidence_quote=quote,
                    )
                ],
            )
        ]
    ).model_dump_json()


@pytest.mark.parametrize(
    "selected,value",
    [
        ("MADDE 49 - Kamuoyu ile paylaşılacak veriler", "33"),
        ("MADDE 49 - Kamuoyu ile paylaşılacak veriler", "49"),
        ("GEÇİCİ MADDE 1 - Süre", "1"),
        ("MADDE 1 - Süre", "GEÇİCİ MADDE 1"),
        ("EK MADDE 1 - Süre", "MÜKERRER 1"),
    ],
)
def test_explicit_selected_article_contradiction_is_rejected(
    selected: str, value: str
) -> None:
    source = _source("source", selected)
    target = _source("target", "MADDE 33 - Piyasa katılımcılarının mali sorumlulukları")
    response = _response(target.chunk_id, source, selected, value)
    with pytest.raises(ValueError, match="article locator contradicts"):
        metadata.validate_metadata_response(
            response, chunks=[source, target], target_ids=[target.chunk_id]
        )


@pytest.mark.parametrize(
    "heading,value",
    [
        ("MADDE 33 - Piyasa katılımcılarının mali sorumlulukları", "33"),
        ("GEÇİCİ MADDE 1 - Süre", "GEÇİCİ MADDE 1"),
        ("GEÇİCİ MADDE 2 - Süre", "GEÇİCİ 2"),
        ("EK MADDE 1 - Süre", "EK 1"),
        ("MÜKERRER MADDE 1 - Süre", "MÜKERRER 1"),
    ],
)
def test_matching_article_namespaces_remain_valid(heading: str, value: str) -> None:
    source = _source("source", heading)
    response = _response(source.chunk_id, source, heading, value)
    assert (
        metadata.validate_metadata_response(
            response, chunks=[source], target_ids=[source.chunk_id]
        )
        .chunks[0]
        .fields[0]
        .value
        == value
    )


def test_body_continuation_uses_source_context_without_treating_citation_as_own_article() -> (
    None
):
    source = _source(
        "source", "MADDE 12 - Ücretler", "(4) 49 uncu maddeye göre uygulanır."
    )
    response = _response(source.chunk_id, source, source.text, "12")
    assert metadata.validate_metadata_response(
        response, chunks=[source], target_ids=[source.chunk_id]
    )


@pytest.mark.parametrize("marker", ["MADDE 49", "MADDE49"])
def test_explicit_body_article_can_correct_stale_structural_heading(
    marker: str,
) -> None:
    source = _source(
        "source", "MADDE 33 - Eski başlık", marker + " -\n(1) Gerçek kaynak hüküm."
    )
    response = _response(source.chunk_id, source, source.text, "49")
    assert metadata.validate_metadata_response(
        response, chunks=[source], target_ids=[source.chunk_id]
    )


def test_wrong_transport_article_cannot_survive_but_valid_prior_fact_keeps_provenance() -> (
    None
):
    target = _source("target", "MADDE 33 - Piyasa katılımcılarının mali sorumlulukları")
    wrong = _source("wrong", "MADDE 49 - Kamuoyu ile paylaşılacak veriler")
    chunks = [target, wrong]
    catalog = metadata.build_metadata_evidence_catalog(chunks)

    def transport(source: metadata.MetadataChunk) -> str:
        evidence = next(
            entry
            for entry in catalog
            if entry.source_chunk_id == source.chunk_id and entry.scope == "heading"
        )
        return json.dumps(
            {
                "chunks": [
                    {
                        "chunk_id": "target",
                        "fields": [
                            {
                                "field": "article_no",
                                "value": "33",
                                "evidence_id": evidence.evidence_id,
                                "basis": "explicit",
                                "reference_evidence_id": "",
                            }
                        ],
                    }
                ],
                "warnings": [],
            }
        )

    invalid, prior = transport(wrong), transport(target)
    with pytest.raises(ValueError, match="article locator contradicts"):
        metadata.validate_metadata_transport_response(
            invalid, chunks=chunks, target_ids=["target"]
        )
    recovered = metadata.recover_metadata_response(
        invalid,
        previous_attempts=[("a" * 64, prior)],
        chunks=chunks,
        target_ids=["target"],
    )
    assert recovered is not None
    assert recovered.chunks[0].fields[0].value == "33"
    assert recovered.chunks[0].fields[0].source_chunk_id == "target"
    assert recovered.chunks[0].fields[0].retained_request_hash == "a" * 64
