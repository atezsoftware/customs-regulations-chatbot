"""Metadata recovery retains only strictly verified omitted facts and their origin."""

import json

import pytest
from pydantic import ValidationError

from onyx.regulatory.document_intake import metadata
from tests.unit.onyx.regulatory.test_intake_metadata_value_recovery import (
    _TITLE,
    _entry,
    _fact,
    _response,
    _sources,
)

_HASH = "b" * 64


def _validated(
    sources: list[metadata.MetadataChunk], fields: list[dict[str, str | int]]
) -> metadata.LegalMetadataResult:
    return metadata.validate_metadata_transport_response(
        _response(fields), chunks=sources, target_ids=["section"]
    )


def test_publication_number_survives_omission_with_original_request_provenance() -> (
    None
):
    sources = _sources()
    publication = _entry(
        sources, "2 Temmuz 2021 CUMA Resmî Gazete Sayı : 31529", "body"
    )
    section = _entry(sources, "BEŞİNCİ BÖLÜM", "heading")
    result = _validated(
        sources, [_fact("section_title", "BEŞİNCİ BÖLÜM", section.evidence_id)]
    )
    previous = _response(
        [
            _fact("publication_number", "31529", publication.evidence_id),
            _fact("article_no", "5", "f" * 64),
            _fact(
                "validity_start_date",
                "2021-07-02",
                _entry(sources, _TITLE, "heading").evidence_id,
            ),
        ]
    )
    previous_payload = json.loads(previous)
    uncertainty = (
        "Publication number 31529 is explicit; publication date remains uncertain."
    )
    previous_payload["warnings"] = [uncertainty]
    previous = json.dumps(previous_payload)
    merged = metadata.retain_validated_metadata_facts(
        result, previous, _HASH, sources, ["section"]
    )
    by_field = {fact.field: fact for fact in merged.chunks[0].fields}
    assert set(by_field) == {"section_title", "publication_number"}
    assert by_field["publication_number"].value == "31529"
    assert by_field["publication_number"].retained_request_hash == _HASH
    assert by_field["section_title"].retained_request_hash is None
    assert merged.warnings == [f"Retained metadata from request {_HASH}: {uncertainty}"]
    assert "retained_request_hash" not in merged.model_dump_json()
    assert "retained_request_hash" not in json.dumps(metadata._response_schema())
    assert len(result.chunks[0].fields) == 1
    metadata.validate_metadata_response(
        merged.model_dump_json(), chunks=sources, target_ids=["section"]
    )


def test_new_fact_has_priority_over_different_independently_valid_old_fact() -> None:
    sources = _sources()
    heading = _entry(sources, "BEŞİNCİ BÖLÜM", "heading")
    body = _entry(sources, sources[0].text, "body")
    result = _validated(
        sources, [_fact("section_title", "BEŞİNCİ BÖLÜM", heading.evidence_id)]
    )
    previous = _response([_fact("section_title", sources[0].text, body.evidence_id)])
    merged = metadata.retain_validated_metadata_facts(
        result, previous, _HASH, sources, ["section"]
    )
    assert merged.chunks[0].fields == result.chunks[0].fields


@pytest.mark.parametrize(
    "case",
    ["malformed", "schema", "wrong_target", "duplicate_target", "duplicate_field"],
)
def test_untrusted_previous_response_shape_retains_nothing(case: str) -> None:
    sources = _sources()
    publication = _entry(
        sources, "2 Temmuz 2021 CUMA Resmî Gazete Sayı : 31529", "body"
    )
    previous = _response(
        [_fact("publication_number", "31529", publication.evidence_id)]
    )
    raw = json.loads(previous)
    if case == "malformed":
        previous = "{broken"
    elif case == "schema":
        raw["unexpected"] = True
    elif case == "wrong_target":
        raw["chunks"][0]["chunk_id"] = "foreign"
    elif case == "duplicate_target":
        raw["chunks"] *= 2
    else:
        raw["chunks"][0]["fields"] *= 2
    if case != "malformed":
        previous = json.dumps(raw)
    result = _validated(sources, [])
    assert (
        metadata.retain_validated_metadata_facts(
            result, previous, _HASH, sources, ["section"]
        )
        is result
    )


def test_merge_revalidates_cross_field_dates_instead_of_accepting_two_valid_facts() -> (
    None
):
    sources = [
        metadata.MetadataChunk(
            chunk_id="section",
            text="Bu Yönetmelik 1/1/2022 tarihinde yürürlüğe girer.\nBu Yönetmelik 1/1/2021 tarihinde sona erer.",
            heading_path=[],
            chunk_type="article",
        )
    ]
    start = _entry(sources, "Bu Yönetmelik 1/1/2022 tarihinde yürürlüğe girer.", "body")
    end = _entry(sources, "Bu Yönetmelik 1/1/2021 tarihinde sona erer.", "body")
    result = _validated(
        sources, [_fact("validity_start_date", "2022-01-01", start.evidence_id)]
    )
    previous = _response([_fact("validity_end_date", "2021-01-01", end.evidence_id)])
    metadata.validate_metadata_transport_response(
        previous, chunks=sources, target_ids=["section"]
    )
    with pytest.raises(ValueError, match="validity end must be after its start"):
        metadata.retain_validated_metadata_facts(
            result, previous, _HASH, sources, ["section"]
        )


def test_retained_fact_field_is_validated_and_remains_outside_wire_payload() -> None:
    fact = metadata.LegalMetadataFact(
        field="publication_number",
        value="31529",
        source_chunk_id="source",
        evidence_quote="Sayı : 31529",
        retained_request_hash=_HASH,
    )
    assert fact.retained_request_hash == _HASH
    assert "retained_request_hash" not in fact.model_dump()
    with pytest.raises(ValidationError):
        metadata.LegalMetadataFact.model_validate(
            {**fact.model_dump(), "retained_request_hash": "forged"}
        )


def test_retention_fills_last_distinct_field_within_eighteen_fact_limit() -> None:
    sources = [
        metadata.MetadataChunk(
            chunk_id="section",
            text=(
                "Sayı : 31529\nBu Yönetmelik 1/1/2021 tarihinde yürürlüğe girer.\n"
                "Bu Yönetmelik 1/1/2022 tarihinde sona erer."
            ),
            heading_path=[],
            chunk_type="article",
        )
    ]
    evidence = _entry(sources, sources[0].text, "body")
    fields = []
    for field in sorted(metadata.LEGAL_METADATA_FIELDS | metadata.VALIDITY_FIELDS):
        if field == "publication_number":
            continue
        value: str | int = (
            0
            if field in metadata.INTEGER_FIELDS
            else "2022-01-01"
            if field == "validity_end_date"
            else "2021-01-01"
            if field in metadata.DATE_FIELDS
            else "31529"
        )
        fields.append(_fact(field, value, evidence.evidence_id))
    assert len(fields) == 17
    result = _validated(sources, fields)
    previous = _response([_fact("publication_number", "31529", evidence.evidence_id)])
    merged = metadata.retain_validated_metadata_facts(
        result, previous, _HASH, sources, ["section"]
    )
    assert len(merged.chunks[0].fields) == 18
    assert merged.chunks[0].fields[-1].field == "publication_number"
    assert merged.chunks[0].fields[-1].retained_request_hash == _HASH


@pytest.mark.parametrize("restore", [False, True])
def test_retained_warning_deduplication_and_no_restoration_preserves_current_warnings(
    restore: bool,
) -> None:
    sources = _sources()
    publication = _entry(
        sources, "2 Temmuz 2021 CUMA Resmî Gazete Sayı : 31529", "body"
    )
    fact = _fact("publication_number", "31529", publication.evidence_id)
    result = _validated(sources, [] if restore else [fact]).model_copy(
        update={"warnings": ["Publication uncertainty"]}
    )
    previous = json.loads(_response([fact]))
    previous["warnings"] = ["Publication uncertainty", "Check the source header"]
    merged = metadata.retain_validated_metadata_facts(
        result, json.dumps(previous), _HASH, sources, ["section"]
    )
    assert merged.warnings == (
        [
            "Publication uncertainty",
            f"Retained metadata from request {_HASH}: Check the source header",
        ]
        if restore
        else result.warnings
    )


def test_retained_warnings_overflow_fails_without_truncating_uncertainty() -> None:
    sources = _sources()
    publication = _entry(
        sources, "2 Temmuz 2021 CUMA Resmî Gazete Sayı : 31529", "body"
    )
    result = _validated(sources, []).model_copy(
        update={"warnings": [f"Current uncertainty {index}" for index in range(32)]}
    )
    previous = json.loads(
        _response([_fact("publication_number", "31529", publication.evidence_id)])
    )
    previous["warnings"] = ["Retained publication uncertainty"]
    with pytest.raises(ValidationError):
        metadata.retain_validated_metadata_facts(
            result, json.dumps(previous), _HASH, sources, ["section"]
        )
    assert len(result.warnings) == 32
