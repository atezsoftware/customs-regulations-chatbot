"""Complementary ACKs repair only exact fields with independently valid evidence."""

import json

import pytest

from onyx.regulatory.document_intake import metadata
from tests.unit.onyx.regulatory.test_intake_metadata_value_recovery import (
    _TITLE,
    _entry,
    _fact,
    _response,
    _sources,
)

_HASH = "c" * 64
_DATE = "(1) Bu Yönetmelik 1/1/2022 tarihinde yürürlüğe girer."


def _date_sources() -> list[metadata.MetadataChunk]:
    return [
        *_sources(),
        metadata.MetadataChunk(
            chunk_id="commencement",
            text=f"MADDE 28 –\n{_DATE}",
            heading_path=[_TITLE, "MADDE 28 - Yürürlük", _DATE],
            chunk_type="paragraph",
        ),
    ]


def test_correct_current_section_and_prior_body_date_compose_without_losing_origin() -> (
    None
):
    sources = _date_sources()
    section = _entry(sources, "BEŞİNCİ BÖLÜM", "heading")
    body = _entry(sources, sources[0].text, "body")
    previous = json.loads(
        _response(
            [
                _fact("section_title", "BEŞİNCİ BÖLÜM", body.evidence_id),
                _fact(
                    "validity_start_date",
                    "2022-01-01",
                    _entry(sources, _DATE, "body").evidence_id,
                ),
            ]
        )
    )
    previous["warnings"] = ["Check this provision's commencement exceptions"]
    current = _response(
        [
            _fact("section_title", "BEŞİNCİ BÖLÜM", section.evidence_id),
            _fact(
                "validity_start_date",
                "2022-01-01",
                _entry(sources, _DATE, "heading").evidence_id,
            ),
        ]
    )
    result = metadata.recover_metadata_response(
        current,
        previous_attempts=[(_HASH, json.dumps(previous))],
        chunks=sources,
        target_ids=["section"],
    )
    assert result is not None
    facts = {fact.field: fact for fact in result.chunks[0].fields}
    assert facts["section_title"].value == "BEŞİNCİ BÖLÜM"
    assert facts["section_title"].retained_request_hash is None
    assert facts["validity_start_date"].evidence_quote == _DATE
    assert facts["validity_start_date"].retained_request_hash == _HASH
    assert result.warnings == [
        f"Retained metadata from request {_HASH}: Check this provision's commencement exceptions"
    ]
    assert "retained_request_hash" not in result.model_dump_json()
    metadata.validate_metadata_response(
        result.model_dump_json(), chunks=sources, target_ids=["section"]
    )


def test_unknown_current_evidence_id_requires_prior_valid_fact_for_the_same_field() -> (
    None
):
    sources = _sources()
    section = _entry(sources, "BEŞİNCİ BÖLÜM", "heading")
    current = _response([_fact("section_title", "BEŞİNCİ BÖLÜM", "f" * 64)])
    prior = _response([_fact("section_title", "BEŞİNCİ BÖLÜM", section.evidence_id)])
    result = metadata.recover_metadata_response(
        current,
        previous_attempts=[(_HASH, prior)],
        chunks=sources,
        target_ids=["section"],
    )
    assert result is not None
    assert result.chunks[0].fields[0].retained_request_hash == _HASH
    publication = _entry(
        sources, "2 Temmuz 2021 CUMA Resmî Gazete Sayı : 31529", "body"
    )
    unrelated = _response(
        [_fact("publication_number", "31529", publication.evidence_id)]
    )
    assert (
        metadata.recover_metadata_response(
            current,
            previous_attempts=[(_HASH, unrelated)],
            chunks=sources,
            target_ids=["section"],
        )
        is None
    )
    assert (
        metadata.recover_metadata_response(
            current, previous_attempts=[], chunks=sources, target_ids=["section"]
        )
        is None
    )


def test_newest_prior_valid_value_wins_when_attempts_arrive_chronologically() -> None:
    sources = _sources()
    heading = _entry(sources, "BEŞİNCİ BÖLÜM", "heading")
    body = _entry(sources, sources[0].text, "body")
    older = _response([_fact("section_title", "BEŞİNCİ BÖLÜM", heading.evidence_id)])
    newer = _response([_fact("section_title", body.text, body.evidence_id)])
    current = _response([_fact("section_title", "BEŞİNCİ BÖLÜM", "f" * 64)])
    result = metadata.recover_metadata_response(
        current,
        previous_attempts=[("a" * 64, older), (_HASH, newer)],
        chunks=sources,
        target_ids=["section"],
    )
    assert result is not None
    assert result.chunks[0].fields[0].value == body.text
    assert result.chunks[0].fields[0].retained_request_hash == _HASH


def test_valid_same_field_on_another_requested_target_cannot_repair_current_target() -> (
    None
):
    sources = _sources()
    section = _entry(sources, "BEŞİNCİ BÖLÜM", "heading")
    sources.append(sources[0].model_copy(update={"chunk_id": "other"}))
    current = json.loads(_response([_fact("section_title", "BEŞİNCİ BÖLÜM", "f" * 64)]))
    current["chunks"].append({"chunk_id": "other", "fields": []})
    previous = json.loads(_response([]))
    previous["chunks"].append(
        {
            "chunk_id": "other",
            "fields": [_fact("section_title", "BEŞİNCİ BÖLÜM", section.evidence_id)],
        }
    )
    assert (
        metadata.recover_metadata_response(
            json.dumps(current),
            previous_attempts=[(_HASH, json.dumps(previous))],
            chunks=sources,
            target_ids=["section", "other"],
        )
        is None
    )


@pytest.mark.parametrize(
    "case",
    ["malformed", "schema", "wrong_target", "duplicate_target", "duplicate_field"],
)
def test_invalid_current_response_shape_never_authorizes_fact_recovery(
    case: str,
) -> None:
    sources = _sources()
    section = _entry(sources, "BEŞİNCİ BÖLÜM", "heading")
    prior = _response([_fact("section_title", "BEŞİNCİ BÖLÜM", section.evidence_id)])
    current = json.loads(prior)
    if case == "schema":
        current["extra"] = True
    elif case == "wrong_target":
        current["chunks"][0]["chunk_id"] = "foreign"
    elif case == "duplicate_target":
        current["chunks"] *= 2
    elif case == "duplicate_field":
        current["chunks"][0]["fields"] *= 2
    assert (
        metadata.recover_metadata_response(
            "{broken" if case == "malformed" else json.dumps(current),
            previous_attempts=[(_HASH, prior)],
            chunks=sources,
            target_ids=["section"],
        )
        is None
    )


def test_complementary_facts_still_reject_cross_field_validity_conflicts() -> None:
    sources = _date_sources()
    end = "Bu Yönetmelik 1/1/2021 tarihinde sona erer."
    sources[0] = sources[0].model_copy(update={"text": end})
    prior = _response(
        [
            _fact(
                "validity_start_date",
                "2022-01-01",
                _entry(sources, _DATE, "body").evidence_id,
            )
        ]
    )
    current = _response(
        [
            _fact(
                "validity_start_date",
                "2022-01-01",
                _entry(sources, _DATE, "heading").evidence_id,
            ),
            _fact(
                "validity_end_date",
                "2021-01-01",
                _entry(sources, end, "body").evidence_id,
            ),
        ]
    )
    with pytest.raises(ValueError, match="validity end must be after its start"):
        metadata.recover_metadata_response(
            current,
            previous_attempts=[(_HASH, prior)],
            chunks=sources,
            target_ids=["section"],
        )
