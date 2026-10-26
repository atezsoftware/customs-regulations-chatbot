"""Evidence IDs select immutable exact spans before unchanged legal validation."""

import json

import pytest

from onyx.regulatory.document_intake import metadata
from tests.unit.onyx.regulatory.test_intake_metadata import _sources


def _response(evidence_id: str, **changes: object) -> str:
    return json.dumps(
        {
            "chunks": [
                {
                    "chunk_id": "article",
                    "fields": [
                        {
                            "field": "document_title",
                            "value": "Yeni TEBLİĞ",
                            "evidence_id": evidence_id,
                            "basis": "explicit",
                            "reference_evidence_id": "",
                            **changes,
                        }
                    ],
                }
            ],
            "warnings": [],
        }
    )


def test_exact_full_heading_hydrates_turkish_unicode_without_copying_quotes() -> None:
    source = metadata.MetadataChunk(
        chunk_id="article",
        text="Yeni TEBLİĞ\nKural uygulanır.",
        heading_path=["Yeni TEBLİĞ"],
        chunk_type="article",
    )
    entries = metadata.build_metadata_evidence_catalog([source])
    heading = next(entry for entry in entries if entry.scope == "heading")
    result = metadata.validate_metadata_transport_response(
        _response(heading.evidence_id), chunks=[source], target_ids=[source.chunk_id]
    )
    fact = result.chunks[0].fields[0]
    assert fact.evidence_quote == "Yeni TEBLİĞ"
    assert fact.source_chunk_id == source.chunk_id
    assert (
        metadata.validate_metadata_response(
            result.model_dump_json(), chunks=[source], target_ids=[source.chunk_id]
        )
        == result
    )


def test_catalog_preserves_full_bodies_original_lines_and_contiguous_long_spans() -> (
    None
):
    text = "  İlk satır\r\n\r\n5 Ocak 2025\n" + "z" * 4100
    source = metadata.MetadataChunk(
        chunk_id="article",
        text=text,
        heading_path=["Tam Başlık", "x" * 2100],
        chunk_type="article",
    )
    entries = metadata.build_metadata_evidence_catalog([source])
    for entry in entries:
        if entry.scope == "body":
            assert entry.text == text[entry.start : entry.end]
    assert any(entry.text == "  İlk satır" for entry in entries)
    assert any(entry.text == "5 Ocak 2025" for entry in entries)
    assert all(0 < len(entry.text) <= 2000 for entry in entries)
    assert [entry.text for entry in entries if entry.scope == "heading"] == [
        "Tam Başlık"
    ]
    full_slices = [
        next(
            entry
            for entry in entries
            if entry.scope == "body"
            and entry.start == start
            and entry.end == min(start + 2000, len(text))
        )
        for start in range(0, len(text), 2000)
    ]
    assert "".join(entry.text for entry in full_slices) == text


def test_catalog_ids_bind_source_scope_location_and_exact_bytes() -> None:
    source = metadata.MetadataChunk(
        chunk_id="article",
        text="Yeni TEBLİĞ",
        heading_path=["Yeni TEBLİĞ"],
        chunk_type="article",
    )
    catalog = metadata.build_metadata_evidence_catalog([source])
    assert catalog == metadata.build_metadata_evidence_catalog([source])
    assert len({entry.evidence_id for entry in catalog}) == 2
    assert not {entry.evidence_id for entry in catalog} & {
        entry.evidence_id
        for entry in metadata.build_metadata_evidence_catalog(
            [source.model_copy(update={"chunk_id": "another-file"})]
        )
    }
    assert not {entry.evidence_id for entry in catalog} & {
        entry.evidence_id
        for entry in metadata.build_metadata_evidence_catalog(
            [
                source.model_copy(
                    update={"text": "Yeni TEBLIĞ", "heading_path": ["Yeni TEBLIĞ"]}
                )
            ]
        )
    }


@pytest.mark.parametrize(
    "case",
    [
        "unknown",
        "other-file",
        "stale",
        "explicit-unused-reference",
        "invented-reference",
        "heading-date",
        "heading-publication-reference",
        "relative-nondate",
        "partial-title",
        "legacy-quote",
    ],
)
def test_invalid_or_wrong_scope_ids_never_override_strict_validation(case: str) -> None:
    source = metadata.MetadataChunk(
        chunk_id="article",
        text="2026-10-26\nYeni TEBLİĞ\nBu Kanun yayımı tarihinde yürürlüğe girer.",
        heading_path=["Yeni TEBLİĞ", "2026-10-26"],
        chunk_type="article",
    )
    catalog = metadata.build_metadata_evidence_catalog([source])
    body = next(
        entry
        for entry in catalog
        if entry.scope == "body" and entry.text == "Yeni TEBLİĞ"
    )
    heading_date = next(
        entry
        for entry in catalog
        if entry.scope == "heading" and entry.text == "2026-10-26"
    )
    selected = body.evidence_id
    changes: dict[str, object] = {}
    if case == "unknown":
        selected = "missing"
    elif case in {"other-file", "stale"}:
        altered = (
            source.model_copy(update={"chunk_id": "other-file"})
            if case == "other-file"
            else source.model_copy(
                update={"text": source.text.replace("TEBLİĞ", "TEBLIĞ")}
            )
        )
        selected = next(
            entry.evidence_id
            for entry in metadata.build_metadata_evidence_catalog([altered])
            if entry.scope == "body" and entry.text.startswith("Yeni ")
        )
    elif case in {"explicit-unused-reference", "invented-reference"}:
        changes["reference_evidence_id"] = (
            body.evidence_id if case == "explicit-unused-reference" else "missing"
        )
    elif case == "heading-date":
        selected = heading_date.evidence_id
        changes.update(field="validity_start_date", value="2026-10-26")
    elif case == "heading-publication-reference":
        selected = next(
            entry.evidence_id
            for entry in catalog
            if entry.scope == "body" and "yayımı" in entry.text
        )
        changes.update(
            field="validity_start_date",
            value="2026-10-26",
            basis="relative_publication",
            reference_evidence_id=heading_date.evidence_id,
        )
    elif case == "relative-nondate":
        changes.update(
            basis="relative_publication",
            reference_evidence_id=next(
                entry.evidence_id
                for entry in catalog
                if entry.scope == "body" and entry.text == "2026-10-26"
            ),
        )
    elif case == "partial-title":
        changes["value"] = "Yeni TEBLİĞ Tam Metni"
    else:
        changes["evidence_quote"] = "untrusted injected quote"
    with pytest.raises(ValueError):
        metadata.validate_metadata_transport_response(
            _response(selected, **changes),
            chunks=[source],
            target_ids=[source.chunk_id],
        )


def test_relative_date_uses_single_body_publication_line_not_multidate_body() -> None:
    chunks = _sources()
    chunks[0] = chunks[0].model_copy(
        update={"text": chunks[0].text + "\nAtıf: 01.01.1999"}
    )
    catalog = metadata.build_metadata_evidence_catalog(chunks)
    publication = next(
        entry
        for entry in catalog
        if entry.scope == "body" and entry.text == "Resmî Gazete Tarihi: 26.10.2026"
    )
    closing = next(
        entry
        for entry in catalog
        if entry.source_chunk_id == "closing" and entry.scope == "body"
    )
    result = metadata.validate_metadata_transport_response(
        _response(
            closing.evidence_id,
            field="validity_start_date",
            value="2026-10-26",
            basis="relative_publication",
            reference_evidence_id=publication.evidence_id,
        ),
        chunks=chunks,
        target_ids=["article"],
    )
    assert result.chunks[0].fields[0].reference_quote == publication.text
    assert result.chunks[0].fields[0].value == "2026-10-26"


def test_v2_request_retains_complete_context_schema_budget_and_separate_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = _sources()
    request = metadata.build_metadata_request(chunks, ["article"])
    payload = json.loads(request.prompt)
    assert payload["version"] == metadata.METADATA_VERSION == 2
    assert payload["complete_document"] == [
        chunk.model_dump(mode="json") for chunk in chunks
    ]
    assert payload["evidence_catalog"]
    assert (
        request.generation_config
        and request.generation_config["maxOutputTokens"] == 32768
    )
    assert request.generation_config["thinkingConfig"] == {"thinkingLevel": "medium"}
    v1_payload = {
        key: value for key, value in payload.items() if key != "evidence_catalog"
    }
    v1_payload["version"] = 1
    assert (
        request.request_hash
        != request.model_copy(update={"prompt": json.dumps(v1_payload)}).request_hash
    )
    monkeypatch.setattr(
        metadata,
        "MAX_METADATA_REQUEST_BYTES",
        len(request.to_generate_content_request().__str__().encode()) // 2,
    )
    with pytest.raises(ValueError, match="8 MiB"):
        metadata.build_metadata_request(chunks, ["article"])


def test_catalog_entry_budget_stops_growth_without_truncating_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = metadata.MetadataChunk(
        chunk_id="article", text="a\nb", heading_path=[], chunk_type="article"
    )
    monkeypatch.setattr(metadata, "MAX_METADATA_CATALOG_ENTRIES", 3)
    assert len(metadata.build_metadata_evidence_catalog([source])) == 3
    monkeypatch.setattr(metadata, "MAX_METADATA_CATALOG_ENTRIES", 2)
    with pytest.raises(ValueError, match="entry budget; source was not truncated"):
        metadata.build_metadata_request([source], [source.chunk_id])
    assert source.text == "a\nb"
    monkeypatch.setattr(metadata, "MAX_METADATA_CATALOG_ENTRIES", 1)
    assert (
        len(
            metadata.build_metadata_evidence_catalog(
                [source.model_copy(update={"text": "a"})]
            )
        )
        == 1
    )
