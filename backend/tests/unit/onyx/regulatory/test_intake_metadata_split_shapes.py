"""Target splitting classifies shape errors and preserves exact parent evidence."""

import json

import pytest

from onyx.regulatory.document_intake import metadata
from tests.unit.onyx.regulatory.test_intake_metadata_value_recovery import _fact

_HASH = "a" * 64


def _parent() -> tuple[list[metadata.MetadataChunk], list[str], str]:
    sources = [
        metadata.MetadataChunk(
            chunk_id=f"target-{index}",
            text=f"Sayı : 31529\nMADDE {index} – Kural.",
            heading_path=[f"MADDE {index}"],
            chunk_type="article",
        )
        for index in range(8)
    ]
    catalog = metadata.build_metadata_evidence_catalog(sources)
    response = json.dumps(
        {
            "chunks": [
                {
                    "chunk_id": source.chunk_id,
                    "fields": [
                        _fact(
                            "publication_number",
                            "31529",
                            next(
                                entry.evidence_id
                                for entry in catalog
                                if entry.source_chunk_id == source.chunk_id
                                and entry.text == "Sayı : 31529"
                                and entry.scope == "body"
                            ),
                        ),
                        _fact("article_no", "1", "f" * 64),
                    ],
                }
                for source in sources
            ],
            "warnings": ["Publication date remains uncertain"],
        }
    )
    return sources, [source.chunk_id for source in sources], response


@pytest.mark.parametrize(
    "case",
    ["empty", "eof", "missing_target", "duplicate_target", "duplicate_field", "schema"],
)
def test_only_shape_failures_are_classified_for_bounded_splitting(case: str) -> None:
    _sources, targets, response = _parent()
    raw = json.loads(response)
    if case == "empty":
        raw["chunks"] = []
    elif case == "missing_target":
        raw["chunks"].pop()
    elif case == "duplicate_target":
        raw["chunks"][-1] = raw["chunks"][0]
    elif case == "duplicate_field":
        raw["chunks"][0]["fields"].append(raw["chunks"][0]["fields"][0])
    elif case == "schema":
        raw["chunks"][0]["fields"][0]["value"] = 3.14
    assert metadata.metadata_response_needs_split(
        '{"chunks":[' if case == "eof" else json.dumps(raw), targets
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("article_no", "1"),
        ("document_title", "Invented title"),
        ("validity_start_date", "not-a-date"),
    ],
)
def test_typed_semantic_errors_never_authorize_shape_splitting(
    field: str, value: str
) -> None:
    _sources, targets, response = _parent()
    raw = json.loads(response)
    for target in raw["chunks"]:
        target["fields"] = [_fact(field, value, "f" * 64)]
    assert not metadata.metadata_response_needs_split(json.dumps(raw), targets)
    assert not metadata.metadata_response_needs_split(" " * (1024 * 1024 + 1), targets)


def test_eight_to_four_projection_preserves_hash_facts_warnings_and_strict_validation() -> (
    None
):
    sources, targets, response = _parent()
    projected = metadata.project_metadata_attempts(
        [(_HASH, response)], parent_target_ids=targets, target_ids=targets[:4]
    )
    assert len(projected) == 1 and projected[0][0] == _HASH
    child = json.loads(projected[0][1])
    parent = json.loads(response)
    assert child == {**parent, "chunks": parent["chunks"][:4]}
    with pytest.raises(ValueError, match="not in this document catalog"):
        metadata.validate_metadata_transport_response(
            projected[0][1], chunks=sources, target_ids=targets[:4]
        )
    current = metadata.LegalMetadataResult(
        chunks=[metadata.ChunkLegalMetadata(chunk_id=target) for target in targets[:4]]
    )
    retained = metadata.retain_validated_metadata_facts(
        current, projected[0][1], _HASH, sources, targets[:4]
    )
    assert all(
        [fact.field for fact in target.fields] == ["publication_number"]
        for target in retained.chunks
    )
    assert all(
        target.fields[0].retained_request_hash == _HASH for target in retained.chunks
    )
    assert retained.warnings == [
        f"Retained metadata from request {_HASH}: Publication date remains uncertain"
    ]
    targets = targets[:4]
    for width in (2, 1):
        projected = metadata.project_metadata_attempts(
            projected, parent_target_ids=targets, target_ids=targets[:width]
        )
        targets = targets[:width]
        assert projected[0][0] == _HASH
        assert json.loads(projected[0][1])["chunks"] == parent["chunks"][:width]


@pytest.mark.parametrize(
    "case",
    [
        "malformed",
        "missing_target",
        "duplicate_target",
        "duplicate_field",
        "schema",
        "oversize",
    ],
)
def test_invalid_parent_shapes_cannot_be_projected(case: str) -> None:
    _sources, targets, response = _parent()
    raw = json.loads(response)
    if case == "missing_target":
        raw["chunks"].pop()
    elif case == "duplicate_target":
        raw["chunks"][-1] = raw["chunks"][0]
    elif case == "duplicate_field":
        raw["chunks"][0]["fields"] *= 2
    elif case == "schema":
        raw["extra"] = True
    response = (
        "{broken"
        if case == "malformed"
        else " " * (1024 * 1024 + 1)
        if case == "oversize"
        else json.dumps(raw)
    )
    assert (
        metadata.project_metadata_attempts(
            [(_HASH, response)], parent_target_ids=targets, target_ids=targets[:4]
        )
        == []
    )


@pytest.mark.parametrize(
    "child",
    [["foreign"], [], ["target-0"] * 2, [f"target-{index}" for index in range(8)]],
)
def test_child_scope_must_be_a_distinct_strict_parent_subset(child: list[str]) -> None:
    _sources, targets, response = _parent()
    with pytest.raises(ValueError, match="strict parent subset"):
        metadata.project_metadata_attempts(
            [(_HASH, response)], parent_target_ids=targets, target_ids=child
        )


def test_projection_rejects_forged_request_provenance() -> None:
    _sources, targets, response = _parent()
    with pytest.raises(ValueError, match="original request hash"):
        metadata.project_metadata_attempts(
            [("forged", response)], parent_target_ids=targets, target_ids=targets[:4]
        )
