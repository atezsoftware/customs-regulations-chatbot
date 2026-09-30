"""Inactive maintenance fields preserve evidence, never grant repair authority."""

import json

import pytest

from onyx.db.regulatory_writer_publication import _retained_binding_payload_matches
from onyx.document_index.publication_models import ObservedPublicationProjection
from onyx.regulatory.amendments.annexes.models import AnnexTemporalProjection
from onyx.regulatory.publication_baseline import observed_baseline_binding
from tests.unit.onyx.document_index.elasticsearch.test_observed_publication import (
    observed,
)
from tests.unit.onyx.regulatory.test_publication_baseline import baseline_case


@pytest.mark.parametrize(
    "fields",
    [
        {"canonical_restore_fields": []},
        {"canonical_restore_fields": [], "allow_frozen_predecessor": False},
        {
            "heading_repair": None,
            "canonical_restore_fields": None,
            "allow_frozen_predecessor": None,
        },
    ],
)
def test_inactive_receipt_survives_read_and_retained_publication(
    fields: dict[str, object],
) -> None:
    inputs, evidence = baseline_case()
    original = observed_baseline_binding(evidence, inputs.canonical)
    payload = original.model_dump(mode="json")
    payload["projection"].update(fields)
    binding = AnnexTemporalProjection.model_validate(payload)
    assert binding.projection.source_json == original.projection.source_json
    assert _retained_binding_payload_matches(payload, binding)
    assert _retained_binding_payload_matches(
        payload, AnnexTemporalProjection.model_validate_json(binding.model_dump_json())
    )
    assert all(payload["projection"][key] == value for key, value in fields.items())
    assert not _retained_binding_payload_matches(
        payload, binding.model_copy(update={"representation_text": "different law"})
    )


@pytest.mark.parametrize(
    "fields",
    [
        {"canonical_restore_fields": ["content"]},
        {"canonical_restore_fields": ["heading_path"]},
        {"canonical_restore_fields": ["content_vector"]},
        {"unknown_authority": False},
    ],
)
def test_active_or_unknown_maintenance_authority_is_rejected(
    fields: dict[str, object],
) -> None:
    payload = observed().model_dump(mode="json")
    payload.update(fields)
    with pytest.raises(ValueError):
        ObservedPublicationProjection.model_validate(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("content", "another law"),
        ("content_vector", [0.9, 0.8]),
        ("heading_path", ["wrong article"]),
        ("regulatory_chunk_id", "other-chunk"),
    ],
)
def test_empty_restore_fields_do_not_relax_immutable_evidence(
    field: str, value: object
) -> None:
    payload = observed().model_dump(mode="json")
    payload["canonical_restore_fields"] = []
    source = json.loads(payload["source_json"])
    source[field] = value
    payload["source_json"] = json.dumps(source)
    with pytest.raises(ValueError):
        ObservedPublicationProjection.model_validate(payload)


@pytest.mark.parametrize(
    "fields", [["heading_path"], ["blurb", "content", "heading_path"]]
)
def test_completed_repair_is_readable_but_future_updates_keep_content_frozen(
    fields: list[str],
) -> None:
    from onyx.document_index.elasticsearch.publication import FencedPublicationIndex
    from onyx.document_index.publication_models import (
        OBSERVED_MUTABLE_SOURCE_FIELDS,
        publication_digest,
    )

    payload = observed().model_dump(mode="json")
    source = json.loads(payload["source_json"])
    payload.update(
        canonical_restore_fields=fields,
        allow_frozen_predecessor=True,
        observed_immutable_sha256=publication_digest(
            {
                key: value
                for key, value in source.items()
                if key not in OBSERVED_MUTABLE_SOURCE_FIELDS and key not in fields
            }
        ),
    )
    repaired = ObservedPublicationProjection.model_validate(payload)
    strict = repaired.for_ordinary_update()
    assert strict.source_json == repaired.source_json
    assert strict.canonical_restore_fields is None
    assert strict.allow_frozen_predecessor is None
    params = {}
    FencedPublicationIndex._guard_observation(
        params,
        {
            **source,
            "publication_evidence": {
                "kind": "observed-v1",
                "observation": repaired.model_dump(
                    mode="json", exclude={"source_json"}
                ),
            },
        },
    )
    assert "content" not in params["observation_mutable"]
    assert "heading_path" not in params["observation_mutable"]
    assert params["observation_heading_repair"] is None
    # Historical interval copies use strict content proof, not the repair's authority.
    closed = {**source, "validity_end_date": 1780000000}
    result = ObservedPublicationProjection.model_validate(
        {
            **strict.model_dump(),
            "source_json": json.dumps(closed),
        }
    )
    assert json.loads(result.source_json)["content_vector"] == source["content_vector"]
    for base in (repaired, strict):
        for field in fields:
            changed = {
                **source,
                field: ["incorrect"] if field == "heading_path" else "incorrect",
            }
            with pytest.raises(ValueError):
                ObservedPublicationProjection.model_validate(
                    {
                        **base.model_dump(),
                        "source_json": json.dumps(changed),
                    }
                )


def test_earlier_repair_receipt_anchors_old_heading_without_admitting_content_drift() -> (
    None
):
    from onyx.document_index.publication_models import (
        OBSERVED_MUTABLE_SOURCE_FIELDS,
        publication_digest,
    )

    payload = observed().model_dump(mode="json")
    source = json.loads(payload["source_json"])
    old_heading = source.get("heading_path")
    source["heading_path"] = ["Correct article"]
    payload.update(
        source_json=json.dumps(source),
        canonical_restore_fields=["heading_path"],
        observed_immutable_sha256=publication_digest(
            {
                k: v
                for k, v in source.items()
                if k not in OBSERVED_MUTABLE_SOURCE_FIELDS and k != "heading_path"
            }
        ),
        heading_repair={
            "source_sha256": "a" * 64,
            "parser_version": "5",
            "source_start": 1,
            "source_end": 10,
            "canonical_chunk_id": source["regulatory_chunk_id"],
            "original_heading_present": old_heading is not None,
            "original_heading_path": old_heading,
            "corrected_heading_path": source["heading_path"],
            "canonical_before_sha256": "b" * 64,
            "canonical_after_sha256": "c" * 64,
        },
    )
    repaired = ObservedPublicationProjection.model_validate(payload)
    assert json.loads(repaired.for_ordinary_update().source_json) == source
    for field, value in (
        ("content", "wrong content"),
        ("heading_path", ["Wrong article"]),
    ):
        with pytest.raises(ValueError):
            ObservedPublicationProjection.model_validate(
                {
                    **payload,
                    "source_json": json.dumps({**source, field: value}),
                }
            )
