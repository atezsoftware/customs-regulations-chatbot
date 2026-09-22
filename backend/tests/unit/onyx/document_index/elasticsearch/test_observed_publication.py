"""Legacy preservation never grants encoder proof or changes the vector space."""

import json

import pytest

from onyx.document_index.publication_models import (
    ObservedPublicationProjection,
    PublicationIndexSnapshot,
)
from tests.unit.onyx.document_index.elasticsearch.test_publication_source import (
    serialized_source,
)


def observed() -> ObservedPublicationProjection:
    return ObservedPublicationProjection.observe(
        context_projection_id="binding-1",
        source_json=json.dumps(serialized_source()),
        observed_index=PublicationIndexSnapshot(
            index_name="physical-index",
            index_uuid="physical-uuid",
            search_settings_id=11,
            model_provider="",
            model_name="legacy",
            vector_dimension=2,
            embedding_config_sha256="0" * 64,
            multitenant=False,
        ),
    )


def test_observation_roundtrip_has_no_invented_encoder_inputs() -> None:
    projection = observed()
    assert (
        ObservedPublicationProjection.model_validate_json(projection.model_dump_json())
        == projection
    )
    assert "embedding_inputs" not in projection.model_dump()
    assert "embedding_config_json" not in projection.model_dump()


@pytest.mark.parametrize(
    "field,value",
    [
        ("content", "different law"),
        ("content_vector", [0.8, 0.9]),
        ("title_vector", [0.8, 0.9]),
        ("chunk_context", "new context"),
        ("regulatory_chunk_id", "sibling"),
        ("heading_path", ["another article"]),
        ("validity_start_date", 0),
        ("validity_end_date", None),
    ],
)
def test_observation_rejects_content_vector_and_interval_expansion(
    field: str, value: object
) -> None:
    payload = observed().model_dump()
    source = json.loads(payload["source_json"])
    source[field] = value
    payload["source_json"] = json.dumps(source)
    with pytest.raises(ValueError):
        ObservedPublicationProjection.model_validate(payload)


def test_observation_allows_metadata_and_same_index_interval_copy() -> None:
    payload = observed().model_dump()
    source = json.loads(payload["source_json"])
    source.update(chunk_index=6, hidden=True, validity_end_date=1780000000)
    payload.update(ordinal=6, source_json=json.dumps(source))
    result = ObservedPublicationProjection.model_validate(payload)
    assert json.loads(result.source_json)["content_vector"] == [0.3, 0.4]
    assert result.accepted_by(result.observed_index)
    assert not result.accepted_by(
        result.observed_index.model_copy(update={"index_uuid": "replacement"})
    )


def test_index_observation_roundtrip_keeps_vectors_and_never_becomes_encoder_proof() -> (
    None
):
    from unittest.mock import MagicMock
    from uuid import uuid4

    from onyx.document_index.elasticsearch.publication import FencedPublicationIndex
    from onyx.document_index.publication_models import FileReservations
    from tests.unit.onyx.regulatory.indexing_jobs.owned_publication_test_helpers import (
        OwnedAuthority,
    )

    authority = OwnedAuthority(uuid4(), tenant_id="public")
    original = observed()
    source = json.loads(original.source_json)
    source["document_id"] = str(authority.owner.user_file_id)
    projection = ObservedPublicationProjection.observe(
        context_projection_id=original.context_projection_id,
        observed_index=original.observed_index,
        source_json=json.dumps(source),
    )
    adapter = FencedPublicationIndex(MagicMock(), projection.observed_index)
    reservations = FileReservations(
        ownership=authority.owner, ordinals=(5,), gate_closed=True
    )
    stored = adapter._source(reservations, projection, 5)
    assert "source_json" not in stored["publication_evidence"]["observation"]
    evidence = adapter._evidence_from_source(reservations, 5, stored)
    assert evidence.frozen_projection is None
    assert evidence.observed_projection == projection
    assert stored["content_vector"] == source["content_vector"]
    assert stored["title_vector"] == source["title_vector"]
    guard = {}
    adapter._guard_observation(guard, stored)
    assert guard["observation"]["content_vector"] == source["content_vector"]
    assert "access_control_list" not in guard["observation"]
    from onyx.document_index.interfaces_new import DocumentChunkVerificationError

    stored["content_vector"] = [0.8, 0.9]
    with pytest.raises(DocumentChunkVerificationError):
        adapter._evidence_from_source(reservations, 5, stored)


def test_activated_observation_cannot_silently_lose_its_index_proof() -> None:
    from onyx.document_index.publication_models import (
        IndexedProjectionEvidence,
        matches_indexed_evidence,
    )

    projection = observed()
    raw = IndexedProjectionEvidence(
        index=projection.observed_index,
        source_json=projection.source_json,
        frozen_projection=None,
        payload_sha256=None,
    )
    assert not matches_indexed_evidence(projection, raw)
    assert matches_indexed_evidence(
        projection, raw.model_copy(update={"observed_projection": projection})
    )


@pytest.mark.parametrize(
    "changed_proof", [None, "context_projection_id", "observed_source_sha256"]
)
def test_observation_accepts_reordered_json_only_with_identical_proof(
    changed_proof: str | None,
) -> None:
    from onyx.document_index.publication_models import (
        IndexedProjectionEvidence,
        matches_indexed_evidence,
    )

    projection = observed()
    source_json = json.dumps(json.loads(projection.source_json), sort_keys=True)
    assert source_json != projection.source_json
    payload = projection.model_dump()
    payload["source_json"] = source_json
    if changed_proof is not None:
        payload[changed_proof] = "f" * 64
    decoded = ObservedPublicationProjection.model_validate(payload)
    evidence = IndexedProjectionEvidence(
        index=projection.observed_index,
        source_json=source_json,
        frozen_projection=None,
        payload_sha256=None,
        observed_projection=decoded,
    )
    assert matches_indexed_evidence(projection, evidence) is (changed_proof is None)


@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_index_authority_is_independent_of_binding_order(reverse: bool) -> None:
    from onyx.document_index.publication_models import (
        PublicationEncoderAuthority,
        PublicationEncoderReceipt,
        merge_publication_indexes,
        publication_digest,
    )

    raw = observed().observed_index
    authority = PublicationEncoderAuthority(
        provider=None,
        model=raw.model_name,
        effective_dimension=2,
        endpoint_sha256="0" * 64,
        deployment_name=None,
        api_version=None,
        normalize=True,
        passage_prefix=None,
    )
    config = authority.model_dump()
    config["dimension"] = config.pop("effective_dimension")
    receipt = PublicationEncoderReceipt(
        configuration_json=json.dumps(config),
        authority=authority,
        resolution_sha256="1" * 64,
    )
    qualified = PublicationIndexSnapshot.model_validate(
        {
            **raw.model_dump(),
            "encoder_authority": authority,
            "encoder_receipts": (receipt,),
            "embedding_config_sha256": publication_digest(config),
        }
    )
    indexes = [raw, qualified] if reverse else [qualified, raw]
    result = merge_publication_indexes(indexes)
    assert result == qualified
    assert result.encoder_authority is not None
    assert observed().accepted_by(result)
