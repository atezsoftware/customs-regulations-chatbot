"""Validate restored ES evidence against an already active physical-index binding."""

from copy import deepcopy
from typing import Any

from onyx.document_index.elasticsearch.publication import indexed_evidence_from_source
from onyx.document_index.publication_models import (
    FrozenPublicationProjection,
    matches_indexed_evidence,
    publication_digest,
)
from onyx.regulatory.amendments.annexes.models import AnnexTemporalProjection


def verified_restored_index_source(
    source: dict[str, Any],
    binding: AnnexTemporalProjection,
    *,
    old_index_uuid: str,
) -> dict[str, Any]:
    """No inferred encoder authority: require the exact stored source and receipt."""
    if not isinstance(binding.projection, FrozenPublicationProjection):
        raise ValueError(
            "restored index repair requires an active binding with encoder proof"
        )
    evidence = source.get("publication_evidence")
    if not isinstance(evidence, dict) or not isinstance(evidence.get("index"), dict):
        raise ValueError("restored index evidence is unavailable")
    stored_uuid = evidence["index"].get("index_uuid")
    if stored_uuid not in {old_index_uuid, binding.index.index_uuid}:
        raise ValueError("unexpected restored physical index")
    old = binding.index.model_copy(update={"index_uuid": stored_uuid})
    actual = indexed_evidence_from_source(old, source)
    if not matches_indexed_evidence(binding.projection, actual):
        raise ValueError("restored payload differs from active binding")
    after = deepcopy(source)
    after["publication_evidence"]["index"]["index_uuid"] = binding.index.index_uuid
    payload = {
        key: value
        for key, value in after.items()
        if key not in {"publication_payload", "publication_operation"}
    }
    payload["publication_floor"] = payload["publication_token"]
    after["publication_payload"] = publication_digest(payload)
    if not matches_indexed_evidence(
        binding.projection, indexed_evidence_from_source(binding.index, after)
    ):
        raise ValueError("restored index does not prove active binding")
    return after
