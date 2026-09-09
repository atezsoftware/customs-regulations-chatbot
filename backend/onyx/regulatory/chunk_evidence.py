"""Typed source references shared by canonical and search projections."""

from typing import Any

from onyx.regulatory.amendments.annexes.models import (
    RegulatoryChunkEvidence as RegulatoryChunkEvidence,
)


def chunk_evidence(metadata: dict[str, Any]) -> RegulatoryChunkEvidence:
    return RegulatoryChunkEvidence.model_validate(
        {
            key: metadata[key]
            for key in RegulatoryChunkEvidence.model_fields
            if key in metadata
        }
    )
