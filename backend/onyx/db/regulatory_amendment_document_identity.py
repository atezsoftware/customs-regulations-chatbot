"""Read and bind draft metadata to the document's current own opening."""

from collections.abc import Mapping
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.models import RegulatoryChunk
from onyx.regulatory.source_document_identity import (
    DocumentIdentityEvidence,
    derive_document_identity,
)

IDENTITY_KEY = "amendment_document_identity"


def load_document_identity(
    session: Session, user_file_id: UUID
) -> DocumentIdentityEvidence | None:
    row = session.scalar(
        select(RegulatoryChunk)
        .where(
            RegulatoryChunk.user_file_id == user_file_id,
            RegulatoryChunk.status == "active",
            RegulatoryChunk.superseded_by_chunk_id.is_(None),
            RegulatoryChunk.chunk_type.is_distinct_from("hierarchical_aggregate"),
            RegulatoryChunk.chunk_metadata["bound_to_regulatory_chunk_id"].astext.is_(
                None
            ),
        )
        .order_by(RegulatoryChunk.position, RegulatoryChunk.id)
        .limit(1)
    )
    if row is None or row.position != 0 or not row.heading_path:
        return None
    return derive_document_identity(
        user_file_id=str(user_file_id),
        source_chunk_id=row.id,
        text=row.text,
        root_heading=row.heading_path[0],
    )


def validate_document_identity(
    session: Session, user_file_id: UUID, metadata: Mapping[str, object]
) -> None:
    supplied = metadata.get(IDENTITY_KEY)
    if supplied is None:
        return
    evidence = DocumentIdentityEvidence.model_validate(supplied)
    current = load_document_identity(session, user_file_id)
    if current != evidence or any(
        metadata.get(key) != value for key, value in evidence.fields.items()
    ):
        raise ValueError(
            "Document identity changed or differs from its own source opening; reanalyze the proposal"
        )
