"""Immutable canonical revisions behind stable editable chunk identities."""

from dataclasses import dataclass
from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from onyx.db.models import RegulatoryCanonicalRevision, RegulatoryTemporalProjection
from onyx.document_index.publication_models import publication_digest
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.amendments.annexes.models import AnnexCanonicalSnapshot


@dataclass(frozen=True)
class CanonicalRevision:
    id: UUID
    snapshot: AnnexCanonicalSnapshot
    sha256: str


def retain_canonical_revision(
    session: Session, snapshot: AnnexCanonicalSnapshot
) -> UUID:
    payload = snapshot.model_dump(mode="json")
    digest = publication_digest(payload)
    identifier = uuid5(NAMESPACE_URL, "canonical-revision:" + digest)
    session.execute(
        insert(RegulatoryCanonicalRevision)
        .values(
            id=identifier,
            user_file_id=UUID(snapshot.user_file_id),
            canonical_chunk_id=snapshot.id,
            payload_sha256=digest,
            payload=payload,
        )
        .on_conflict_do_nothing()
    )
    retained = get_canonical_revision(session, identifier)
    if retained.snapshot != snapshot:
        raise ValueError("canonical revision digest collision")
    # Import pre-revision bindings only while their original text authority can
    # still be proved. Approval payloads and hashes are never rewritten.
    for binding in session.scalars(
        select(RegulatoryTemporalProjection)
        .where(
            RegulatoryTemporalProjection.canonical_chunk_id == snapshot.id,
            RegulatoryTemporalProjection.canonical_revision_id.is_(None),
        )
        .with_for_update()
    ):
        if binding.payload.get("canonical_base_sha256") != context_hash(snapshot.text):
            raise ValueError("unretained temporal canonical authority has changed")
        binding.canonical_revision_id = identifier
    session.flush()
    return identifier


def get_canonical_revision(session: Session, identifier: UUID) -> CanonicalRevision:
    row = session.get(RegulatoryCanonicalRevision, identifier)
    if row is None:
        raise ValueError("canonical revision does not exist")
    if publication_digest(row.payload) != row.payload_sha256:
        raise ValueError("canonical revision payload changed")
    snapshot = AnnexCanonicalSnapshot.model_validate(row.payload)
    if (
        snapshot.id != row.canonical_chunk_id
        or UUID(snapshot.user_file_id) != row.user_file_id
    ):
        raise ValueError("canonical revision scope mismatch")
    return CanonicalRevision(id=row.id, snapshot=snapshot, sha256=row.payload_sha256)


def validate_temporal_canonical_revision(
    session: Session, row: RegulatoryTemporalProjection
) -> None:
    if row.canonical_revision_id is None:
        # Existing bindings are imported under ownership before their first edit.
        return
    revision = get_canonical_revision(session, row.canonical_revision_id)
    if (
        revision.snapshot.id != row.canonical_chunk_id
        or UUID(revision.snapshot.user_file_id) != row.user_file_id
        or context_hash(revision.snapshot.text)
        != row.payload.get("canonical_base_sha256")
    ):
        raise ValueError("temporal canonical revision authority mismatch")


def list_canonical_revisions(
    session: Session, canonical_chunk_id: str
) -> list[CanonicalRevision]:
    return [
        get_canonical_revision(session, identifier)
        for identifier in session.scalars(
            select(RegulatoryCanonicalRevision.id)
            .where(RegulatoryCanonicalRevision.canonical_chunk_id == canonical_chunk_id)
            .order_by(
                RegulatoryCanonicalRevision.created_at, RegulatoryCanonicalRevision.id
            )
        )
    ]
