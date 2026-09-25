"""Immutable canonical revisions behind stable editable chunk identities."""

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from onyx.db.models import RegulatoryCanonicalRevision, RegulatoryTemporalProjection
from onyx.document_index.publication_models import publication_digest
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.amendments.annexes.models import AnnexCanonicalSnapshot

_REVISION_BATCH_SIZE = 256


@dataclass(frozen=True)
class CanonicalRevision:
    id: UUID
    snapshot: AnnexCanonicalSnapshot
    sha256: str


def retain_canonical_revision(
    session: Session, snapshot: AnnexCanonicalSnapshot
) -> UUID:
    return retain_canonical_revisions(session, [snapshot])[snapshot.id]


def retain_canonical_revisions(
    session: Session, snapshots: Sequence[AnnexCanonicalSnapshot]
) -> dict[str, UUID]:
    if len({snapshot.id for snapshot in snapshots}) != len(snapshots):
        raise ValueError("duplicate canonical revision input")
    result: dict[str, UUID] = {}
    for offset in range(0, len(snapshots), _REVISION_BATCH_SIZE):
        group = snapshots[offset : offset + _REVISION_BATCH_SIZE]
        values = []
        by_chunk = {snapshot.id: snapshot for snapshot in group}
        for snapshot in group:
            payload = snapshot.model_dump(mode="json")
            digest = publication_digest(payload)
            identifier = uuid5(NAMESPACE_URL, "canonical-revision:" + digest)
            result[snapshot.id] = identifier
            values.append(
                dict(
                    id=identifier,
                    user_file_id=UUID(snapshot.user_file_id),
                    canonical_chunk_id=snapshot.id,
                    payload_sha256=digest,
                    payload=payload,
                )
            )
        session.execute(
            insert(RegulatoryCanonicalRevision).values(values).on_conflict_do_nothing()
        )
        retained = get_canonical_revisions(
            session, [result[snapshot.id] for snapshot in group]
        )
        for snapshot in group:
            if retained[result[snapshot.id]].snapshot != snapshot:
                raise ValueError("canonical revision digest collision")
        # Pre-revision bindings still need their exact original text authority.
        for binding in session.scalars(
            select(RegulatoryTemporalProjection)
            .where(
                RegulatoryTemporalProjection.canonical_chunk_id.in_(by_chunk),
                RegulatoryTemporalProjection.canonical_revision_id.is_(None),
            )
            .order_by(RegulatoryTemporalProjection.id)
            .with_for_update()
        ):
            snapshot = by_chunk[binding.canonical_chunk_id]
            if binding.payload.get("canonical_base_sha256") != context_hash(
                snapshot.text
            ):
                raise ValueError("unretained temporal canonical authority has changed")
            binding.canonical_revision_id = result[snapshot.id]
        session.flush()
    return result


def get_canonical_revision(session: Session, identifier: UUID) -> CanonicalRevision:
    row = session.get(RegulatoryCanonicalRevision, identifier)
    if row is None:
        raise ValueError("canonical revision does not exist")
    return _validated_revision(row)


def _validated_revision(row: RegulatoryCanonicalRevision) -> CanonicalRevision:
    if publication_digest(row.payload) != row.payload_sha256:
        raise ValueError("canonical revision payload changed")
    snapshot = AnnexCanonicalSnapshot.model_validate(row.payload)
    if (
        snapshot.id != row.canonical_chunk_id
        or UUID(snapshot.user_file_id) != row.user_file_id
    ):
        raise ValueError("canonical revision scope mismatch")
    return CanonicalRevision(id=row.id, snapshot=snapshot, sha256=row.payload_sha256)


def get_canonical_revisions(
    session: Session, identifiers: Sequence[UUID]
) -> dict[UUID, CanonicalRevision]:
    unique_ids = list(dict.fromkeys(identifiers))
    result: dict[UUID, CanonicalRevision] = {}
    for offset in range(0, len(unique_ids), _REVISION_BATCH_SIZE):
        group = unique_ids[offset : offset + _REVISION_BATCH_SIZE]
        for row in session.scalars(
            select(RegulatoryCanonicalRevision)
            .where(RegulatoryCanonicalRevision.id.in_(group))
            .execution_options(populate_existing=True)
        ):
            result[row.id] = _validated_revision(row)
        if not set(group).issubset(result):
            raise ValueError("canonical revision does not exist")
    return result


def validate_temporal_canonical_revision(
    session: Session, row: RegulatoryTemporalProjection
) -> None:
    if row.canonical_revision_id is None:
        # Existing bindings are imported under ownership before their first edit.
        return
    revision = get_canonical_revision(session, row.canonical_revision_id)
    _validate_temporal_revision(row, revision)


def validate_temporal_canonical_revisions(
    session: Session, rows: Sequence[RegulatoryTemporalProjection]
) -> None:
    revisions = get_canonical_revisions(
        session,
        [row.canonical_revision_id for row in rows if row.canonical_revision_id],
    )
    for row in rows:
        if row.canonical_revision_id is not None:
            _validate_temporal_revision(row, revisions[row.canonical_revision_id])


def validate_joined_temporal_canonical_revision(
    row: RegulatoryTemporalProjection,
    revision: RegulatoryCanonicalRevision | None,
) -> None:
    """Validate an outer-joined revision without performing another database read."""
    if row.canonical_revision_id is None:
        return
    if revision is None or revision.id != row.canonical_revision_id:
        raise ValueError("canonical revision does not exist")
    _validate_temporal_revision(row, _validated_revision(revision))


def _validate_temporal_revision(
    row: RegulatoryTemporalProjection, revision: CanonicalRevision
) -> None:
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
