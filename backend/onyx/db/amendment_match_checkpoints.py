"""Lease-guarded matching checkpoints; source content is never modified here."""

from datetime import datetime, timezone
from hashlib import sha256
from uuid import UUID

from sqlalchemy import Text, func, select
from sqlalchemy.orm import Session

from onyx.db.enums import AmendmentBatchStatus
from onyx.db.models import (
    AmendmentBatch,
    AmendmentMatchCheckpoint,
    RegulatoryChunk,
    RegulatoryFilePublication,
    UserFile,
)
from onyx.regulatory.amendments.match_checkpoint import (
    MatchedInstruction,
    MatchEvidence,
)
from onyx.regulatory.amendments.ranker import CandidateChunk


def _source_fingerprints(db_session: Session, chunk_ids: list[str]) -> dict[str, str]:
    fingerprint = func.md5(
        func.jsonb_build_array(
            RegulatoryChunk.user_file_id,
            RegulatoryChunk.text,
            RegulatoryChunk.position,
            RegulatoryChunk.chunk_type,
            RegulatoryChunk.chunk_metadata,
            RegulatoryChunk.heading_path,
            RegulatoryChunk.status,
            RegulatoryChunk.validity_start_date,
            RegulatoryChunk.validity_end_date,
            RegulatoryFilePublication.scope_key,
            RegulatoryFilePublication.epoch,
            RegulatoryFilePublication.fencing_token,
            RegulatoryFilePublication.gate_closed,
            UserFile.name,
        ).cast(Text)
    )
    return dict(
        db_session.execute(
            select(RegulatoryChunk.id, fingerprint)
            .join(UserFile, UserFile.id == RegulatoryChunk.user_file_id)
            .outerjoin(
                RegulatoryFilePublication,
                RegulatoryFilePublication.user_file_id == RegulatoryChunk.user_file_id,
            )
            .where(RegulatoryChunk.id.in_(chunk_ids))
        )
        .tuples()
        .all()
    )


def match_scope_fingerprint(db_session: Session, file_ids: list[str]) -> str:
    # Writer acquisition does not change visible sources. Epoch/gate changes do;
    # candidate-specific fingerprints separately retain their writer fencing.
    roots = (
        select(RegulatoryChunk.user_file_id, RegulatoryChunk.heading_path)
        .where(
            RegulatoryChunk.user_file_id.in_([UUID(value) for value in file_ids]),
            RegulatoryChunk.status == "active",
        )
        .distinct(RegulatoryChunk.user_file_id)
        .order_by(
            RegulatoryChunk.user_file_id, RegulatoryChunk.position, RegulatoryChunk.id
        )
        .subquery()
    )
    fingerprint = func.md5(
        func.jsonb_build_array(
            UserFile.id,
            UserFile.name,
            roots.c.heading_path,
            RegulatoryFilePublication.scope_key,
            RegulatoryFilePublication.epoch,
            RegulatoryFilePublication.gate_closed,
        ).cast(Text)
    )
    rows = db_session.execute(
        select(UserFile.id, fingerprint)
        .outerjoin(roots, roots.c.user_file_id == UserFile.id)
        .outerjoin(
            RegulatoryFilePublication,
            RegulatoryFilePublication.user_file_id == UserFile.id,
        )
        .where(UserFile.id.in_([UUID(value) for value in file_ids]))
        .order_by(UserFile.id)
    )
    digest = sha256()
    for file_id, value in rows:
        digest.update(f"{file_id}:{value};".encode())
    digest.update(str(sorted(file_ids)).encode())
    return digest.hexdigest()


def capture_match_evidence(
    db_session: Session, file_ids: list[str], candidates: list[CandidateChunk]
) -> MatchEvidence:
    # Keep the canonical rows stable while validating and fingerprinting them.
    by_id = {candidate.chunk_id: candidate for candidate in candidates}
    rows = db_session.execute(
        select(
            RegulatoryChunk.id,
            RegulatoryChunk.user_file_id,
            RegulatoryChunk.text,
            RegulatoryChunk.chunk_metadata,
            RegulatoryChunk.heading_path,
            RegulatoryChunk.status,
        )
        .where(RegulatoryChunk.id.in_(by_id))
        .with_for_update(read=True)
    ).all()
    if len(rows) != len(by_id) or any(
        text != by_id[chunk_id].text
        or str(file_id) != by_id[chunk_id].user_file_id
        or str(file_id) not in file_ids
        or status != "active"
        or {k: v for k, v in by_id[chunk_id].metadata.items() if k != "heading_path"}
        != {k: v for k, v in metadata.items() if k != "heading_path"}
        or by_id[chunk_id].metadata.get("heading_path", []) != heading
        for chunk_id, file_id, text, metadata, heading, status in rows
    ):
        raise ValueError("Matching checkpoint evidence changed or is outside its scope")
    return MatchEvidence(
        scope_sha256=match_scope_fingerprint(db_session, file_ids),
        candidates=_source_fingerprints(db_session, list(by_id)),
    )


def assert_match_evidence(
    db_session: Session, file_ids: list[str], evidence: MatchEvidence
) -> None:
    db_session.execute(
        select(RegulatoryChunk.id)
        .where(RegulatoryChunk.id.in_(evidence.candidates))
        .with_for_update(read=True)
    ).all()
    if (
        match_scope_fingerprint(db_session, file_ids) != evidence.scope_sha256
        or _source_fingerprints(db_session, list(evidence.candidates))
        != evidence.candidates
    ):
        raise ValueError("Matching checkpoint evidence changed during confirmation")


def load_match_checkpoint(
    db_session: Session,
    *,
    batch_id: int,
    instruction_index: int,
    input_sha256: str,
) -> MatchedInstruction | None:
    row = db_session.get(AmendmentMatchCheckpoint, (batch_id, instruction_index))
    if row is None or row.input_sha256 != input_sha256:
        return None
    if (
        _source_fingerprints(db_session, list(row.source_fingerprints))
        != row.source_fingerprints
    ):
        return None
    checkpoint = MatchedInstruction.model_validate(row.payload)
    file_ids = db_session.scalar(
        select(AmendmentBatch.user_file_ids).where(AmendmentBatch.id == batch_id)
    )
    if (
        checkpoint.evidence is None
        or file_ids is None
        or checkpoint.evidence.scope_sha256
        != match_scope_fingerprint(db_session, file_ids)
        or checkpoint.evidence.candidates != row.source_fingerprints
    ):
        return None
    if checkpoint.instruction_index != instruction_index:
        raise ValueError("Matching checkpoint instruction identity is inconsistent")
    return checkpoint


def persist_match_checkpoint(
    db_session: Session,
    *,
    batch_id: int,
    lease_generation: int,
    input_sha256: str,
    checkpoint: MatchedInstruction,
) -> bool:
    batch = db_session.scalar(
        select(AmendmentBatch).where(AmendmentBatch.id == batch_id).with_for_update()
    )
    if (
        batch is None
        or batch.status != AmendmentBatchStatus.ANALYZING.value
        or batch.lease_generation != lease_generation
        or not 0 <= checkpoint.instruction_index < batch.instruction_count
    ):
        db_session.rollback()
        return False
    candidates = {candidate.chunk_id: candidate for candidate in checkpoint.candidates}
    if (
        checkpoint.match.old_chunk_id is not None
        and checkpoint.match.old_chunk_id not in candidates
    ):
        raise ValueError("Matching checkpoint target is outside its candidate set")
    evidence = capture_match_evidence(
        db_session, batch.user_file_ids, checkpoint.candidates
    )
    if checkpoint.evidence is None or evidence != checkpoint.evidence:
        raise ValueError("Matching checkpoint evidence changed during confirmation")
    fingerprints = evidence.candidates
    row = db_session.get(
        AmendmentMatchCheckpoint, (batch_id, checkpoint.instruction_index)
    )
    if row is None:
        row = AmendmentMatchCheckpoint(
            batch_id=batch_id, instruction_index=checkpoint.instruction_index
        )
        db_session.add(row)
    row.input_sha256 = input_sha256
    row.lease_generation = lease_generation
    row.payload = checkpoint.model_dump(mode="json")
    row.source_fingerprints = fingerprints
    batch.heartbeat_at = datetime.now(timezone.utc)
    db_session.commit()
    return True


def match_checkpoint_counts(
    db_session: Session, batch_ids: list[int]
) -> dict[int, int]:
    return dict(
        db_session.execute(
            select(AmendmentMatchCheckpoint.batch_id, func.count())
            .where(AmendmentMatchCheckpoint.batch_id.in_(batch_ids))
            .group_by(AmendmentMatchCheckpoint.batch_id)
        )
        .tuples()
        .all()
    )
