"""Keep proven intake headings as frozen context and derived label sources."""

from collections.abc import Mapping, Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, load_only

from onyx.db.document_intake_structural_readers import (
    validate_current_structural_reader_proof,
)
from onyx.db.models import RegulatoryDerivedLabelProjection, RegulatoryLabelingItem
from onyx.db.regulatory_annex_changes import capture_canonical_scope
from onyx.regulatory.amendments.annexes.models import AnnexCanonicalSnapshot
from onyx.regulatory.document_intake.structural_readers import NativeChapterReaderProof
from onyx.regulatory.labeling.intake_abstention_repair import ABSTENTION_REPAIR_KEY

STRUCTURAL_READER_KEY = "intake_structural_reader"
FrozenLabelingSource = RegulatoryLabelingItem | RegulatoryDerivedLabelProjection
_READER_LIMIT = 1024


def validate_requested_readers(
    session: Session,
    readers: Mapping[str, NativeChapterReaderProof],
    *,
    file_ids: Sequence[UUID],
    atomic_ids: Sequence[str],
) -> dict[str, NativeChapterReaderProof]:
    """Validate all run-local roles before changing classification target scope."""
    if len(readers) > _READER_LIMIT or not set(readers).issubset(atomic_ids):
        raise ValueError("Structural readers must be bounded canonical intake sources")
    scopes: dict[UUID, list[AnnexCanonicalSnapshot]] = {}
    validated: dict[str, NativeChapterReaderProof] = {}
    for identifier, proof in readers.items():
        file_id = UUID(str(proof.user_file_id))
        if proof.heading_chunk_id != identifier or file_id not in file_ids:
            raise ValueError("Structural reader scope differs from its frozen source")
        if file_id not in scopes:
            scopes[file_id] = capture_canonical_scope(session, file_id)
        validated[identifier] = validate_current_structural_reader_proof(
            session, proof, scopes[file_id]
        )
        if (
            not proof.governed_chunk_ids
            or set(proof.governed_chunk_ids) & set(readers)
            or not set(proof.governed_chunk_ids).issubset(atomic_ids)
        ):
            raise ValueError(
                "Structural reader lineage must use ordinary canonical leaves"
            )
    return validated


def validated_reader_proof(
    session: Session,
    target: RegulatoryDerivedLabelProjection,
    *,
    scopes: dict[UUID, list[AnnexCanonicalSnapshot]] | None = None,
) -> NativeChapterReaderProof:
    proof = NativeChapterReaderProof.model_validate(
        target.source_snapshot.get(STRUCTURAL_READER_KEY)
    )
    file_id = UUID(str(proof.user_file_id))
    if (
        proof.heading_chunk_id != target.regulatory_chunk_id
        or file_id != target.user_file_id
        or proof.heading_text != target.text_snapshot
    ):
        raise ValueError("The structural reader's frozen target changed")
    scopes = scopes if scopes is not None else {}
    if file_id not in scopes:
        scopes[file_id] = capture_canonical_scope(session, file_id)
    validated = validate_current_structural_reader_proof(
        session, proof, scopes[file_id]
    )
    current = next(row for row in scopes[file_id] if row.id == proof.heading_chunk_id)
    expected = {
        "position": current.position,
        "heading_path": current.heading_path,
        "validity_start_date": current.validity_start_date.isoformat()
        if current.validity_start_date
        else None,
        "validity_end_date": current.validity_end_date.isoformat()
        if current.validity_end_date
        else None,
        "chunk_type": current.chunk_type,
        "chunk_metadata": current.metadata,
    }
    if any(target.source_snapshot.get(key) != value for key, value in expected.items()):
        raise ValueError("The structural reader's frozen canonical snapshot changed")
    return validated


def reader_context_sources(
    session: Session,
    run_id: UUID,
    *,
    identifiers: Sequence[str] | None = None,
    file_id: UUID | None = None,
    strict: bool = True,
) -> list[RegulatoryDerivedLabelProjection]:
    """Read only proof-backed reader snapshots, never ordinary derived aggregates."""
    query = (
        select(RegulatoryDerivedLabelProjection)
        .options(
            load_only(
                RegulatoryDerivedLabelProjection.regulatory_chunk_id,
                RegulatoryDerivedLabelProjection.user_file_id,
                RegulatoryDerivedLabelProjection.text_snapshot,
                RegulatoryDerivedLabelProjection.source_snapshot,
            )
        )
        .where(
            RegulatoryDerivedLabelProjection.run_id == run_id,
            RegulatoryDerivedLabelProjection.source_snapshot.op("?")(
                STRUCTURAL_READER_KEY
            ),
        )
        .order_by(RegulatoryDerivedLabelProjection.regulatory_chunk_id)
        .limit(_READER_LIMIT + 1)
    )
    if identifiers is not None:
        query = query.where(
            RegulatoryDerivedLabelProjection.regulatory_chunk_id.in_(identifiers)
        )
    if file_id is not None:
        query = query.where(RegulatoryDerivedLabelProjection.user_file_id == file_id)
    readers = list(session.scalars(query))
    if len(readers) > _READER_LIMIT:
        raise ValueError("The frozen structural reader scope is too broad")
    scopes: dict[UUID, list[AnnexCanonicalSnapshot]] = {}
    valid = []
    for reader in readers:
        try:
            validated_reader_proof(session, reader, scopes=scopes)
        except ValueError:
            if strict:
                raise
        else:
            valid.append(reader)
    return valid


def complete_frozen_sources(
    session: Session,
    run_id: UUID,
    items: Sequence[RegulatoryLabelingItem],
    identifiers: Sequence[str],
) -> dict[str, FrozenLabelingSource]:
    """Preserve source-only headings in unchanged canonical context receipts."""
    sources: dict[str, FrozenLabelingSource] = {}
    for item in items:
        if ABSTENTION_REPAIR_KEY in item.source_snapshot:
            from onyx.db.models import RegulatoryLabelingRun
            from onyx.db.regulatory_labeling_semantic_repair import (
                validate_item_semantic_repair_audit,
            )

            try:
                validate_item_semantic_repair_audit(
                    session,
                    item=item,
                    run=session.get_one(RegulatoryLabelingRun, run_id),
                )
            except ValueError:
                continue
        sources[item.regulatory_chunk_id] = item
    missing = [identifier for identifier in identifiers if identifier not in sources]
    if missing:
        for reader in reader_context_sources(
            session, run_id, identifiers=missing, strict=False
        ):
            sources[reader.regulatory_chunk_id] = reader
    return sources
