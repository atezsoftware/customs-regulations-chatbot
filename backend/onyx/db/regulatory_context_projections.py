"""Scoped immutable context inputs and independently effective retrieval versions."""

import datetime
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryContextGeneration,
    RegulatoryContextProjection,
    RegulatoryContextProjectionCall,
    RegulatoryContextSnapshot,
)
from onyx.document_index.publication_models import accepts_publication_projection
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.amendments.annexes.models import (
    ContextGenerationCall,
    ContextSourceSnapshot,
    FrozenContextProjection,
    PreparedContextView,
)

if TYPE_CHECKING:
    from onyx.document_index.publication_models import PublicationIndexSnapshot
    from onyx.regulatory.amendments.annexes.models import AnnexTemporalProjection


def persist_context_view(
    session: Session, *, user_file_id: UUID, view: PreparedContextView
) -> list[str]:
    """Persist preparation without making it effective or changing legal content."""
    if view.issues:
        raise ValueError("cannot persist incomplete context preparation")
    ids = {projection.canonical_chunk_id for projection in view.projections}
    ids.update(
        item.canonical_chunk_id
        for snapshot in view.snapshots
        for item in snapshot.ordered_ranges
    )
    rows = {
        row.id: row
        for row in session.scalars(
            select(RegulatoryChunk).where(
                RegulatoryChunk.id.in_(ids),
                RegulatoryChunk.user_file_id == user_file_id,
            )
        )
    }
    if set(rows) != ids:
        raise ValueError("context canonical scope mismatch")
    snapshots, calls = persist_context_evidence(
        session, user_file_id=user_file_id, view=view
    )
    result: list[str] = []
    for projection in view.projections:
        row = rows[projection.canonical_chunk_id]
        if projection.canonical_text_sha256 != context_hash(row.text):
            raise ValueError("canonical text changed during context preparation")
        if projection.embedding_input_sha256 != context_hash(
            projection.embedding_texts
        ) or projection.embedding_config_sha256 != context_hash(
            projection.embedding_config
        ):
            raise ValueError("frozen embedding input or configuration hash mismatch")
        if projection.source_snapshot_sha256 not in snapshots or not set(
            projection.request_hashes
        ).issubset(calls):
            raise ValueError("context projection has incomplete source dependencies")
        payload = projection.model_dump(mode="json", exclude={"projection_id"})
        digest = context_hash(payload)
        session.execute(
            insert(RegulatoryContextProjection)
            .values(
                id=uuid4(),
                canonical_chunk_id=row.id,
                source_snapshot_id=snapshots[projection.source_snapshot_sha256],
                payload_sha256=digest,
                embedding_input_sha256=projection.embedding_input_sha256,
                embedding_config_sha256=projection.embedding_config_sha256,
                payload=payload,
            )
            .on_conflict_do_nothing(constraint="uq_context_projection_payload")
        )
        stored_projection = session.scalars(
            select(RegulatoryContextProjection).where(
                RegulatoryContextProjection.canonical_chunk_id == row.id,
                RegulatoryContextProjection.payload_sha256 == digest,
            )
        ).one()
        for request_hash in projection.request_hashes:
            session.execute(
                insert(RegulatoryContextProjectionCall)
                .values(
                    projection_id=stored_projection.id,
                    generation_id=calls[request_hash],
                )
                .on_conflict_do_nothing()
            )
        result.append(str(stored_projection.id))
    session.flush()
    return result


def persist_context_evidence(
    session: Session, *, user_file_id: UUID, view: PreparedContextView
) -> tuple[dict[str, UUID], dict[str, UUID]]:
    """Persist immutable contextual inputs independently of a dated representation."""
    if view.issues:
        raise ValueError("cannot persist incomplete context evidence")
    ids = {
        item.canonical_chunk_id
        for snapshot in view.snapshots
        for item in snapshot.ordered_ranges
    }
    if (
        set(
            session.scalars(
                select(RegulatoryChunk.id).where(
                    RegulatoryChunk.user_file_id == user_file_id,
                    RegulatoryChunk.id.in_(ids),
                )
            )
        )
        != ids
    ):
        raise ValueError("context evidence canonical scope mismatch")
    snapshots: dict[str, UUID] = {}
    for snapshot in view.snapshots:
        payload = snapshot.model_dump(mode="json")
        session.execute(
            insert(RegulatoryContextSnapshot)
            .values(
                id=uuid4(),
                user_file_id=user_file_id,
                sha256=snapshot.sha256,
                payload=payload,
            )
            .on_conflict_do_nothing(constraint="uq_context_snapshot_file_hash")
        )
        stored = session.scalars(
            select(RegulatoryContextSnapshot).where(
                RegulatoryContextSnapshot.user_file_id == user_file_id,
                RegulatoryContextSnapshot.sha256 == snapshot.sha256,
            )
        ).one()
        if stored.payload != payload:
            raise ValueError("context snapshot hash conflicts with frozen payload")
        snapshots[snapshot.sha256] = stored.id
    calls: dict[str, UUID] = {}
    for call in view.calls:
        payload = call.model_dump(mode="json")
        session.execute(
            insert(RegulatoryContextGeneration)
            .values(
                id=uuid4(),
                user_file_id=user_file_id,
                request_sha256=call.request_sha256,
                payload=payload,
            )
            .on_conflict_do_nothing(constraint="uq_context_generation_file_hash")
        )
        stored_call = session.scalars(
            select(RegulatoryContextGeneration).where(
                RegulatoryContextGeneration.user_file_id == user_file_id,
                RegulatoryContextGeneration.request_sha256 == call.request_sha256,
            )
        ).one()
        if stored_call.payload != payload:
            raise ValueError("context request conflicts with frozen output")
        calls[call.request_sha256] = stored_call.id
    return snapshots, calls


def activate_context_projection(
    session: Session,
    projection_id: UUID,
    *,
    effective_start: datetime.date | None,
    effective_end: datetime.date | None,
) -> None:
    """Publication boundary: intersect context lifetime with canonical legal life."""
    projection = session.get(RegulatoryContextProjection, projection_id)
    if projection is None:
        raise ValueError("context projection does not exist")
    row = session.scalars(
        select(RegulatoryChunk)
        .where(RegulatoryChunk.id == projection.canonical_chunk_id)
        .with_for_update()
    ).one()
    starts = [
        value
        for value in (effective_start, row.validity_start_date)
        if value is not None
    ]
    ends = [
        value for value in (effective_end, row.validity_end_date) if value is not None
    ]
    start, end = max(starts) if starts else None, min(ends) if ends else None
    if start is not None and end is not None and start >= end:
        raise ValueError("context projection has no canonical validity intersection")
    if projection.published_at is not None:
        if projection.effective_start != start or projection.effective_end != end:
            raise ValueError("published context projection interval is immutable")
        return
    overlapping = list(
        session.scalars(
            select(RegulatoryContextProjection)
            .where(
                RegulatoryContextProjection.canonical_chunk_id == row.id,
                RegulatoryContextProjection.published_at.is_not(None),
                or_(
                    RegulatoryContextProjection.effective_end.is_(None),
                    RegulatoryContextProjection.effective_end
                    > (start or datetime.date.min),
                ),
                or_(
                    RegulatoryContextProjection.effective_start.is_(None),
                    RegulatoryContextProjection.effective_start
                    < (end or datetime.date.max),
                ),
            )
            .with_for_update()
        )
    )
    if overlapping:
        raise ValueError("context projection overlaps published history")
    projection.effective_start, projection.effective_end = start, end
    projection.published_at = datetime.datetime.now(datetime.timezone.utc)
    session.flush()


def get_effective_context_projection(
    session: Session, canonical_chunk_id: str, *, as_of_date: datetime.date
) -> RegulatoryContextProjection | None:
    return session.scalars(
        select(RegulatoryContextProjection)
        .join(
            RegulatoryChunk,
            RegulatoryChunk.id == RegulatoryContextProjection.canonical_chunk_id,
        )
        .where(
            RegulatoryChunk.id == canonical_chunk_id,
            RegulatoryContextProjection.published_at.is_not(None),
            or_(
                RegulatoryContextProjection.effective_start.is_(None),
                RegulatoryContextProjection.effective_start <= as_of_date,
            ),
            or_(
                RegulatoryContextProjection.effective_end.is_(None),
                RegulatoryContextProjection.effective_end > as_of_date,
            ),
            or_(
                RegulatoryChunk.validity_start_date.is_(None),
                RegulatoryChunk.validity_start_date <= as_of_date,
            ),
            or_(
                RegulatoryChunk.validity_end_date.is_(None),
                RegulatoryChunk.validity_end_date > as_of_date,
            ),
        )
    ).one_or_none()


def load_context_view(
    session: Session, *, user_file_id: UUID, as_of_date: datetime.date
) -> PreparedContextView:
    rows = session.scalars(
        select(RegulatoryChunk).where(RegulatoryChunk.user_file_id == user_file_id)
    ).all()
    projections: list[FrozenContextProjection] = []
    snapshots: dict[UUID, ContextSourceSnapshot] = {}
    calls: dict[UUID, ContextGenerationCall] = {}
    for row in rows:
        projection = get_effective_context_projection(
            session, row.id, as_of_date=as_of_date
        )
        if projection is None:
            continue
        projections.append(
            FrozenContextProjection.model_validate(
                {
                    **projection.payload,
                    "projection_id": str(projection.id),
                    "vector_reuse_verified": True,
                }
            )
        )
        if projection.source_snapshot_id not in snapshots:
            snapshot = session.get(
                RegulatoryContextSnapshot, projection.source_snapshot_id
            )
            if snapshot is None:
                raise ValueError("context snapshot missing")
            snapshots[snapshot.id] = ContextSourceSnapshot.model_validate(
                snapshot.payload
            )
        for call in session.scalars(
            select(RegulatoryContextGeneration)
            .join(
                RegulatoryContextProjectionCall,
                RegulatoryContextProjectionCall.generation_id
                == RegulatoryContextGeneration.id,
            )
            .where(RegulatoryContextProjectionCall.projection_id == projection.id)
        ):
            calls[call.id] = ContextGenerationCall.model_validate(call.payload)
    return PreparedContextView(
        projections=projections,
        snapshots=list(snapshots.values()),
        calls=list(calls.values()),
    )


def load_context_generation_calls(
    session: Session, *, user_file_id: UUID
) -> list[ContextGenerationCall]:
    return [
        ContextGenerationCall.model_validate(row.payload)
        for row in session.scalars(
            select(RegulatoryContextGeneration).where(
                RegulatoryContextGeneration.user_file_id == user_file_id
            )
        )
    ]


def retire_context_projection(
    session: Session, projection_id: UUID, *, effective_end: datetime.date
) -> None:
    """Close an existing context version; its legal identity and payload stay intact."""
    projection = session.scalars(
        select(RegulatoryContextProjection)
        .where(RegulatoryContextProjection.id == projection_id)
        .with_for_update()
    ).one_or_none()
    if projection is None or projection.published_at is None:
        raise ValueError("published context projection does not exist")
    if (
        projection.effective_start is not None
        and effective_end <= projection.effective_start
    ):
        raise ValueError("cannot erase context projection history")
    if (
        projection.effective_end is not None
        and effective_end > projection.effective_end
    ):
        raise ValueError("cannot reopen context projection history")
    projection.effective_end = effective_end
    session.flush()


def activate_temporal_projection(
    session: Session,
    *,
    user_file_id: UUID,
    binding: "AnnexTemporalProjection",
    canonical_revision_id: UUID | None = None,
) -> None:
    """Join an owned publication transaction; never alter another index's history."""
    import json

    from onyx.db.models import RegulatoryTemporalProjection
    from onyx.document_index.publication_models import publication_digest
    from onyx.regulatory.amendments.annexes.context_dependencies import (
        canonical_dependency_ids,
        rebuild_context_aggregates,
    )
    from onyx.regulatory.amendments.annexes.models import AnnexTemporalProjection
    from onyx.utils.text_processing import remove_invalid_unicode_chars

    binding = AnnexTemporalProjection.model_validate(binding.model_dump(mode="json"))
    source = json.loads(binding.projection.source_json)

    def epoch(value: datetime.date | None) -> int | None:
        return (
            int(
                datetime.datetime.combine(
                    value, datetime.time.min, datetime.timezone.utc
                ).timestamp()
            )
            if value is not None
            else None
        )

    if source.get("validity_start_date") != epoch(
        binding.effective_start
    ) or source.get("validity_end_date") != epoch(binding.effective_end):
        raise ValueError("temporal source interval mismatch")
    if (
        source.get("chunk_index") != binding.projection.ordinal
        or len(source["content_vector"]) != binding.index.vector_dimension
        or not accepts_publication_projection(binding.index, binding.projection)
    ):
        raise ValueError("temporal encoder/source identity mismatch")
    canonical = session.get(RegulatoryChunk, source["regulatory_chunk_id"])
    retained_revision_id = canonical_revision_id
    if retained_revision_id is not None:
        from onyx.db.regulatory_canonical_revisions import get_canonical_revision
        from onyx.regulatory.amendments.annexes.staging import canonical_snapshot_rows

        revision = get_canonical_revision(session, retained_revision_id)
        if (
            revision.snapshot.id != source["regulatory_chunk_id"]
            or UUID(revision.snapshot.user_file_id) != user_file_id
        ):
            raise ValueError("temporal retained canonical scope mismatch")
        canonical = canonical_snapshot_rows([revision.snapshot])[0]
    if (
        canonical is None
        or canonical.user_file_id != user_file_id
        or binding.canonical_base_sha256 != context_hash(canonical.text)
    ):
        raise ValueError("temporal canonical base changed")
    if source["document_id"] != str(
        user_file_id
    ) or binding.projection.context_projection_id != str(binding.id):
        raise ValueError("temporal projection identity mismatch")

    def dated_dependency(row: RegulatoryChunk) -> RegulatoryChunk:
        from onyx.regulatory.contextual import context_reference_date

        if (
            row.user_file_id != user_file_id
            or row.validity_start_date is not None
            and (
                binding.effective_start is None
                or row.validity_start_date > binding.effective_start
            )
            or row.validity_end_date is not None
            and (
                binding.effective_end is None
                or row.validity_end_date < binding.effective_end
            )
        ):
            raise ValueError("derived dependency exceeds its legal source window")
        # The lookup anchor selects source authority, not the context generation date.
        qualified = get_indexed_temporal_projection(
            session,
            row.id,
            index=binding.index,
            as_of_date=context_reference_date(
                binding.effective_start, binding.effective_end
            ),
        )
        if qualified is None:
            overlapping = select(RegulatoryTemporalProjection.id).where(
                RegulatoryTemporalProjection.canonical_chunk_id == row.id,
                RegulatoryTemporalProjection.retired_at.is_(None),
                RegulatoryTemporalProjection.index_identity_sha256.in_(
                    binding.index.temporal_lookup_identities()
                ),
            )
            if binding.effective_start is not None:
                overlapping = overlapping.where(
                    or_(
                        RegulatoryTemporalProjection.effective_end.is_(None),
                        RegulatoryTemporalProjection.effective_end
                        > binding.effective_start,
                    )
                )
            if binding.effective_end is not None:
                overlapping = overlapping.where(
                    or_(
                        RegulatoryTemporalProjection.effective_start.is_(None),
                        RegulatoryTemporalProjection.effective_start
                        < binding.effective_end,
                    )
                )
            if session.scalar(overlapping.limit(1)) is not None:
                raise ValueError("qualified dependency window does not cover parent")
            return row
        if (
            qualified.effective_start is not None
            and (
                binding.effective_start is None
                or qualified.effective_start > binding.effective_start
            )
            or qualified.effective_end is not None
            and (
                binding.effective_end is None
                or qualified.effective_end < binding.effective_end
            )
        ):
            raise ValueError("qualified dependency window does not cover parent")
        return RegulatoryChunk(
            id=row.id,
            user_file_id=row.user_file_id,
            text=qualified.representation_text,
            position=qualified.semantic_position,
            projection_ordinal=row.projection_ordinal,
            heading_path=row.heading_path,
            chunk_metadata=qualified.representation_metadata,
            chunk_type=row.chunk_type,
            source=row.source,
            status=row.status,
            validity_start_date=row.validity_start_date,
            validity_end_date=row.validity_end_date,
            supersedes_chunk_id=row.supersedes_chunk_id,
        )

    if binding.derived_role == "canonical":
        if binding.representation_text != canonical.text:
            raise ValueError("direct canonical text mismatch")
    elif binding.derived_role == "hierarchical_aggregate":
        if canonical.chunk_metadata.get("chunk_variant") != "hierarchical_aggregate":
            raise ValueError("canonical row is not a derived aggregate")
        candidate = RegulatoryChunk(
            id=canonical.id,
            user_file_id=user_file_id,
            text=canonical.text,
            position=canonical.position,
            projection_ordinal=canonical.projection_ordinal,
            heading_path=canonical.heading_path,
            chunk_metadata=binding.representation_metadata,
            chunk_type=canonical.chunk_type,
            source=canonical.source,
            status=canonical.status,
            validity_start_date=canonical.validity_start_date,
            validity_end_date=canonical.validity_end_date,
        )
        if canonical_dependency_ids(candidate) != binding.dependency_ids:
            raise ValueError("derived dependency provenance mismatch")
        dependencies = list(
            session.scalars(
                select(RegulatoryChunk).where(
                    RegulatoryChunk.user_file_id == user_file_id,
                    RegulatoryChunk.id.in_(binding.dependency_ids),
                )
            )
        )
        rebuilt = rebuild_context_aggregates(
            [*(dated_dependency(row) for row in dependencies), candidate],
            changed_ids=[canonical.id],
        )[-1]
        if rebuilt.text != binding.representation_text:
            raise ValueError("derived aggregate content mismatch")
    else:
        predecessor = canonical.chunk_metadata.get("bound_to_regulatory_chunk_id")
        target_id = binding.representation_metadata.get("bound_to_regulatory_chunk_id")
        target = (
            session.get(RegulatoryChunk, target_id)
            if isinstance(target_id, str)
            else None
        )
        if (
            not isinstance(predecessor, str)
            or binding.representation_text != canonical.text
            or target is None
            or target.user_file_id != user_file_id
            or binding.dependency_ids != [target.id]
        ):
            raise ValueError("image companion provenance mismatch")
        from onyx.document_index.publication_models import ObservedPublicationProjection

        if isinstance(binding.projection, ObservedPublicationProjection):
            from onyx.db.regulatory_annex_changes import capture_canonical_scope
            from onyx.regulatory.amendments.annexes.selective_impact import (
                recover_source_membership,
            )

            metadata = dict(canonical.chunk_metadata)
            if target.id != predecessor:
                recovered = recover_source_membership(
                    capture_canonical_scope(session, user_file_id)
                )
                if recovered.get(canonical.id) != [target.id]:
                    raise ValueError("observed image split lineage is unavailable")
                metadata["bound_to_regulatory_chunk_id"] = target.id
                metadata["source_regulatory_chunk_ids"] = []
            if metadata != binding.representation_metadata:
                raise ValueError("observed image source metadata changed")
            dated_dependency(target)
        else:
            ancestor = target
            visited: set[str] = set()
            while ancestor.id != predecessor:
                if ancestor.id in visited or ancestor.supersedes_chunk_id is None:
                    raise ValueError(
                        "image companion target has no reviewed legal lineage"
                    )
                visited.add(ancestor.id)
                parent = session.get(RegulatoryChunk, ancestor.supersedes_chunk_id)
                if parent is None or parent.user_file_id != user_file_id:
                    raise ValueError("image companion target lineage leaves file scope")
                ancestor = parent
            target = dated_dependency(target)
            for key in (
                "image_file_id",
                "image_file_ids",
                "source_asset_ids",
                "annex_element_ids",
            ):
                if binding.representation_metadata.get(
                    key
                ) != target.chunk_metadata.get(key):
                    raise ValueError("image companion source evidence mismatch")
    from onyx.regulatory.chunk_evidence import chunk_evidence

    evidence = chunk_evidence(binding.representation_metadata)
    if source.get("image_file_id") != evidence.image_file_id or source.get(
        "source_links"
    ) != (json.dumps(evidence.source_links) if evidence.source_links else None):
        raise ValueError("temporal source/image evidence mismatch")
    expected = remove_invalid_unicode_chars(
        source["doc_summary"]
        + binding.representation_text
        + source["chunk_context"]
        + (source.get("metadata_suffix") or "")
    )
    if source["content"] != expected:
        raise ValueError(
            "projection source does not represent canonical/derived content"
        )
    starts = [
        value
        for value in (canonical.validity_start_date, binding.effective_start)
        if value is not None
    ]
    ends = [
        value
        for value in (canonical.validity_end_date, binding.effective_end)
        if value is not None
    ]
    if (max(starts) if starts else None) != binding.effective_start or (
        min(ends) if ends else None
    ) != binding.effective_end:
        raise ValueError("temporal projection exceeds canonical legal window")
    identity = binding.index.temporal_lookup_identity()
    payload = binding.model_dump(mode="json")
    existing = session.get(RegulatoryTemporalProjection, binding.id)
    if existing is not None:
        if existing.retired_at is not None:
            raise ValueError("retired temporal identity cannot be reactivated")
        if existing.payload != payload:
            raise ValueError(
                "temporal projection identity reused with different payload"
            )
        return
    overlapping = session.scalar(
        select(RegulatoryTemporalProjection.id).where(
            RegulatoryTemporalProjection.canonical_chunk_id == canonical.id,
            RegulatoryTemporalProjection.retired_at.is_(None),
            RegulatoryTemporalProjection.index_identity_sha256 == identity,
            or_(
                RegulatoryTemporalProjection.effective_end.is_(None),
                RegulatoryTemporalProjection.effective_end
                > (binding.effective_start or datetime.date.min),
            ),
            or_(
                RegulatoryTemporalProjection.effective_start.is_(None),
                RegulatoryTemporalProjection.effective_start
                < (binding.effective_end or datetime.date.max),
            ),
        )
    )
    if overlapping is not None:
        raise ValueError("temporal projection overlaps qualified history")
    from onyx.db.regulatory_canonical_revisions import retain_canonical_revision
    from onyx.regulatory.amendments.annexes.publication_representations import _snapshot

    canonical_revision_id = retained_revision_id or retain_canonical_revision(
        session, _snapshot(canonical)
    )
    session.add(
        RegulatoryTemporalProjection(
            canonical_revision_id=canonical_revision_id,
            id=binding.id,
            user_file_id=user_file_id,
            canonical_chunk_id=canonical.id,
            index_uuid=binding.index.index_uuid,
            index_identity_sha256=identity,
            projection_ordinal=binding.projection.ordinal,
            effective_start=binding.effective_start,
            effective_end=binding.effective_end,
            payload=payload,
            payload_sha256=publication_digest(payload),
        )
    )
    session.flush()


def get_indexed_temporal_projection(
    session: Session,
    canonical_chunk_id: str,
    *,
    index: "PublicationIndexSnapshot",
    as_of_date: datetime.date,
) -> "AnnexTemporalProjection | None":
    """Public caller supplies its frozen actual query index/model configuration."""
    from onyx.db.models import RegulatoryTemporalProjection
    from onyx.document_index.publication_models import publication_digest
    from onyx.regulatory.amendments.annexes.models import AnnexTemporalProjection

    row = session.scalars(
        select(RegulatoryTemporalProjection).where(
            RegulatoryTemporalProjection.canonical_chunk_id == canonical_chunk_id,
            RegulatoryTemporalProjection.retired_at.is_(None),
            RegulatoryTemporalProjection.index_identity_sha256.in_(
                index.temporal_lookup_identities()
            ),
            or_(
                RegulatoryTemporalProjection.effective_start.is_(None),
                RegulatoryTemporalProjection.effective_start <= as_of_date,
            ),
            or_(
                RegulatoryTemporalProjection.effective_end.is_(None),
                RegulatoryTemporalProjection.effective_end > as_of_date,
            ),
        )
    ).one_or_none()
    if row is None:
        return None
    if publication_digest(row.payload) != row.payload_sha256:
        raise ValueError("temporal binding payload changed")
    from onyx.db.regulatory_canonical_revisions import (
        validate_temporal_canonical_revision,
    )

    validate_temporal_canonical_revision(session, row)
    binding = AnnexTemporalProjection.model_validate(row.payload)

    if not accepts_publication_projection(index, binding.projection):
        raise ValueError(
            "temporal binding encoder receipt is not accepted by query configuration"
        )
    return binding


def resolve_preparation_context_call(
    *,
    user_file_id: UUID,
    call: ContextGenerationCall,
    generate: Callable[[], str],
) -> ContextGenerationCall:
    """Cache exact generation receipts without activating projections or vectors."""
    from onyx.db.engine.sql_engine import get_session_with_current_tenant

    def validated(payload: dict[str, Any]) -> ContextGenerationCall:
        stored = ContextGenerationCall.model_validate(payload)
        if stored.model_copy(update={"output": ""}) != call.model_copy(
            update={"output": ""}
        ):
            raise ValueError("context generation cache input proof changed")
        if not stored.output.strip():
            raise ValueError("context generation cache has no output")
        return stored

    with get_session_with_current_tenant() as session:
        cached = session.scalar(
            select(RegulatoryContextGeneration.payload).where(
                RegulatoryContextGeneration.user_file_id == user_file_id,
                RegulatoryContextGeneration.request_sha256 == call.request_sha256,
            )
        )
    if cached is not None:
        return validated(cached)
    generated = call.model_copy(update={"output": generate()})
    validated(generated.model_dump(mode="json"))
    with get_session_with_current_tenant() as session:
        session.execute(
            insert(RegulatoryContextGeneration)
            .values(
                id=uuid4(),
                user_file_id=user_file_id,
                request_sha256=call.request_sha256,
                payload=generated.model_dump(mode="json"),
            )
            .on_conflict_do_nothing(constraint="uq_context_generation_file_hash")
        )
        stored = session.scalar(
            select(RegulatoryContextGeneration.payload).where(
                RegulatoryContextGeneration.user_file_id == user_file_id,
                RegulatoryContextGeneration.request_sha256 == call.request_sha256,
            )
        )
        assert stored is not None
        result = validated(stored)
        session.commit()
        return result
