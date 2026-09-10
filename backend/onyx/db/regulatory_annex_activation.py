"""Atomic legal, source and derived history activation under file publication authority."""

import json
from collections.abc import Mapping
from datetime import date, datetime, timezone
from uuid import UUID, uuid5

from pydantic import JsonValue
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.models import (
    AnnexPublicationManifest,
    RegulatoryAnnex,
    RegulatoryAnnexElement,
    RegulatoryAnnexElementChunk,
    RegulatoryAnnexRevision,
    RegulatoryAnnexRevisionElement,
    RegulatoryChunk,
    RegulatoryTemporalProjection,
)
from onyx.db.regulatory_annex_execution import (
    owned_execution,
    validate_database_baseline,
    validate_staged_history,
)
from onyx.db.regulatory_publication import PublicationStore
from onyx.document_index.publication_models import (
    FileOwnership,
    PublicationVerification,
    publication_digest,
)
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.amendments.annexes.models import (
    AnnexChangeDraft,
    AnnexPublicationPreparation,
    AnnexTemporalProjection,
)
from onyx.regulatory.amendments.annexes.publication_execution_models import (
    AnnexPublicationDelivery,
    AnnexPublicationOperations,
)


def close_preapproved_temporal_binding(
    session: Session,
    *,
    previous: AnnexTemporalProjection,
    updated: AnnexTemporalProjection,
    user_file_id: UUID,
) -> None:
    """Only shorten a frozen prior interval; retain all source/vector/context identity."""
    row = session.get(RegulatoryTemporalProjection, previous.id)
    if (
        row is None
        or row.user_file_id != user_file_id
        or row.payload_sha256 != publication_digest(previous.model_dump(mode="json"))
        or row.payload != previous.model_dump(mode="json")
    ):
        raise ValueError("preapproved temporal baseline changed")
    before, after = previous.model_dump(mode="json"), updated.model_dump(mode="json")
    before_source, after_source = (
        json.loads(previous.projection.source_json),
        json.loads(updated.projection.source_json),
    )
    if (
        previous.id != updated.id
        or not previous.index.matches_temporal_index(updated.index)
        or previous.effective_start != updated.effective_start
        or updated.effective_end is None
        or previous.effective_end is not None
        and updated.effective_end > previous.effective_end
        or updated.effective_start is not None
        and updated.effective_end <= updated.effective_start
    ):
        raise ValueError("invalid preapproved temporal closing")
    for source in (before_source, after_source):
        source.pop("validity_end_date", None)
    for payload in (before, after):
        payload.pop("effective_end")
        payload.pop("index")
        payload["projection"].pop("source_json")
        payload["projection"]["embedding_config_json"] = json.loads(
            payload["projection"]["embedding_config_json"]
        )
    if before_source != after_source or before != after:
        raise ValueError("temporal closing changes immutable representation")
    # The enclosing manifest retains the original full payload and the approved final operation.
    payload = updated.model_dump(mode="json")
    row.effective_end, row.payload, row.payload_sha256 = (
        updated.effective_end,
        payload,
        publication_digest(payload),
    )
    session.flush()


def retire_preapproved_temporal_binding(
    session: Session,
    *,
    previous: AnnexTemporalProjection,
    prepared: AnnexPublicationPreparation,
    bindings: list[AnnexTemporalProjection],
    now: datetime,
) -> None:
    row = session.get(RegulatoryTemporalProjection, previous.id)
    if (
        row is None
        or row.retired_at is not None
        or row.user_file_id != prepared.user_file_id
        or row.payload != previous.model_dump(mode="json")
        or row.payload_sha256 != publication_digest(row.payload)
    ):
        raise ValueError("preapproved retired binding changed")
    canonical_id = json.loads(previous.projection.source_json)["regulatory_chunk_id"]
    legal = next(
        item for item in prepared.legal.canonical_rows if item.id == canonical_id
    )
    start = max(
        previous.effective_start or date.min, legal.validity_start_date or date.min
    )
    end = min(previous.effective_end or date.max, legal.validity_end_date or date.max)
    intervals = sorted(
        (binding.effective_start or date.min, binding.effective_end or date.max)
        for binding in bindings
        if binding.index.index_uuid == previous.index.index_uuid
        and json.loads(binding.projection.source_json)["regulatory_chunk_id"]
        == canonical_id
    )
    covered = start
    for lower, upper in intervals:
        if lower <= covered:
            covered = max(covered, upper)
    if covered < end:
        raise ValueError("retired binding would lose preapproved historical coverage")
    row.retired_at = now
    session.flush()


def _activate_legal(session: Session, prepared: AnnexPublicationPreparation) -> None:
    for wanted in prepared.legal.canonical_rows:
        row = session.get(RegulatoryChunk, wanted.id)
        if row is None:
            row = RegulatoryChunk(
                id=wanted.id,
                user_file_id=prepared.user_file_id,
                text=wanted.text,
                position=wanted.position,
                chunk_type=wanted.chunk_type,
                chunk_metadata=wanted.metadata,
                heading_path=wanted.heading_path,
                validity_start_date=wanted.validity_start_date,
                validity_end_date=wanted.validity_end_date,
                status=wanted.status,
                source=wanted.source,
                projection_ordinal=wanted.projection_ordinal,
            )
            session.add(row)
        else:
            # Prepared legal rows contain only reviewed predecessor closures or cessation.
            row.validity_end_date, row.status = wanted.validity_end_date, wanted.status
    session.flush()
    for wanted in prepared.legal.canonical_rows:
        row = session.get(RegulatoryChunk, wanted.id)
        assert row is not None
        row.supersedes_chunk_id, row.superseded_by_chunk_id = (
            wanted.supersedes_chunk_id,
            wanted.superseded_by_chunk_id,
        )
    session.flush()


def _element_ids(metadata: Mapping[str, JsonValue]) -> list[str]:
    values = metadata.get("annex_element_ids")
    if not isinstance(values, list) or not all(
        isinstance(value, str) for value in values
    ):
        return []
    return [value for value in values if isinstance(value, str)]


def _activate_sources(
    session: Session,
    delivery: AnnexPublicationDelivery,
    draft: AnnexChangeDraft,
    prepared: AnnexPublicationPreparation,
    now: datetime,
) -> None:
    if (
        draft.baseline is None
        or draft.baseline.revision_id is None
        or draft.new_extraction is None
        or draft.new_evidence_remapping is None
    ):
        raise ValueError("frozen annex source revision missing")
    previous = session.get(RegulatoryAnnexRevision, UUID(draft.baseline.revision_id))
    if previous is None or previous.baseline_sha256 != draft.baseline.baseline_sha256:
        raise ValueError("annex source revision baseline changed")
    annex = session.get(RegulatoryAnnex, previous.annex_id)
    if annex is None or annex.user_file_id != prepared.user_file_id:
        raise ValueError("annex source scope changed")
    old_end = previous.effective_end
    if (
        previous.effective_start is not None
        and previous.effective_start >= prepared.legal.effective_start
        or old_end is not None
        and old_end <= prepared.legal.effective_start
    ):
        raise ValueError("annex source revision no longer covers publication")
    previous.effective_end = prepared.legal.effective_start
    identifier = uuid5(delivery.change_set_id, "annex-source-revision")
    revision = RegulatoryAnnexRevision(
        id=identifier,
        annex_id=annex.id,
        predecessor_revision_id=previous.id,
        baseline_sha256=context_hash([delivery.review_sha256, "source"]),
        snapshot={
            "extraction": draft.new_extraction.model_dump(mode="json"),
            "new_evidence_remapping": draft.new_evidence_remapping.model_dump(
                mode="json"
            ),
            "review_id": str(delivery.change_set_id),
            "review_sha256": delivery.review_sha256,
        },
        effective_start=prepared.legal.effective_start,
        effective_end=prepared.legal.effective_end or old_end,
        approved_at=now,
    )
    session.add(revision)
    session.flush()
    mappings = {item.position: item for item in draft.new_evidence_remapping.elements}
    for position, element in enumerate(draft.new_extraction.elements):
        mapping = mappings[position]
        identity = session.get(RegulatoryAnnexElement, mapping.element_id)
        if identity is None:
            session.add(
                RegulatoryAnnexElement(id=mapping.element_id, annex_id=annex.id)
            )
            session.flush()
        elif identity.annex_id != annex.id:
            raise ValueError("NEW source element identity outside annex")
        session.add(
            RegulatoryAnnexRevisionElement(
                revision_id=identifier,
                element_id=mapping.element_id,
                position=position,
                payload=element.model_dump(mode="json"),
            )
        )
        session.flush()
        chunk_ids = {
            plan.row.id
            for plan in prepared.projections
            if (plan.effective_start or prepared.legal.effective_start)
            <= prepared.legal.effective_start
            and (
                plan.effective_end is None
                or prepared.legal.effective_start < plan.effective_end
            )
            and str(mapping.element_id) in _element_ids(plan.row.metadata)
        }
        for chunk_id in chunk_ids:
            session.add(
                RegulatoryAnnexElementChunk(
                    revision_id=identifier,
                    element_id=mapping.element_id,
                    chunk_id=chunk_id,
                )
            )
    if (
        draft.after_window_authority
        and draft.after_window_authority.kind == "restore_predecessor"
    ):
        restored = RegulatoryAnnexRevision(
            id=uuid5(delivery.change_set_id, "annex-source-restoration"),
            annex_id=annex.id,
            predecessor_revision_id=identifier,
            source_asset_id=previous.source_asset_id,
            baseline_sha256=context_hash(
                [delivery.review_sha256, "source-restoration"]
            ),
            snapshot=previous.snapshot,
            effective_start=prepared.legal.effective_end,
            effective_end=old_end,
            approved_at=now,
        )
        session.add(restored)
        session.flush()
        replacements = {
            old: new for new, old in prepared.legal.restoration_predecessors.items()
        }
        for old_element in session.scalars(
            select(RegulatoryAnnexRevisionElement).where(
                RegulatoryAnnexRevisionElement.revision_id == previous.id
            )
        ):
            session.add(
                RegulatoryAnnexRevisionElement(
                    revision_id=restored.id,
                    element_id=old_element.element_id,
                    position=old_element.position,
                    payload=old_element.payload,
                )
            )
        session.flush()
        for link in session.scalars(
            select(RegulatoryAnnexElementChunk).where(
                RegulatoryAnnexElementChunk.revision_id == previous.id
            )
        ):
            session.add(
                RegulatoryAnnexElementChunk(
                    revision_id=restored.id,
                    element_id=link.element_id,
                    chunk_id=replacements.get(link.chunk_id, link.chunk_id),
                )
            )
        annex.latest_approved_revision_id = restored.id
    else:
        annex.latest_approved_revision_id = identifier
    session.flush()


def activate_publication(
    owner: FileOwnership,
    delivery: AnnexPublicationDelivery,
    prepared: AnnexPublicationPreparation,
    operations: AnnexPublicationOperations,
    verifications: list[PublicationVerification],
) -> None:
    from onyx.db.regulatory_context_projections import activate_temporal_projection

    with owned_execution(owner, delivery) as (session, review, draft):
        validate_database_baseline(session, draft, prepared)
        manifest = session.get(AnnexPublicationManifest, delivery.change_set_id)
        if (
            manifest is None
            or not manifest.es_started
            or manifest.payload != prepared.model_dump(mode="json")
            or manifest.operations != operations.model_dump(mode="json")
        ):
            raise ValueError("activation manifest differs from durable publication")
        validate_staged_history(session, draft, prepared, manifest)
        reservations = PublicationStore(owner.scope).lock_owned_snapshot(session, owner)
        if (
            list(reservations.ordinals) != prepared.reserved_ordinals
            or [proof.index for proof in verifications] != prepared.indexes
        ):
            raise ValueError("activation requires every frozen index/reservation")
        for proof in verifications:
            expected = [
                op.binding.projection
                for op in operations.operations
                if op.index_uuid == proof.index.index_uuid and op.binding
            ]
            if proof.reservations != reservations or proof.live_ordinals != tuple(
                sorted(item.ordinal for item in expected)
            ):
                raise ValueError("activation verification scope differs from manifest")
        now = datetime.now(timezone.utc)
        _activate_legal(session, prepared)
        from onyx.db.regulatory_context_projections import persist_context_evidence

        for view in prepared.views.values():
            persist_context_evidence(
                session, user_file_id=prepared.user_file_id, view=view
            )
        _activate_sources(session, delivery, draft, prepared, now)
        prior = {
            item.binding.id: item.binding
            for item in prepared.indexed_baseline
            if item.binding
        }
        bindings = [op.binding for op in operations.operations if op.binding]
        # Close reviewed old windows first, then register dependencies before consumers.
        for binding in bindings:
            existing = prior.get(binding.id)
            if existing is not None and existing != binding:
                close_preapproved_temporal_binding(
                    session,
                    previous=existing,
                    updated=binding,
                    user_file_id=prepared.user_file_id,
                )
        for identifier, previous in prior.items():
            if identifier not in {binding.id for binding in bindings}:
                retire_preapproved_temporal_binding(
                    session,
                    previous=previous,
                    prepared=prepared,
                    bindings=bindings,
                    now=now,
                )
        pending = list(bindings)
        while pending:
            ready = [
                binding
                for binding in pending
                if not any(
                    other is not binding
                    and other.index.index_uuid == binding.index.index_uuid
                    and json.loads(other.projection.source_json)["regulatory_chunk_id"]
                    in binding.dependency_ids
                    for other in pending
                )
            ]
            if not ready:
                raise ValueError("temporal dependency cycle")
            for binding in ready:
                activate_temporal_projection(
                    session, user_file_id=prepared.user_file_id, binding=binding
                )
                pending.remove(binding)
        manifest.approved_at = now
        review.status, review.error_message, review.heartbeat_at = "approved", None, now
        PublicationStore(owner.scope).finalize(session, owner, verifications[0])
