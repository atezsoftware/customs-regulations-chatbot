"""Atomic grouped review checkpoints and staged publication contracts."""

import datetime
import hashlib
from collections.abc import Mapping
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.amendment_sources import (
    get_source_package,
    list_source_assets,
    require_ready_source_package,
)
from onyx.db.file_record import get_filerecord_by_file_id_optional
from onyx.db.models import (
    AmendmentBatch,
    AmendmentProposal,
    AnnexChangeEvidence,
    AnnexChangeItem,
    AnnexChangeSet,
    AnnexPublicationIntent,
    RegulatoryChunk,
)
from onyx.db.regulatory_annexes import require_annex_file_scope
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.amendments.annexes.models import (
    AnnexCanonicalSnapshot,
    AnnexChangeDraft,
    AnnexInstructionGroup,
    AnnexReviewEvidence,
    AnnexReviewEvidenceScope,
)


def capture_canonical_scope(
    session: Session, user_file_id: UUID
) -> list[AnnexCanonicalSnapshot]:
    rows = session.scalars(
        select(RegulatoryChunk)
        .where(RegulatoryChunk.user_file_id == user_file_id)
        .order_by(RegulatoryChunk.position, RegulatoryChunk.id)
        .execution_options(populate_existing=True)
    )
    return [
        AnnexCanonicalSnapshot(
            id=row.id,
            user_file_id=str(row.user_file_id),
            chunk_type=row.chunk_type,
            status=row.status,
            projection_ordinal=row.projection_ordinal,
            supersedes_chunk_id=row.supersedes_chunk_id,
            superseded_by_chunk_id=row.superseded_by_chunk_id,
            position=row.position,
            text=row.text,
            heading_path=row.heading_path,
            metadata=row.chunk_metadata,
            source=row.source,
            validity_start_date=row.validity_start_date,
            validity_end_date=row.validity_end_date,
        )
        for row in rows
    ]


def list_annex_changes(session: Session, batch_id: int) -> list[AnnexChangeSet]:
    revisions = list(
        session.scalars(
            select(AnnexChangeSet)
            .where(AnnexChangeSet.batch_id == batch_id)
            .order_by(
                AnnexChangeSet.instruction_index, AnnexChangeSet.review_revision.desc()
            )
        )
    )
    latest: dict[UUID, AnnexChangeSet] = {}
    for revision in revisions:
        latest.setdefault(revision.logical_group_id, revision)
    return list(latest.values())


def persist_annex_checkpoint(
    session: Session,
    *,
    batch_id: int,
    lease_generation: int,
    draft: AnnexChangeDraft,
    environment: str,
) -> AnnexChangeSet | None:
    draft = AnnexChangeDraft.model_validate(draft.model_dump(mode="json"))
    batch = session.scalar(
        select(AmendmentBatch).where(AmendmentBatch.id == batch_id).with_for_update()
    )
    if (
        batch is None
        or batch.status != "analyzing"
        or batch.lease_generation != lease_generation
    ):
        session.rollback()
        return None
    covered = set(
        batch.processed_instruction_indices or range(batch.processed_instruction_count)
    )
    indices = set(draft.instruction_indices)
    if not indices.issubset(range(batch.instruction_count)):
        raise ValueError("instruction scope mismatch")
    existing = session.scalar(
        select(AnnexChangeSet).where(
            AnnexChangeSet.batch_id == batch_id,
            AnnexChangeSet.review_revision == 1,
            AnnexChangeSet.instruction_index == draft.instruction_indices[0],
        )
    )
    if existing is not None:
        if (
            existing.environment != environment
            or existing.review_sha256 != context_hash(draft.model_dump(mode="json"))
        ):
            raise ValueError("checkpoint review changed")
        if (
            existing.instruction_indices != draft.instruction_indices
            or not indices.issubset(covered)
        ):
            raise ValueError("checkpoint coverage mismatch")
        return existing
    if indices.intersection(covered):
        raise ValueError("instruction already checkpointed")
    validate_annex_review_scope(
        session, batch=batch, draft=draft, environment=environment
    )
    payload = draft.model_dump(mode="json")
    ready = (
        not draft.issues
        and draft.patch_plan is not None
        and draft.patch_plan.ready
        and draft.impact is not None
        and draft.impact.ready
    )
    if ready:
        validate_prepared_annex_change(
            session, batch=batch, draft=draft, environment=environment
        )
    change = AnnexChangeSet(
        batch_id=batch_id,
        logical_group_id=uuid4(),
        review_revision=1,
        instruction_index=draft.instruction_indices[0],
        instruction_indices=draft.instruction_indices,
        environment=environment,
        user_file_id=draft.user_file_id,
        status="pending" if ready else "blocked",
        review_payload=payload,
        review_sha256=context_hash(payload),
    )
    session.add(change)
    session.flush()
    _add_review_associations(session, change=change, draft=draft)
    covered.update(indices)
    batch.processed_instruction_indices = sorted(covered)
    batch.processed_instruction_count = len(covered)
    batch.heartbeat_at = datetime.datetime.now(datetime.timezone.utc)
    session.commit()
    return change


def load_review_evidence_scope(
    session: Session,
    *,
    batch_id: int,
    user_file_id: UUID,
    environment: str,
    created_by: UUID | None,
) -> AnnexReviewEvidenceScope:
    batch = session.get(AmendmentBatch, batch_id)
    if (
        batch is None
        or batch.created_by != created_by
        or str(user_file_id) not in batch.user_file_ids
    ):
        raise ValueError("review evidence batch scope mismatch")
    file = require_annex_file_scope(session, batch.document_set_id, user_file_id)
    old_ids = {file.file_id}
    for row in capture_canonical_scope(session, user_file_id):
        image_id = row.metadata.get("image_file_id")
        if isinstance(image_id, str):
            old_ids.add(image_id)
        image_ids = row.metadata.get("image_file_ids")
        if isinstance(image_ids, list):
            old_ids.update(value for value in image_ids if isinstance(value, str))
    new_ids: list[str] = []
    if batch.source_package_id is not None:
        package = get_source_package(
            session,
            package_id=batch.source_package_id,
            document_set_id=batch.document_set_id,
            environment=environment,
        )
        if package is None or package.created_by != created_by:
            raise ValueError("source package owner scope mismatch")
        new_ids = [asset.file_id for asset in list_source_assets(session, package.id)]
    return AnnexReviewEvidenceScope(
        batch_id=batch_id,
        document_set_id=batch.document_set_id,
        user_file_id=user_file_id,
        created_by=created_by,
        environment=environment,
        old_original_file_ids=sorted(old_ids),
        new_original_file_ids=new_ids,
    )


def validate_frozen_evidence(
    session: Session,
    *,
    scope: AnnexReviewEvidenceScope,
    evidence: list[AnnexReviewEvidence],
) -> None:
    for item in evidence:
        record = get_filerecord_by_file_id_optional(item.file_id, session)
        if (
            record is None
            or not isinstance(record.file_metadata, Mapping)
            or cast(Mapping[str, object], record.file_metadata).get(
                "annex_review_scope"
            )
            != scope.storage_identity()
        ):
            raise ValueError("frozen evidence scope mismatch")
        metadata = cast(Mapping[str, object], record.file_metadata)
        if (
            record.file_type != item.mime_type
            or metadata.get("byte_count") != item.byte_count
            or metadata.get("side") != item.side
            or metadata.get("kind") != item.kind
            or metadata.get("locator") != item.locator.model_dump(mode="json")
            or metadata.get("sha256") != item.sha256
            or metadata.get("parent_sha256") != item.parent_sha256
            or metadata.get("parent_file_id") != item.parent_file_id
        ):
            raise ValueError("frozen evidence identity mismatch")


def get_annex_review_evidence(
    session: Session,
    *,
    change_set_id: UUID,
    evidence_id: UUID,
    document_set_id: int,
    created_by: UUID | None,
    environment: str,
) -> AnnexReviewEvidence:
    association = session.scalar(
        select(AnnexChangeEvidence)
        .join(AnnexChangeSet, AnnexChangeSet.id == AnnexChangeEvidence.change_set_id)
        .join(AmendmentBatch, AmendmentBatch.id == AnnexChangeSet.batch_id)
        .where(
            AnnexChangeSet.id == change_set_id,
            AnnexChangeEvidence.evidence_id == evidence_id,
            AmendmentBatch.document_set_id == document_set_id,
            AmendmentBatch.created_by == created_by,
            AnnexChangeSet.environment == environment,
        )
    )
    if association is None:
        raise ValueError("annex review evidence scope mismatch")
    return AnnexReviewEvidence.model_validate(association.payload)


def validate_prepared_annex_change(
    session: Session,
    *,
    batch: AmendmentBatch,
    draft: AnnexChangeDraft,
    environment: str,
) -> None:
    """Check the frozen preparation contract before a group can become pending."""
    from onyx.regulatory.amendments.annexes.patch_plan import prepare_annex_patch

    if (
        draft.user_file_id is None
        or draft.source_package_id is None
        or draft.baseline is None
        or draft.old_extraction is None
        or draft.new_extraction is None
        or draft.comparison is None
        or draft.patch_plan is None
        or draft.impact is None
        or draft.baseline_context is None
        or draft.effective_date is None
        or not draft.evidence
    ):
        raise ValueError("prepared annex review is incomplete")
    package = require_ready_source_package(
        session,
        package_id=draft.source_package_id,
        document_set_id=batch.document_set_id,
        environment=environment,
    )
    if (
        draft.source_text_sha256 is None
        or hashlib.sha256(batch.raw_text.encode()).hexdigest()
        != draft.source_text_sha256
        or package.created_by != batch.created_by
        or package.manifest_sha256 != draft.source_manifest_sha256
        or batch.source_text_sha256 != draft.source_text_sha256
    ):
        raise ValueError("prepared source ownership or identity changed")
    if draft.batch_id is not None:
        from onyx.file_store.file_store import get_default_file_store
        from onyx.regulatory.amendments.annexes.evidence import (
            read_original_source_text,
            read_source_graph,
        )

        if package.manifest_file_id is None or package.manifest_sha256 is None:
            raise ValueError("prepared source manifest missing")
        assets = list_source_assets(session, package.id)
        store = get_default_file_store()
        graph = read_source_graph(
            store,
            manifest_file_id=package.manifest_file_id,
            manifest_sha256=package.manifest_sha256,
            assets=assets,
        )
        if (
            graph != draft.source_graph
            or context_hash([item.model_dump(mode="json") for item in graph])
            != draft.source_graph_sha256
            or read_original_source_text(store, assets)[1]
            != draft.original_source_text_sha256
        ):
            raise ValueError("prepared source graph or original text changed")
    if (
        draft.submitted_source_text is not None
        and draft.submitted_source_text != batch.raw_text
    ):
        raise ValueError("submitted source text changed")
    if draft.date_resolution is not None and (
        draft.date_resolution.effective_start_date != draft.effective_date.isoformat()
        or draft.date_resolution.effective_end_date is not None
    ):
        raise ValueError("prepared effective date differs from reviewed resolution")
    if draft.new_extraction.evidence_view is not None and any(
        occurrence not in draft.source_graph
        for occurrence in draft.new_extraction.evidence_view.source_occurrences
    ):
        raise ValueError("NEW source occurrence outside frozen package graph")
    if draft.source_only_canonical_ids and (
        draft.comparison.changes
        or draft.items
        or set(draft.source_only_canonical_ids)
        != {
            element.canonical_chunk_id
            for element in draft.baseline.elements
            if element.canonical_chunk_id
        }
    ):
        raise ValueError(
            "source-only association differs from unchanged canonical scope"
        )
    if draft.raw_new_extraction is not None:
        from onyx.regulatory.amendments.annexes.corrections import (
            apply_bound_corrections,
            correction_input_hash,
        )

        if draft.new_extraction != apply_bound_corrections(
            draft.raw_new_extraction, draft.corrections
        ):
            raise ValueError("corrected extraction differs from immutable human review")
        if draft.corrections and (
            draft.corrected_by is None
            or draft.correction_reconciliation is None
            or not draft.correction_reconciliation.supported
            or draft.correction_reconciliation.input_sha256
            != correction_input_hash(draft, draft.corrections)
            or draft.correction_reconciliation.model_snapshot is None
        ):
            raise ValueError("human correction lacks source reconciliation")
    actual_plan = prepare_annex_patch(
        baseline=draft.baseline,
        old=draft.old_extraction,
        new=draft.new_extraction,
        comparison=draft.comparison,
        effective_date=draft.effective_date,
        package_complete=True,
    )
    if not actual_plan.ready or actual_plan != draft.patch_plan:
        raise ValueError("prepared patch validation changed")
    from onyx.regulatory.amendments.annexes.evidence import validate_compared_evidence
    from onyx.regulatory.amendments.annexes.models import AnnexOriginalEvidence

    validate_compared_evidence(
        old=draft.old_extraction,
        new=draft.new_extraction,
        old_originals=draft.baseline.originals,
        new_originals=[
            AnnexOriginalEvidence(
                file_id=asset.file_id,
                sha256=asset.sha256,
                mime_type=asset.mime_type,
                available=True,
            )
            for asset in list_source_assets(session, package.id)
        ],
        comparison=draft.comparison,
        evidence=draft.evidence,
    )
    if draft.new_evidence_remapping is not None:
        from onyx.regulatory.amendments.annexes.evidence import (
            validate_new_evidence_remapping,
        )

        validate_new_evidence_remapping(
            mapping=draft.new_evidence_remapping,
            new=draft.new_extraction,
            evidence=draft.evidence,
            asset_ids={
                asset.file_id: asset.id
                for asset in list_source_assets(session, package.id)
            },
        )
    if draft.preparation_configuration:
        actual_configuration = capture_preparation_configuration(
            session, user_file_id=draft.user_file_id
        )
        if any(
            draft.preparation_configuration.get(key) != value
            for key, value in actual_configuration.items()
        ):
            raise ValueError("prepared file or search configuration changed")
    from onyx.regulatory.amendments.annexes.staging import (
        staged_canonical_predecessors,
        validate_staged_items,
    )

    validate_staged_items(
        plan=actual_plan,
        baseline_scope=draft.baseline_scope,
        items=draft.items,
        comparison=draft.comparison,
        insertion_after_chunk_id=draft.insertion_after_chunk_id,
        evidence_remapping=draft.new_evidence_remapping,
    )
    from onyx.regulatory.amendments.annexes.context_dependencies import (
        compare_context_views,
        validate_complete_context_view,
    )
    from onyx.regulatory.amendments.annexes.staging import (
        canonical_snapshot_rows,
        prepare_staged_candidate_rows,
    )

    candidate_rows = prepare_staged_candidate_rows(
        baseline_scope=draft.baseline_scope,
        items=draft.items,
        effective_date=draft.effective_date,
        evidence_remapping=draft.new_evidence_remapping,
    )
    validate_complete_context_view(
        rows=canonical_snapshot_rows(draft.baseline_scope),
        view=draft.baseline_context,
        as_of_date=draft.effective_date,
    )
    validate_complete_context_view(
        rows=candidate_rows, view=draft.impact.prepared, as_of_date=draft.effective_date
    )
    expected_impact = compare_context_views(
        old=draft.baseline_context,
        new=draft.impact.prepared,
        direct_canonical_changes=[
            chunk.id for item in draft.items for chunk in item.new_chunks
        ],
        metadata_only=actual_plan.metadata_only,
        canonical_predecessors=staged_canonical_predecessors(draft.items)
        if draft.new_evidence_remapping is not None
        else None,
    )
    if expected_impact != draft.impact:
        raise ValueError(
            "prepared context impact differs from complete candidate views"
        )


def validate_annex_review_scope(
    session: Session,
    *,
    batch: AmendmentBatch,
    draft: AnnexChangeDraft,
    environment: str,
) -> None:
    if batch.superseded_by_batch_id is not None:
        raise ValueError("stale source text revision")
    if draft.batch_id is not None and draft.batch_id != batch.id:
        raise ValueError("draft batch identity changed")
    if draft.user_file_id is not None:
        require_annex_file_scope(session, batch.document_set_id, draft.user_file_id)
        if str(draft.user_file_id) not in batch.user_file_ids:
            raise ValueError("file outside frozen batch scope")
    if (
        draft.user_file_id is not None
        and draft.baseline_scope != capture_canonical_scope(session, draft.user_file_id)
    ):
        raise ValueError("canonical baseline changed")
    baseline_ids = {row.id for row in draft.baseline_scope}
    old_ids = [chunk_id for item in draft.items for chunk_id in item.old_chunk_ids]
    new_chunks = [chunk for item in draft.items for chunk in item.new_chunks]
    new_ids = [chunk.id for chunk in new_chunks]
    if not set(old_ids).issubset(baseline_ids) or len(old_ids) != len(set(old_ids)):
        raise ValueError("old canonical lineage scope mismatch")
    if len(new_ids) != len(set(new_ids)) or set(new_ids).intersection(baseline_ids):
        raise ValueError("prospective canonical identity collision")
    for chunk in new_chunks:
        UUID(chunk.id)
        if (
            chunk.user_file_id != str(draft.user_file_id)
            or session.get(RegulatoryChunk, chunk.id) is not None
        ):
            raise ValueError("prospective canonical scope mismatch")
    if draft.evidence:
        if draft.user_file_id is None:
            raise ValueError("evidence file scope missing")
        scope = load_review_evidence_scope(
            session,
            batch_id=batch.id,
            user_file_id=draft.user_file_id,
            environment=environment,
            created_by=batch.created_by,
        )
        validate_frozen_evidence(session, scope=scope, evidence=draft.evidence)
    if draft.source_package_id != batch.source_package_id:
        raise ValueError("source package scope mismatch")


def _add_review_associations(
    session: Session, *, change: AnnexChangeSet, draft: AnnexChangeDraft
) -> None:
    for position, item in enumerate(draft.items):
        session.add(
            AnnexChangeItem(
                change_set_id=change.id,
                position=position,
                operation=item.operation,
                old_chunk_ids=item.old_chunk_ids,
                prospective_chunk_ids=[chunk.id for chunk in item.new_chunks],
                payload=item.model_dump(mode="json"),
            )
        )
    for evidence in draft.evidence:
        session.add(
            AnnexChangeEvidence(
                change_set_id=change.id,
                evidence_id=evidence.id,
                file_id=evidence.file_id,
                payload=evidence.model_dump(mode="json"),
            )
        )


def require_current_annex_review(
    session: Session,
    *,
    change_set_id: UUID,
    expected_review_sha256: str,
    environment: str,
) -> AnnexChangeSet:
    review = session.get(AnnexChangeSet, change_set_id)
    if review is None or review.environment != environment:
        raise ValueError("review scope mismatch")
    # Serialize edits, decisions and sibling revisions through their immutable batch.
    session.scalar(
        select(AmendmentBatch)
        .where(AmendmentBatch.id == review.batch_id)
        .with_for_update()
    )
    session.refresh(review)
    latest = session.scalar(
        select(AnnexChangeSet)
        .where(AnnexChangeSet.logical_group_id == review.logical_group_id)
        .order_by(AnnexChangeSet.review_revision.desc())
        .limit(1)
    )
    if (
        latest is None
        or latest.id != review.id
        or review.review_sha256 != expected_review_sha256
    ):
        raise ValueError("stale review revision or hash")
    return review


def revise_annex_review(
    session: Session,
    *,
    change_set_id: UUID,
    expected_review_sha256: str,
    draft: AnnexChangeDraft,
    environment: str,
) -> AnnexChangeSet:
    previous = require_current_annex_review(
        session,
        change_set_id=change_set_id,
        expected_review_sha256=expected_review_sha256,
        environment=environment,
    )
    if previous.status not in ("pending", "blocked", "rejected", "failed"):
        raise ValueError("review state does not allow edits")
    if previous.publication_generation:
        raise ValueError("publication intent prevents review edits")
    batch = session.get(AmendmentBatch, previous.batch_id)
    assert batch is not None
    if (
        draft.instruction_indices != previous.instruction_indices
        or draft.source_package_id != batch.source_package_id
    ):
        raise ValueError("review instruction or source scope changed")
    prior = AnnexChangeDraft.model_validate(previous.review_payload)
    if (draft.annex_label, draft.instruction_texts, draft.target_sources) != (
        prior.annex_label,
        prior.instruction_texts,
        prior.target_sources,
    ) or (prior.user_file_id is not None and draft.user_file_id != prior.user_file_id):
        raise ValueError("immutable review target scope changed")
    if prior.raw_new_extraction is not None and (
        draft.raw_new_extraction != prior.raw_new_extraction
        or draft.old_extraction != prior.old_extraction
        or draft.evidence != prior.evidence
    ):
        raise ValueError("immutable raw review evidence changed")
    draft = AnnexChangeDraft.model_validate(draft.model_dump(mode="json"))
    validate_annex_review_scope(
        session, batch=batch, draft=draft, environment=environment
    )
    ready = (
        not draft.issues
        and draft.patch_plan is not None
        and draft.patch_plan.ready
        and draft.impact is not None
        and draft.impact.ready
    )
    if ready:
        validate_prepared_annex_change(
            session, batch=batch, draft=draft, environment=environment
        )
    payload = draft.model_dump(mode="json")
    review = AnnexChangeSet(
        batch_id=batch.id,
        logical_group_id=previous.logical_group_id,
        review_revision=previous.review_revision + 1,
        instruction_index=previous.instruction_index,
        instruction_indices=previous.instruction_indices,
        environment=environment,
        user_file_id=draft.user_file_id,
        status="pending" if ready else "blocked",
        review_sha256=context_hash(payload),
        review_payload=payload,
    )
    session.add(review)
    session.flush()
    _add_review_associations(session, change=review, draft=draft)
    session.commit()
    return review


def capture_preparation_configuration(
    session: Session, *, user_file_id: UUID
) -> dict[str, str]:
    """Hash complete persisted file/settings inputs without exposing credentials."""
    from sqlalchemy import inspect

    from onyx.db.models import UserFile
    from onyx.db.search_settings import get_current_search_settings

    file = session.get(UserFile, user_file_id)
    if file is None:
        raise ValueError("file unavailable")
    settings = get_current_search_settings(session)
    session.flush()
    configuration = {
        name: context_hash(
            {
                column.key: getattr(row, column.key)
                for column in inspect(type(row)).columns
                if column.key not in ("created_at", "updated_at", "last_accessed_at")
            }
        )
        for name, row in (("user_file", file), ("search_settings", settings))
    }
    from onyx.configs import app_configs
    from onyx.prompts.contextual_retrieval import (
        CONTEXTUAL_RAG_PROMPT1,
        CONTEXTUAL_RAG_PROMPT2,
    )
    from shared_configs.configs import DOC_EMBEDDING_CONTEXT_SIZE

    configuration["runtime_policy"] = context_hash(
        [
            app_configs.ENABLE_CONTEXTUAL_RAG,
            app_configs.USE_DOCUMENT_SUMMARY,
            app_configs.USE_CHUNK_SUMMARY,
            app_configs.REGULATORY_BATCH_INDEXING_ENABLED,
            app_configs.BLURB_SIZE,
            DOC_EMBEDDING_CONTEXT_SIZE,
            CONTEXTUAL_RAG_PROMPT1,
            CONTEXTUAL_RAG_PROMPT2,
        ]
    )
    return configuration


def resolve_annex_instruction_file(
    session: Session,
    *,
    batch: AmendmentBatch,
    annex_label: str,
    target_sources: list[str],
    effective_date: datetime.date,
) -> UUID:
    from onyx.db.regulatory_annexes import load_legacy_annex_chunks
    from onyx.regulatory.amendments.structural_target import source_identity_matches

    matches: list[UUID] = []
    for identifier in batch.user_file_ids:
        file_id = UUID(identifier)
        file = require_annex_file_scope(session, batch.document_set_id, file_id)
        if not all(
            source_identity_matches(source, file.name) for source in target_sources
        ):
            continue
        if load_legacy_annex_chunks(
            session,
            document_set_id=batch.document_set_id,
            user_file_id=file_id,
            annex_label=annex_label,
            as_of_date=effective_date,
        ):
            matches.append(file_id)
    if len(matches) != 1:
        raise ValueError(
            "annex_file_missing" if not matches else "annex_file_ambiguous"
        )
    return matches[0]


def get_scoped_annex_review(
    session: Session,
    *,
    change_set_id: UUID,
    batch_id: int,
    environment: str,
    created_by: UUID | None,
) -> AnnexChangeSet:
    row = session.scalar(
        select(AnnexChangeSet)
        .join(AmendmentBatch)
        .where(
            AnnexChangeSet.id == change_set_id,
            AnnexChangeSet.batch_id == batch_id,
            AnnexChangeSet.environment == environment,
            AmendmentBatch.created_by == created_by,
        )
    )
    if row is None:
        raise ValueError("review scope mismatch")
    return row


def queue_annex_publication(
    session: Session,
    *,
    change_set_id: UUID,
    expected_review_sha256: str,
    environment: str,
    tenant_id: str,
    database_identity: str,
    decided_by: UUID | None,
    retry: bool = False,
) -> "AnnexPublicationIntent":
    from onyx.db.models import AnnexPublicationIntent

    review = require_current_annex_review(
        session,
        change_set_id=change_set_id,
        expected_review_sha256=expected_review_sha256,
        environment=environment,
    )
    if review.status in ("approving", "preparing", "publishing"):
        intent = session.scalar(
            select(AnnexPublicationIntent).where(
                AnnexPublicationIntent.change_set_id == review.id,
                AnnexPublicationIntent.publication_generation
                == review.publication_generation,
            )
        )
        if intent is None or (intent.tenant_id, intent.database_identity) != (
            tenant_id,
            database_identity,
        ):
            raise ValueError("publication intent scope mismatch")
        return intent
    if review.status != ("failed" if retry else "pending"):
        raise ValueError("review state does not allow publication")
    batch = session.get(AmendmentBatch, review.batch_id)
    assert batch is not None
    draft = AnnexChangeDraft.model_validate(review.review_payload)
    if not draft.preparation_configuration or draft.new_evidence_remapping is None:
        raise ValueError("live preparation must be revalidated before approval")
    lock_annex_preparation_scope(session, draft.user_file_id)
    validate_annex_review_scope(
        session, batch=batch, draft=draft, environment=environment
    )
    validate_prepared_annex_change(
        session, batch=batch, draft=draft, environment=environment
    )
    if not tenant_id or not database_identity:
        raise ValueError("publication scope missing")
    review.status = "approving"
    review.publication_generation += 1
    review.decided_by = decided_by
    review.decided_at = review.heartbeat_at = datetime.datetime.now(
        datetime.timezone.utc
    )
    review.error_message = None
    intent = AnnexPublicationIntent(
        change_set_id=review.id,
        logical_group_id=review.logical_group_id,
        review_revision=review.review_revision,
        review_sha256=review.review_sha256,
        publication_generation=review.publication_generation,
        tenant_id=tenant_id,
        environment=environment,
        database_identity=database_identity,
    )
    session.add(intent)
    session.commit()
    return intent


def reject_annex_review(
    session: Session,
    *,
    change_set_id: UUID,
    expected_review_sha256: str,
    environment: str,
    decided_by: UUID | None,
) -> AnnexChangeSet:
    review = require_current_annex_review(
        session,
        change_set_id=change_set_id,
        expected_review_sha256=expected_review_sha256,
        environment=environment,
    )
    if (
        review.status not in ("pending", "blocked", "rejected")
        or review.publication_generation
    ):
        raise ValueError("review state does not allow rejection")
    review.status, review.decided_by = "rejected", decided_by
    review.decided_at = datetime.datetime.now(datetime.timezone.utc)
    session.commit()
    return review


def list_annex_review_revisions(
    session: Session, *, logical_group_id: UUID
) -> list[AnnexChangeSet]:
    return list(
        session.scalars(
            select(AnnexChangeSet)
            .where(AnnexChangeSet.logical_group_id == logical_group_id)
            .order_by(AnnexChangeSet.review_revision)
        )
    )


def create_source_text_revision(
    session: Session,
    *,
    batch_id: int,
    raw_text: str,
    expected_source_text_sha256: str,
    environment: str,
    created_by: UUID | None,
    source_package_id: UUID | None = None,
) -> AmendmentBatch:
    from onyx.db.amendment_sources import attach_source_package_to_batch
    from onyx.db.regulatory_amendments import create_batch

    batch = session.scalar(
        select(AmendmentBatch).where(AmendmentBatch.id == batch_id).with_for_update()
    )
    if batch is None or batch.created_by != created_by:
        raise ValueError("source batch scope mismatch")
    if (
        batch.superseded_by_batch_id is not None
        or hashlib.sha256(batch.raw_text.encode()).hexdigest()
        != expected_source_text_sha256
    ):
        raise ValueError("stale source text revision")
    if any(
        review.status in ("approving", "preparing", "publishing", "approved")
        or review.publication_generation
        for review in list_annex_changes(session, batch_id)
    ):
        raise ValueError("publication state prevents source edits")
    legacy_proposals = session.scalars(
        select(AmendmentProposal)
        .where(AmendmentProposal.batch_id == batch.id)
        .with_for_update()
    )
    if any(
        proposal.status in ("approving", "approved", "approval_failed")
        or proposal.applied_new_chunk_id
        for proposal in legacy_proposals
    ):
        raise ValueError("legacy publication state prevents source edits")
    revised = create_batch(
        session,
        document_set_id=batch.document_set_id,
        user_file_ids=[UUID(value) for value in batch.user_file_ids],
        raw_text=raw_text,
        created_by=created_by,
    )
    revised.source_parent_batch_id = batch.id
    revised.source_text_sha256 = hashlib.sha256(raw_text.encode()).hexdigest()
    package_id = source_package_id or batch.source_package_id
    if package_id is not None:
        package = require_ready_source_package(
            session,
            package_id=package_id,
            document_set_id=batch.document_set_id,
            environment=environment,
        )
        if package.created_by != created_by:
            raise ValueError("source package owner mismatch")
        attach_source_package_to_batch(
            session, batch=revised, package_id=package_id, environment=environment
        )
    batch.superseded_by_batch_id = revised.id
    session.commit()
    return revised


def lock_annex_preparation_scope(session: Session, user_file_id: UUID | None) -> None:
    from sqlalchemy import inspect

    from onyx.db.models import UserFile
    from onyx.db.search_settings import get_current_search_settings

    if user_file_id is None:
        raise ValueError("review file scope missing")
    session.scalar(
        select(UserFile)
        .where(UserFile.id == user_file_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    list(
        session.scalars(
            select(RegulatoryChunk)
            .where(RegulatoryChunk.user_file_id == user_file_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
    )
    settings = get_current_search_settings(session, for_update=True)
    if inspect(settings).persistent:
        session.refresh(settings)


def list_pending_annex_publication_intents(
    session: Session,
    *,
    environment: str,
    tenant_id: str,
    database_identity: str,
    limit: int = 100,
) -> list[AnnexPublicationIntent]:
    """Task5 recovery re-delivers the current immutable generation after broker loss."""
    if not 1 <= limit <= 1000:
        raise ValueError("invalid publication recovery limit")
    return list(
        session.scalars(
            select(AnnexPublicationIntent)
            .join(AnnexChangeSet)
            .where(
                AnnexPublicationIntent.environment == environment,
                AnnexPublicationIntent.tenant_id == tenant_id,
                AnnexPublicationIntent.database_identity == database_identity,
                AnnexPublicationIntent.publication_generation
                == AnnexChangeSet.publication_generation,
                AnnexChangeSet.status.in_(("approving", "preparing", "publishing")),
            )
            .order_by(AnnexPublicationIntent.created_at, AnnexPublicationIntent.id)
            .limit(limit)
        )
    )


def legacy_text_annex_is_complete(
    session: Session,
    *,
    batch: AmendmentBatch,
    group: "AnnexInstructionGroup",
    reference_date: datetime.date,
) -> bool:
    import re

    from onyx.db.regulatory_annexes import load_legacy_annex_chunks
    from onyx.regulatory.amendments.models import AmendmentInstruction
    from onyx.regulatory.amendments.ranker import CandidateChunk
    from onyx.regulatory.amendments.structural_target import (
        appendix_replacement_attention_message,
        parse_amendment_structural_target,
    )

    if batch.source_package_id is not None or any(
        re.search(r"https?://", text) for text in group.instruction_texts
    ):
        return False
    try:
        file_id = resolve_annex_instruction_file(
            session,
            batch=batch,
            annex_label=group.annex_label,
            target_sources=group.target_sources,
            effective_date=reference_date,
        )
        rows = load_legacy_annex_chunks(
            session,
            document_set_id=batch.document_set_id,
            user_file_id=file_id,
            annex_label=group.annex_label,
            as_of_date=reference_date,
        )
    except ValueError:
        return False
    if len(rows) != 1:
        return False
    row = rows[0]
    if row.chunk_type == "image" or any(
        row.chunk_metadata.get(key)
        for key in ("image_file_id", "image_file_ids", "bound_to_regulatory_chunk_id")
    ):
        return False
    for text in group.instruction_texts:
        instruction = AmendmentInstruction(instruction_text=text)
        target = parse_amendment_structural_target(instruction)
        if target is None or target.appendix_label is None:
            return False
        candidate = CandidateChunk(
            chunk_id=row.id,
            user_file_id=str(file_id),
            text=row.text,
            metadata={**row.chunk_metadata, "appendix_label": target.appendix_label},
        )
        if appendix_replacement_attention_message(instruction, [candidate]) is not None:
            return False
    return True


def resume_unpublished_annex_review(
    session: Session,
    *,
    change_set_id: UUID,
    expected_review_sha256: str,
    environment: str,
) -> AnnexChangeSet:
    """Resume only existing frozen preparation; changed inputs require revalidation."""
    review = require_current_annex_review(
        session,
        change_set_id=change_set_id,
        expected_review_sha256=expected_review_sha256,
        environment=environment,
    )
    if review.publication_generation or review.status not in (
        "pending",
        "blocked",
        "rejected",
        "failed",
    ):
        raise ValueError("review state does not allow unpublished resume")
    batch = session.get(AmendmentBatch, review.batch_id)
    assert batch is not None
    draft = AnnexChangeDraft.model_validate(review.review_payload)
    validate_annex_review_scope(
        session, batch=batch, draft=draft, environment=environment
    )
    if (
        not draft.issues
        and draft.patch_plan is not None
        and draft.patch_plan.ready
        and draft.impact is not None
        and draft.impact.ready
    ):
        lock_annex_preparation_scope(session, draft.user_file_id)
        validate_prepared_annex_change(
            session, batch=batch, draft=draft, environment=environment
        )
        if review.status in ("rejected", "failed"):
            review.status = "pending"
            review.error_message = None
            review.decided_by = None
            review.decided_at = None
    session.commit()
    return review
