"""Short durable job snapshots under the shared file publication authority."""

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from datetime import date

    from onyx.db.regulatory_indexing_jobs import RegulatoryIndexingPreparedItem
    from onyx.regulatory.amendments.annexes.models import AnnexTemporalProjection
    from onyx.regulatory.writer_publication_models import WriterPublicationManifest

from sqlalchemy.orm import Session

from onyx.db.engine.sql_engine import get_session_with_tenant
from onyx.db.enums import RegulatoryIndexingStage, UserFileStatus
from onyx.db.models import RegulatoryIndexingJob, SearchSettings, UserFile
from onyx.db.regulatory_indexing_jobs import (
    RegulatoryIndexingRuntime,
    get_regulatory_indexing_runtime,
)
from onyx.db.regulatory_publication import PublicationStore
from onyx.document_index.publication_models import FileOwnership, publication_digest
from onyx.regulatory.amendments.annexes.publication_representations import _snapshot


def durable_publication_input_digest(runtime: RegulatoryIndexingRuntime) -> str:
    settings = runtime.search_settings
    if settings is None:
        raise ValueError("durable publication configured index disappeared")
    return publication_digest(
        {
            "configured_index": {
                "id": settings.id,
                "status": settings.status.value,
                "index_name": settings.index_name,
                "provider": settings.provider_type.value
                if settings.provider_type
                else None,
                "model": settings.model_name,
                "model_dimension": settings.model_dim,
                "reduced_dimension": settings.reduced_dimension,
                "normalize": settings.normalize,
                "query_prefix": settings.query_prefix,
                "passage_prefix": settings.passage_prefix,
                "endpoint_sha256": publication_digest(settings.api_url),
                "api_version": settings.api_version,
                "deployment_name": settings.deployment_name,
            },
            "job_id": str(runtime.job.id),
            "user_file_id": str(runtime.job.user_file_id),
            "configuration": runtime.job.config_snapshot,
            "canonical": [
                _snapshot(row).model_dump(mode="json")
                for row in runtime.regulatory_chunks
            ],
            "items": [
                {
                    "id": str(item.id),
                    "canonical": item.regulatory_chunk_id,
                    "projection_id": str(item.projection_id)
                    if item.projection_id
                    else None,
                    "ordinal": item.projection_ordinal,
                    "effective_start": item.effective_start.isoformat()
                    if item.effective_start
                    else None,
                    "effective_end": item.effective_end.isoformat()
                    if item.effective_end
                    else None,
                    "input": item.projection_input,
                    "request_hash": item.request_hash,
                    "context": item.context,
                    "status": item.status,
                    "vector": item.vector,
                }
                for item in sorted(
                    runtime.indexing_items, key=lambda item: str(item.id)
                )
            ],
        }
    )


def lock_owned_durable_runtime(
    session: Session,
    owner: FileOwnership,
    *,
    job_id: UUID,
    expected_stage: RegulatoryIndexingStage,
    expected_generation: int,
) -> RegulatoryIndexingRuntime:
    PublicationStore(owner.scope).lock_owned_snapshot(session, owner)
    file = session.get(UserFile, owner.user_file_id, with_for_update=True)
    job = session.get(RegulatoryIndexingJob, job_id, with_for_update=True)
    if (
        file is None
        or file.status in {UserFileStatus.CANCELED, UserFileStatus.DELETING}
        or job is None
        or (
            job.user_file_id != owner.user_file_id
            or job.status != "RUNNING"
            or job.stage != expected_stage.value
            or job.lease_generation != expected_generation
        )
    ):
        raise ValueError("durable publication job lease is no longer current")
    runtime = get_regulatory_indexing_runtime(session, job_id)
    if runtime is None or runtime.search_settings is None:
        raise ValueError("durable publication configured index disappeared")
    # Eagerly resolve the configured provider before detaching the short snapshot.
    _ = runtime.search_settings.cloud_provider
    return runtime


def load_owned_durable_runtime(
    owner: FileOwnership,
    *,
    job_id: UUID,
    expected_stage: RegulatoryIndexingStage,
    expected_generation: int,
) -> RegulatoryIndexingRuntime:
    with get_session_with_tenant(tenant_id=owner.scope.tenant_id) as session:
        runtime = lock_owned_durable_runtime(
            session,
            owner,
            job_id=job_id,
            expected_stage=expected_stage,
            expected_generation=expected_generation,
        )
        session.commit()
        session.expunge_all()
        return runtime


def lock_owned_cancellation(
    session: Session,
    owner: FileOwnership,
    *,
    job_id: UUID,
    expected_generation: int,
) -> RegulatoryIndexingJob:
    PublicationStore(owner.scope).lock_owned_snapshot(session, owner)
    file = session.get(UserFile, owner.user_file_id, with_for_update=True)
    job = session.get(RegulatoryIndexingJob, job_id, with_for_update=True)
    if (
        file is None
        or job is None
        or (
            job.user_file_id != owner.user_file_id
            or job.status != "CANCELLING"
            or job.lease_generation != expected_generation
            or job.cancellation_phase not in {"INDEX_DELETE", "FINALIZE"}
        )
    ):
        raise ValueError("cancellation publication lease is no longer current")
    return job


@dataclass(frozen=True)
class CancellationPublicationInputs:
    pending: "WriterPublicationManifest | None"
    canonical_before_sha256: str
    bindings: list["AnnexTemporalProjection"]
    canonical_revisions: dict[UUID, UUID]
    settings: list[SearchSettings]
    index_state_sha256: str | None


def prepare_cancellation_manifest(
    owner: FileOwnership,
    *,
    job_id: UUID,
    expected_generation: int,
) -> CancellationPublicationInputs:
    from sqlalchemy import select

    from onyx.db.models import RegulatoryFilePublication, RegulatoryTemporalProjection
    from onyx.db.regulatory_annex_publication import load_file_temporal_bindings
    from onyx.db.regulatory_writer_publication import (
        canonical_scope_digest,
        writer_index_state_digest,
    )
    from onyx.db.search_settings import get_active_search_settings_list
    from onyx.regulatory.writer_publication_models import WriterPublicationManifest

    with get_session_with_tenant(tenant_id=owner.scope.tenant_id) as session:
        job = lock_owned_cancellation(
            session, owner, job_id=job_id, expected_generation=expected_generation
        )
        from onyx.db.regulatory_publication import archive_canonical_revisions

        archive_canonical_revisions(session, owner)
        state = session.get(RegulatoryFilePublication, owner.user_file_id)
        assert state is not None
        pending = None
        if state.writer_manifest is not None:
            if (
                publication_digest(state.writer_manifest)
                != state.writer_manifest_sha256
            ):
                raise ValueError("cancellation pending writer evidence changed")
            pending = WriterPublicationManifest.model_validate(state.writer_manifest)
            if pending.kind == "cancellation" and pending.cancellation_job_id == job_id:
                return CancellationPublicationInputs(
                    pending,
                    pending.canonical_before_sha256,
                    pending.bindings,
                    pending.canonical_revisions,
                    [],
                    pending.index_state_sha256,
                )
            if pending.kind != "durable" or pending.durable_job_id != job_id:
                raise ValueError("cancellation must wait for the other owned writer")
        elif state.gate_closed:
            raise ValueError("cancellation must wait for the pending annex publication")
        bindings = load_file_temporal_bindings(session, owner.user_file_id)
        revisions = {
            row.id: row.canonical_revision_id
            for row in session.scalars(
                select(RegulatoryTemporalProjection).where(
                    RegulatoryTemporalProjection.id.in_(
                        [binding.id for binding in bindings]
                    )
                )
            )
        }
        if any(value is None for value in revisions.values()):
            raise ValueError(
                "cancellation requires retained canonical revision authority"
            )
        settings = get_active_search_settings_list(session)
        if job.search_settings_id not in {setting.id for setting in settings}:
            target = session.get(SearchSettings, job.search_settings_id)
            if target is not None:
                settings.append(target)
        for setting in settings:
            _ = setting.cloud_provider
        revision_ids: dict[UUID, UUID] = {
            key: value for key, value in revisions.items() if value is not None
        }
        result = CancellationPublicationInputs(
            pending,
            canonical_scope_digest(session, owner.user_file_id),
            bindings,
            revision_ids,
            settings,
            writer_index_state_digest(session, [item.id for item in settings]),
        )
        session.commit()
        session.expunge_all()
        return result


def durable_context_reference_dates(owner: FileOwnership) -> dict[str, "date"]:
    from sqlalchemy import select

    from onyx.db.models import RegulatoryContextSnapshot
    from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
    from onyx.regulatory.amendments.annexes.models import ContextSourceSnapshot

    with get_session_with_tenant(tenant_id=owner.scope.tenant_id) as session:
        PublicationStore(owner.scope).lock_owned_snapshot(session, owner)
        references = {}
        for row in session.scalars(
            select(RegulatoryContextSnapshot).where(
                RegulatoryContextSnapshot.user_file_id == owner.user_file_id
            )
        ):
            snapshot = ContextSourceSnapshot.model_validate(row.payload)
            if snapshot.sha256 != row.sha256 or snapshot.sha256 != context_hash(
                [
                    str(owner.user_file_id),
                    snapshot.text,
                    [span.model_dump() for span in snapshot.ordered_ranges],
                    snapshot.reference_date,
                ]
            ):
                raise ValueError("durable context source snapshot changed")
            references[row.sha256] = snapshot.reference_date
        return references


def repair_durable_item_checkpoints(
    owner: FileOwnership,
    *,
    job_id: UUID,
    expected_generation: int,
    stage: RegulatoryIndexingStage,
    expected_input_sha256: str,
    prepared: list["RegulatoryIndexingPreparedItem"],
) -> bool:
    from datetime import datetime, timezone
    from uuid import uuid4

    from onyx.db.models import RegulatoryIndexingItem
    from onyx.db.regulatory_context_projections import persist_context_view
    from onyx.regulatory.amendments.annexes.models import PreparedContextView
    from onyx.regulatory.indexing_jobs.embedding_receipts import (
        has_proven_vector,
        item_embedding_receipt,
    )
    from onyx.regulatory.indexing_jobs.models import (
        IndexingPublicationIndeterminateError,
    )

    with get_session_with_tenant(tenant_id=owner.scope.tenant_id) as session:
        runtime = lock_owned_durable_runtime(
            session,
            owner,
            job_id=job_id,
            expected_stage=stage,
            expected_generation=expected_generation,
        )
        if durable_publication_input_digest(runtime) != expected_input_sha256:
            raise IndexingPublicationIndeterminateError()
        job = runtime.job
        if (
            job.openrouter_submission_state != "NONE"
            or job.vertex_submission_state
            not in {"NONE", "SUBMITTED", "RETRY_CLEANUP_REQUIRED"}
        ):
            raise IndexingPublicationIndeterminateError()
        existing = {
            item.projection_id: item
            for item in runtime.indexing_items
            if item.projection_id is not None
        }
        legacy = {
            item.regulatory_chunk_id: item
            for item in runtime.indexing_items
            if item.projection_id is None
        }
        if set(existing) - {item.projection_id for item in prepared}:
            raise IndexingPublicationIndeterminateError()
        missing_context = False
        now = datetime.now(timezone.utc)
        all_items = []
        # Prefer the original proven request when a legacy canonical has several windows.
        prepared = sorted(
            prepared,
            key=lambda item: (
                not (
                    item.regulatory_chunk_id in legacy
                    and (legacy[item.regulatory_chunk_id].context or {}).get(
                        "context_input"
                    )
                    == item.context_input
                )
            ),
        )
        snapshots = {
            item.source_snapshot.sha256: item.source_snapshot
            for item in prepared
            if item.source_snapshot
        }
        persist_context_view(
            session,
            user_file_id=owner.user_file_id,
            view=PreparedContextView(snapshots=list(snapshots.values())),
        )
        for desired in prepared:
            if desired.projection_id is None or desired.projection_input is None:
                raise ValueError("owned durable recovery requires dated frozen inputs")
            item = existing.get(desired.projection_id)
            if item is None:
                item = legacy.pop(desired.regulatory_chunk_id, None)
            if item is None:
                item = RegulatoryIndexingItem(
                    id=uuid4(),
                    job_id=job_id,
                    regulatory_chunk_id=desired.regulatory_chunk_id,
                    request_hash=desired.request_hash,
                    status="PENDING",
                )
                session.add(item)
            context = dict(item.context or {})
            proven = (
                context.get("context_input") == desired.context_input
                and item.request_hash == desired.request_hash
                and (desired.skip_context or bool(context.get("contextual_text")))
            )
            if not proven:
                audit = context.get("replaced_context_checkpoint", context)
                context = {
                    "context_input": desired.context_input,
                    "replaced_context_checkpoint": audit,
                }
                item.vector = None
                item.status = "SKIPPED" if desired.skip_context else "PENDING"
                item.context_attempt_count = 0
                item.embedding_attempt_count = 0
            if item.status == "PENDING":
                missing_context = True
            item.context = context
            item.request_hash = desired.request_hash
            item.projection_id, item.projection_ordinal = (
                desired.projection_id,
                desired.projection_ordinal,
            )
            item.effective_start, item.effective_end = (
                desired.effective_start,
                desired.effective_end,
            )
            item.projection_input = (
                desired.projection_input.model_dump(mode="json")
                if desired.projection_input
                else None
            )
            item.updated_at = now
            all_items.append(item)
        if legacy:
            raise IndexingPublicationIndeterminateError()
        unproven_vectors = any(
            (receipt := item_embedding_receipt(item)) is None
            or not has_proven_vector(item, receipt)
            for item in all_items
        )
        next_stage = None
        if missing_context:
            if job.remote_vertex_job_name:
                job.vertex_submission_state = "RETRY_CLEANUP_REQUIRED"
                next_stage = "CONTEXT_APPLY"
            else:
                job.vertex_submission_state = "NONE"
                next_stage = "CONTEXT_SUBMIT"
        elif unproven_vectors and stage is not RegulatoryIndexingStage.EMBEDDING:
            next_stage = "EMBEDDING"
        elif stage in {RegulatoryIndexingStage.VERIFY, RegulatoryIndexingStage.PUBLISH}:
            next_stage = "INDEX_WRITE"
        if next_stage:
            job.stage, job.status, job.attempt_count = next_stage, "QUEUED", 0
            job.next_retry_at, job.error_code, job.error_message = None, None, None
            job.heartbeat_at = job.updated_at = now
        session.commit()
        return next_stage is not None
