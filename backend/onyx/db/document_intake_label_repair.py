"""Read frozen requests only while an unpublished intake still owns its source."""

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from onyx.db import regulatory_labeling as repository
from onyx.db.document_intake_labeling import is_current_intake_labeling_run
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryLabelingShard,
)
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from onyx.regulatory.labeling.provider import (
    TaxonomyDefinition,
    build_labeling_request,
)


class IntakeLabelRepairSourceChangedError(repository.LabelingStateConflictError):
    """The original result can be rejected without requesting another inference."""


def load_current_intake_label_repair_request(
    session: Session,
    lease: repository.RunLease,
    shard_id: UUID,
    item_id: UUID,
) -> VertexBatchRequest:
    """Validate fresh authority without changing a request, result, or lease."""
    with session.no_autoflush:
        run = session.scalar(
            select(RegulatoryLabelingRun)
            .options(selectinload(RegulatoryLabelingRun.taxonomy))
            .where(
                RegulatoryLabelingRun.id == lease.run_id,
                RegulatoryLabelingRun.lease_generation == lease.generation,
                RegulatoryLabelingRun.lease_token == lease.token,
            )
            .execution_options(populate_existing=True)
            .with_for_update()
        )
        if (
            run is None
            or run.status != "running"
            or run.lease_expires_at is None
            or run.lease_expires_at <= datetime.now(timezone.utc)
        ):
            raise repository.LabelingStateConflictError(
                "The labeling repair lease is no longer owned"
            )
        if run.cancel_requested:
            raise repository.LabelingCancellationRequested(
                "The labeling run was cancelled before quote repair"
            )
        if run.provider_binding.get(
            "transport"
        ) != "vertex_direct_v1" or not is_current_intake_labeling_run(session, run):
            raise repository.LabelingStateConflictError(
                "Quote repair requires the current unpublished intake"
            )
        shard = session.scalar(
            select(RegulatoryLabelingShard)
            .where(
                RegulatoryLabelingShard.id == shard_id,
                RegulatoryLabelingShard.run_id == run.id,
            )
            .execution_options(populate_existing=True)
        )
        item = session.scalar(
            select(RegulatoryLabelingItem)
            .where(
                RegulatoryLabelingItem.id == item_id,
                RegulatoryLabelingItem.run_id == run.id,
                RegulatoryLabelingItem.shard_id == shard_id,
            )
            .execution_options(populate_existing=True)
        )
        if (
            shard is None
            or shard.status != "submitted"
            or item is None
            or item.status != "submitted"
            or str(item.id) not in shard.item_ids
        ):
            raise repository.LabelingStateConflictError(
                "The labeling repair item is outside the submitted shard"
            )
        try:
            if (
                item.request_payload is None
                or item.context_snapshot is None
                or not item.context_sha256
                or item.canonical_text_sha256 is None
                or item.canonical_text_sha256 != context_hash(item.text_snapshot)
            ):
                raise ValueError("Incomplete frozen labeling request")
            request = VertexBatchRequest.model_validate(item.request_payload)
            rebuilt = build_labeling_request(
                chunk_id=item.regulatory_chunk_id,
                text=item.text_snapshot,
                context=item.context_snapshot,
                taxonomy=TaxonomyDefinition.model_validate(run.taxonomy.definition),
                source_hash=item.canonical_text_sha256,
            )
            if request.request_hash != item.request_hash or request != rebuilt:
                raise ValueError("The frozen labeling request changed")
            lower, upper = repository._snapshot_context_window(item.source_snapshot)
            required_ids = repository._snapshot_string_list(
                item.source_snapshot, "context_member_ids"
            )
        except ValueError as error:
            raise IntakeLabelRepairSourceChangedError(
                "The frozen labeling repair request is invalid"
            ) from error

        snapshot_items = {
            frozen.regulatory_chunk_id: frozen
            for frozen in session.scalars(
                select(RegulatoryLabelingItem)
                .where(
                    RegulatoryLabelingItem.run_id == run.id,
                    RegulatoryLabelingItem.regulatory_chunk_id.in_(required_ids),
                )
                .execution_options(populate_existing=True)
            )
        }
        rows, boundaries, complete = repository._current_file_context_candidates(
            session,
            document_set_id=run.document_set_id,
            user_file_id=item.user_file_id,
            lower=lower,
            upper=upper,
        )
        # Candidate membership is queried fresh; refresh cached row values in one read.
        rows = list(
            session.scalars(
                select(RegulatoryChunk)
                .where(RegulatoryChunk.id.in_([row.id for row in rows]))
                .execution_options(populate_existing=True)
            )
        )
        rows.sort(key=lambda row: (row.position, row.id))
        advisories = repository._latest_context_advisories(
            session, [item.regulatory_chunk_id]
        )
        if not repository._item_snapshot_is_current(
            item,
            snapshot_items=snapshot_items,
            current_rows={row.id: row for row in rows},
            current_file_rows={item.user_file_id: rows},
            current_boundary_ids={item.user_file_id: boundaries},
            current_windows_complete={item.user_file_id: complete},
            advisory_hashes={
                key: context_hash(value) for key, value in advisories.items()
            },
        ):
            raise IntakeLabelRepairSourceChangedError(
                "The canonical context changed before label quote repair"
            )
        return request
