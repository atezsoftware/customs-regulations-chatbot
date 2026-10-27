"""Correct unpublished intake chunks and relabel only their changed file context."""

import datetime
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4, uuid5

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import regulatory_labeling as labeling
from onyx.db.document_intake_edit_budget import (
    prospective_intake_edit_rows,
    validate_intake_edit_budgets,
)
from onyx.db.document_intake_metadata import (
    invalidate_metadata_file_context,
    manual_edit_metadata,
)
from onyx.db.enums import UserFileStatus
from onyx.db.labeling_configuration import resolve_intake_labeling_provider_binding
from onyx.db.models import (
    DocumentSet,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryLabelingShard,
    RegulatorySourceAsset,
    User,
    UserFile,
)
from onyx.db.regulatory_annex_changes import capture_canonical_scope
from onyx.db.regulatory_canonical_revisions import retain_canonical_revisions
from onyx.db.regulatory_chunks import is_hierarchical_aggregate_chunk
from onyx.db.user_file import get_user_file_projection_repair
from onyx.document_index.publication_models import publication_digest
from onyx.regulatory.amendments.annexes.context_dependencies import (
    canonical_dependency_ids,
    rebuild_context_aggregates,
)
from onyx.regulatory.amendments.annexes.models import SourceLink
from onyx.regulatory.document_intake.models import IntakeSourceAsset
from onyx.regulatory.document_intake.text import (
    normalize_attachment_navigation,
    require_resolved_attachment_references,
)

_EDITABLE_FIELDS = frozenset(
    {
        "text",
        "heading_path",
        "chunk_type",
        "chunk_metadata",
        "validity_start_date",
        "validity_end_date",
    }
)
MAX_EDIT_TEXT_CHARS = 2_000_000
MAX_INTAKE_CHUNK_EDITS = 256
_TERMINAL_LABELING_STATUSES = frozenset(
    {"completed", "completed_with_errors", "failed", "cancelled"}
)


@dataclass(frozen=True)
class IntakeChunkEdit:
    document_id: UUID
    chunk_id: str
    updates: Mapping[str, Any]


@dataclass(frozen=True)
class _PreparedChunkEdit:
    chunk: RegulatoryChunk
    updates: dict[str, Any]


def edit_review_digest(session: Session, batch: RegulatoryDocumentIntake) -> str | None:
    """An edit receipt never grants publication of an incomplete label review."""
    if (
        batch.labeling_run_id is None
        or batch.status not in {"ready", "partial"}
        or (batch.status == "partial" and batch.stage != "labeling")
    ):
        return None
    run = session.get(RegulatoryLabelingRun, batch.labeling_run_id)
    if (
        run is None
        or run.status not in _TERMINAL_LABELING_STATUSES
        or (batch.status == "ready" and run.status != "completed")
        or labeling.get_active_run_id(session, batch.document_set_id) is not None
        or (
            run.lease_token is not None
            and run.lease_expires_at is not None
            and run.lease_expires_at > datetime.datetime.now(datetime.timezone.utc)
        )
    ):
        return None
    members = intake.intake_files(session, batch)
    if (
        not members
        or set(run.file_ids) != {str(member.user_file_id) for member in members}
        or any(member.approval_attempt_id is not None for member in members)
    ):
        return None
    for member in members:
        file = session.get(UserFile, member.user_file_id)
        if (
            file is None
            or file.status != UserFileStatus.CHUNKED
            or get_user_file_projection_repair(session, file.id)
        ):
            return None
    items = list(
        session.scalars(
            select(RegulatoryLabelingItem)
            .where(RegulatoryLabelingItem.run_id == run.id)
            .order_by(RegulatoryLabelingItem.id)
        )
    )
    if any(item.status == "submitted" for item in items) or session.scalar(
        select(RegulatoryLabelingShard.id)
        .where(
            RegulatoryLabelingShard.run_id == run.id,
            RegulatoryLabelingShard.status.in_(
                ("submitting", "reconcile_required", "submitted")
            ),
        )
        .limit(1)
    ):
        return None
    try:
        canonical_digest = intake.review_digest(session, batch)
    except intake.IntakeConflictError:
        return None
    if batch.status == "ready":
        return batch.review_sha256 if batch.review_sha256 == canonical_digest else None
    return publication_digest(
        {
            "purpose": "unpublished_intake_edit",
            "canonical_review": canonical_digest,
            "run_id": str(run.id),
            "status": run.status,
            "error": run.error,
            "items": [
                {
                    "id": str(item.id),
                    "chunk_id": item.regulatory_chunk_id,
                    "file_id": str(item.user_file_id),
                    "status": item.status,
                    "error": item.error,
                    "request_hash": item.request_hash,
                }
                for item in items
            ],
        }
    )


def _same_labeling_provenance(
    previous: RegulatoryLabelingRun, replacement: RegulatoryLabelingRun
) -> bool:
    return (
        previous.taxonomy_id == replacement.taxonomy_id
        and previous.model == replacement.model
        and previous.model_configuration_id == replacement.model_configuration_id
        and previous.provider_binding == replacement.provider_binding
    )


def _copy_completed_item(
    previous: RegulatoryLabelingItem, replacement: RegulatoryLabelingItem
) -> None:
    for field in (
        "canonical_text_sha256",
        "context_sha256",
        "text_snapshot",
        "source_snapshot",
        "context_snapshot",
        "request_hash",
        "request_payload",
        "labels",
        "assignments",
    ):
        setattr(replacement, field, deepcopy(getattr(previous, field)))
    # Provider shard ownership belongs to the immutable previous run.
    replacement.status, replacement.error = "completed", None


def reuse_completed_intake_labels(
    session: Session,
    *,
    previous: RegulatoryLabelingRun,
    replacement: RegulatoryLabelingRun,
) -> int:
    """Reuse exact current results on retry; stale or incomplete work stays pending."""
    if previous.id == replacement.id or not _same_labeling_provenance(
        previous, replacement
    ):
        return 0
    previous_items = list(
        session.scalars(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == previous.id,
                RegulatoryLabelingItem.status == "completed",
            )
        )
    )
    valid_ids = labeling.current_labeling_items_for_search(
        session, run=previous, items=previous_items
    )
    reusable = {
        item.regulatory_chunk_id: item
        for item in previous_items
        if item.regulatory_chunk_id in valid_ids
    }
    copied = 0
    for item in session.scalars(
        select(RegulatoryLabelingItem).where(
            RegulatoryLabelingItem.run_id == replacement.id,
            RegulatoryLabelingItem.status == "pending",
        )
    ):
        old = reusable.get(item.regulatory_chunk_id)
        if (
            old is None
            or old.user_file_id != item.user_file_id
            or old.text_snapshot != item.text_snapshot
            or any(
                old.source_snapshot.get(key) != value
                for key, value in item.source_snapshot.items()
            )
        ):
            continue
        _copy_completed_item(old, item)
        copied += 1
    replacement.completed_chunks += copied
    session.flush()
    return copied


def _validated_updates(
    chunk: RegulatoryChunk, updates: Mapping[str, Any]
) -> dict[str, Any]:
    if not updates or set(updates) - _EDITABLE_FIELDS:
        raise ValueError(
            "Supply only editable chunk text, headings, legal metadata, type or dates"
        )
    if "chunk_metadata" in updates and not isinstance(updates["chunk_metadata"], dict):
        raise ValueError("Chunk legal metadata must be a JSON object")
    text = updates.get("text", chunk.text)
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_EDIT_TEXT_CHARS:
        raise ValueError("Chunk text must contain between 1 and 2000000 characters")
    headings = updates.get("heading_path", chunk.heading_path)
    if (
        not isinstance(headings, list)
        or len(headings) > 64
        or any(
            not isinstance(heading, str) or len(heading) > 2000 for heading in headings
        )
    ):
        raise ValueError("Chunk headings must be a list of at most 64 short strings")
    kind = updates.get("chunk_type", chunk.chunk_type)
    if kind is not None and (
        not isinstance(kind, str)
        or not kind.strip()
        or len(kind) > 200
        or kind == "hierarchical_aggregate"
    ):
        raise ValueError("Chunk type must describe an ordinary canonical chunk")
    dates = [
        updates.get("validity_start_date", chunk.validity_start_date),
        updates.get("validity_end_date", chunk.validity_end_date),
    ]
    if any(value is not None and type(value) is not datetime.date for value in dates):
        raise ValueError("Validity boundaries must be dates or null")
    if dates[0] is not None and dates[1] is not None and dates[0] >= dates[1]:
        raise ValueError("Chunk validity end must be after its start")
    return {
        key: deepcopy(value)
        for key, value in updates.items()
        if getattr(chunk, key) != value
    }


def _copy_unchanged_label_results(
    session: Session,
    *,
    previous: RegulatoryLabelingRun,
    replacement: RegulatoryLabelingRun,
    changed_file_ids: set[UUID],
    allow_incomplete: bool = False,
) -> None:
    if not _same_labeling_provenance(previous, replacement):
        raise intake.IntakeConflictError(
            "The frozen labeling taxonomy or model changed; reload its review"
        )
    previous_items = list(
        session.scalars(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == previous.id,
                RegulatoryLabelingItem.user_file_id.not_in(changed_file_ids),
                RegulatoryLabelingItem.status == "completed",
            )
        )
    )
    valid_ids = labeling.current_labeling_items_for_search(
        session, run=previous, items=previous_items
    )
    reusable = {
        item.regulatory_chunk_id: item
        for item in previous_items
        if item.regulatory_chunk_id in valid_ids
    }
    for item in session.scalars(
        select(RegulatoryLabelingItem).where(
            RegulatoryLabelingItem.run_id == replacement.id,
            RegulatoryLabelingItem.user_file_id.not_in(changed_file_ids),
        )
    ):
        old = reusable.get(item.regulatory_chunk_id)
        if (
            old is None
            or old.user_file_id != item.user_file_id
            or old.text_snapshot != item.text_snapshot
            or any(
                old.source_snapshot.get(key) != value
                for key, value in item.source_snapshot.items()
            )
        ):
            if allow_incomplete:
                continue
            raise intake.IntakeConflictError(
                "An unchanged document no longer has current labels; reload its review"
            )
        _copy_completed_item(old, item)
        replacement.completed_chunks += 1


def edit_intake_chunk(
    session: Session,
    batch: RegulatoryDocumentIntake,
    *,
    document_id: UUID,
    chunk_id: str,
    expected_review_sha256: str,
    updates: Mapping[str, Any],
) -> UUID | None:
    """Preserve the single-chunk contract through the same atomic bulk operation."""
    return edit_intake_chunks(
        session,
        batch,
        expected_review_sha256=expected_review_sha256,
        edits=[IntakeChunkEdit(document_id, chunk_id, updates)],
    )


def edit_intake_chunks(
    session: Session,
    batch: RegulatoryDocumentIntake,
    *,
    expected_review_sha256: str,
    edits: Sequence[IntakeChunkEdit],
) -> UUID | None:
    """Save one review atomically; the caller commits before dispatching one run."""
    if not edits or len(edits) > MAX_INTAKE_CHUNK_EDITS:
        raise ValueError("Supply between 1 and 256 chunk edits")
    if len({edit.chunk_id for edit in edits}) != len(edits):
        raise ValueError("Each canonical chunk can be edited only once per review")
    with session.begin_nested():
        return _edit_intake_chunks(
            session,
            batch,
            expected_review_sha256=expected_review_sha256,
            edits=edits,
        )


def _edit_intake_chunks(
    session: Session,
    batch: RegulatoryDocumentIntake,
    *,
    expected_review_sha256: str,
    edits: Sequence[IntakeChunkEdit],
) -> UUID | None:
    session.refresh(batch, with_for_update=True)
    intake.refresh_intake(session, batch)
    session.get_one(DocumentSet, batch.document_set_id, with_for_update=True)
    previous = (
        session.get(
            RegulatoryLabelingRun,
            batch.labeling_run_id,
            with_for_update=True,
            populate_existing=True,
        )
        if batch.labeling_run_id
        else None
    )
    if edit_review_digest(session, batch) != expected_review_sha256:
        raise intake.IntakeConflictError(
            "The document edit review changed; reload before editing"
        )
    partial_review = batch.status == "partial"
    document_ids = {edit.document_id for edit in edits}
    members = list(
        session.scalars(
            select(RegulatoryDocumentIntakeFile)
            .where(
                RegulatoryDocumentIntakeFile.id.in_(document_ids),
                RegulatoryDocumentIntakeFile.intake_id == batch.id,
            )
            .order_by(RegulatoryDocumentIntakeFile.id)
            .with_for_update()
        )
    )
    if {member.id for member in members} != document_ids:
        raise ValueError("Intake document not found")
    member_by_id = {member.id: member for member in members}
    files = {
        member.user_file_id: session.get_one(
            UserFile, member.user_file_id, with_for_update=True
        )
        for member in sorted(members, key=lambda member: member.user_file_id)
    }
    for file in files.values():
        if (
            file.status != UserFileStatus.CHUNKED
            or file.user_id != batch.created_by
            or get_user_file_projection_repair(session, file.id)
        ):
            raise intake.IntakeConflictError(
                "Only unpublished imported files owned by this review can be edited"
            )
    rows = list(
        session.scalars(
            select(RegulatoryChunk)
            .where(RegulatoryChunk.user_file_id.in_(files))
            .order_by(
                RegulatoryChunk.user_file_id,
                RegulatoryChunk.position,
                RegulatoryChunk.id,
            )
            .with_for_update()
        )
    )
    by_id = {row.id: row for row in rows}
    rows_by_file: dict[UUID, list[RegulatoryChunk]] = {}
    for row in rows:
        rows_by_file.setdefault(row.user_file_id, []).append(row)
    prepared_by_file: dict[UUID, list[_PreparedChunkEdit]] = {}
    assets_by_package: dict[UUID, list[IntakeSourceAsset]] = {}
    for edit in edits:
        member = member_by_id[edit.document_id]
        chunk = by_id.get(edit.chunk_id)
        if chunk is None or chunk.user_file_id != member.user_file_id:
            raise ValueError("Chunk does not belong to this intake document")
        if (
            is_hierarchical_aggregate_chunk(chunk)
            or canonical_dependency_ids(chunk)
            or chunk.source != "indexed"
            or chunk.status != "active"
            or chunk.supersedes_chunk_id is not None
            or chunk.superseded_by_chunk_id is not None
        ):
            raise ValueError("Only unversioned canonical chunks can be edited directly")
        _validated_updates(chunk, edit.updates)
        normalized_updates = dict(edit.updates)
        if "text" in normalized_updates:
            asset = session.get_one(RegulatorySourceAsset, member.source_asset_id)
            links = [SourceLink.model_validate(value) for value in member.relationships]
            require_resolved_attachment_references(
                normalized_updates["text"], asset_sha256=asset.sha256, links=links
            )
            if asset.package_id not in assets_by_package:
                assets_by_package[asset.package_id] = [
                    IntakeSourceAsset.model_validate(value)
                    for value in session.scalars(
                        select(RegulatorySourceAsset).where(
                            RegulatorySourceAsset.package_id == asset.package_id
                        )
                    )
                ]
            normalized_updates["text"] = normalize_attachment_navigation(
                normalized_updates["text"],
                asset_sha256=asset.sha256,
                links=links,
                package_assets=assets_by_package[asset.package_id],
            )
        normalized_updates = manual_edit_metadata(chunk, normalized_updates)
        changes = (
            _validated_updates(chunk, normalized_updates) if normalized_updates else {}
        )
        if changes:
            prepared_by_file.setdefault(chunk.user_file_id, []).append(
                _PreparedChunkEdit(chunk, changes)
            )
    if not prepared_by_file:
        return None
    if previous is None or previous.status not in _TERMINAL_LABELING_STATUSES:
        raise intake.IntakeConflictError(
            "The completed labeling snapshot is unavailable"
        )
    validate_intake_edit_budgets(
        session,
        files,
        {
            file_id: prospective_intake_edit_rows(
                rows_by_file[file_id],
                {change.chunk.id: change.updates for change in changes},
            )
            for file_id, changes in prepared_by_file.items()
        },
    )
    user = session.get_one(User, batch.created_by)
    binding = resolve_intake_labeling_provider_binding(
        session, batch.labeling_model_configuration_id, user=user
    )
    # No canonical record changes until every submitted edit has been validated.
    for file_id in sorted(prepared_by_file):
        changes = prepared_by_file[file_id]
        file_rows = rows_by_file[file_id]
        retain_canonical_revisions(session, capture_canonical_scope(session, file_id))
        for change in changes:
            for key, value in change.updates.items():
                setattr(change.chunk, key, value)
            change.chunk.chunk_metadata = {
                **change.chunk.chunk_metadata,
                "intake_edit_revision": str(uuid4()),
            }
        if any(
            field in change.updates
            for change in changes
            for field in ("text", "heading_path", "chunk_type")
        ):
            invalidate_metadata_file_context(file_rows)
        rebuilt = rebuild_context_aggregates(
            file_rows, changed_ids=[change.chunk.id for change in changes]
        )
        for replacement in rebuilt:
            if replacement is not by_id[replacement.id]:
                by_id[replacement.id].text = replacement.text
        session.flush()
        retain_canonical_revisions(session, capture_canonical_scope(session, file_id))
    run, _created = labeling.create_labeling_run(
        session,
        document_set_id=batch.document_set_id,
        taxonomy=previous.taxonomy,
        model_configuration_id=batch.labeling_model_configuration_id,
        model=binding.model,
        provider_binding=binding.model_dump(mode="json"),
        requested_by_id=batch.created_by,
        idempotency_key=uuid5(previous.id, intake.review_digest(session, batch)),
        target_file_ids=[
            member.user_file_id for member in intake.intake_files(session, batch)
        ],
    )
    _copy_unchanged_label_results(
        session,
        previous=previous,
        replacement=run,
        changed_file_ids=set(prepared_by_file),
        allow_incomplete=partial_review,
    )
    batch.labeling_run_id = run.id
    batch.status, batch.stage, batch.error = "preparing", "labeling", None
    batch.review_sha256, batch.lease_token, batch.lease_expires_at = None, None, None
    for member in intake.intake_files(session, batch):
        member.status, member.error = "labeling", None
    session.flush()
    return run.id
