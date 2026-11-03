"""Current owned sources and durable authority for duplicate-label selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from uuid import UUID

from pydantic import JsonValue
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.document_intake_source_binding import (
    require_owned_intake_source,
    require_owned_v4_intake_source,
)
from onyx.db.labeling_configuration import LabelingProviderBinding
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryFilePublication,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryLabelingShard,
    RegulatoryLabelTaxonomy,
    UserFile,
)
from onyx.db.regulatory_labeling_duplicate_evidence import validate_duplicate_ack_header
from onyx.file_processing.original_ingestion import OriginalIngestionReceipt
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from onyx.regulatory.labeling.intake_duplicate_label_repair import (
    DUPLICATE_REPAIR_KEY,
    DuplicateLabelRepairAudit,
    DuplicateSourceBinding,
    build_duplicate_label_repair,
    build_duplicate_repair_audit,
)
from onyx.regulatory.labeling.provider import TaxonomyDefinition, build_labeling_request

if TYPE_CHECKING:
    from onyx.db.regulatory_labeling import RunLease

_SNAPSHOT = (
    "position",
    "heading_path",
    "chunk_type",
    "chunk_metadata",
    "validity_start_date",
    "validity_end_date",
)
_OTHER_REPAIRS = {
    "intake_abstention_repair",
    "intake_source_only_self_repeal",
    "intake_source_only_article_lead_in",
    "intake_source_only_chapter_caption",
    "intake_source_only_reviewed_fragment",
}


def require_current_duplicate_source(
    session: Session, *, item: RegulatoryLabelingItem, run: RegulatoryLabelingRun
) -> DuplicateSourceBinding:
    if (
        item.run_id != run.id
        or run.document_set_id is None
        or not isinstance(run.file_ids, list)
        or not all(isinstance(file_id, str) for file_id in run.file_ids)
        or len(run.file_ids) != len(set(run.file_ids))
        or str(item.user_file_id) not in run.file_ids
        or _OTHER_REPAIRS.intersection(item.source_snapshot)
    ):
        raise ValueError("Duplicate-label selection differs from its owned run")
    members = item.source_snapshot.get("context_member_ids")
    if (
        not isinstance(members, list)
        or not 1 <= len(members) <= 40
        or not all(isinstance(value, str) for value in members)
        or len(members) != len(set(members))
        or item.regulatory_chunk_id not in members
    ):
        raise ValueError("Duplicate-label selection has incomplete frozen context")
    rows = session.execute(
        select(RegulatoryChunk, RegulatoryLabelingItem)
        .join(
            RegulatoryLabelingItem,
            (RegulatoryLabelingItem.regulatory_chunk_id == RegulatoryChunk.id)
            & (RegulatoryLabelingItem.user_file_id == RegulatoryChunk.user_file_id),
        )
        .where(
            RegulatoryLabelingItem.run_id == run.id,
            RegulatoryChunk.user_file_id == item.user_file_id,
            RegulatoryChunk.id.in_(members),
        )
        .execution_options(populate_existing=True)
    ).all()
    if len(rows) != len(members) or {row.id for row, _ in rows} != set(members):
        raise ValueError("Duplicate-label selection lost frozen current members")
    target: RegulatoryChunk | None = None
    for row, frozen in rows:
        current = {
            "position": row.position,
            "heading_path": row.heading_path,
            "chunk_type": row.chunk_type,
            "chunk_metadata": row.chunk_metadata,
            "validity_start_date": row.validity_start_date.isoformat()
            if row.validity_start_date
            else None,
            "validity_end_date": row.validity_end_date.isoformat()
            if row.validity_end_date
            else None,
        }
        audit = row.chunk_metadata.get("intake_legal_metadata")
        if (
            row.status != "active"
            or row.superseded_by_chunk_id is not None
            or row.chunk_metadata.get("chunk_variant") != "atomic"
            or row.chunk_metadata.get("intake_chunking_version")
            not in ("2", "3", "4", "5")
            or not isinstance(audit, Mapping)
            or audit.get("input_stale") is not False
            or row.text != frozen.text_snapshot
            or context_hash(row.text) != frozen.canonical_text_sha256
            or any(frozen.source_snapshot.get(key) != current[key] for key in _SNAPSHOT)
        ):
            raise ValueError("The duplicate-label frozen source or metadata changed")
        if row.id == item.regulatory_chunk_id:
            target = row
    if (
        target is None
        or target.text != item.text_snapshot
        or context_hash(target.text) != item.canonical_text_sha256
    ):
        raise ValueError("The duplicate-label target changed")
    if target.chunk_metadata.get("intake_chunking_version") in ("4", "5"):
        asset = require_owned_v4_intake_source(
            session,
            user_file_id=item.user_file_id,
            metadata=target.chunk_metadata,
            chunk_type=target.chunk_type,
            document_set_id=run.document_set_id,
        )
    else:
        asset = require_owned_intake_source(
            session, user_file_id=item.user_file_id, document_set_id=run.document_set_id
        )
    publication = session.get_one(
        RegulatoryFilePublication, item.user_file_id, populate_existing=True
    )
    original = OriginalIngestionReceipt.model_validate(
        publication.original_ingestion_receipt
    )
    member = session.scalar(
        select(RegulatoryDocumentIntakeFile)
        .where(RegulatoryDocumentIntakeFile.user_file_id == item.user_file_id)
        .execution_options(populate_existing=True)
    )
    batch = (
        session.get(RegulatoryDocumentIntake, member.intake_id, populate_existing=True)
        if member is not None
        else None
    )
    file = session.get(UserFile, item.user_file_id, populate_existing=True)
    if (
        member is None
        or batch is None
        or file is None
        or file.user_id != batch.created_by
        or batch.created_by != run.requested_by_id
        or batch.document_set_id != run.document_set_id
    ):
        raise ValueError("The duplicate-label intake owner changed")
    return DuplicateSourceBinding(
        document_set_id=run.document_set_id,
        user_file_id=str(item.user_file_id),
        source_asset_id=str(asset.id),
        intake_id=str(batch.id),
        package_id=str(asset.package_id),
        owner_id=str(batch.created_by),
        source_file_id=asset.file_id,
        original_file_id=original.file_id,
        raw_sha256=asset.sha256,
        frozen_context_sources_sha256=context_hash(
            {
                "members": members,
                "sources": [
                    {
                        "chunk_id": row.id,
                        "file_id": str(row.user_file_id),
                        "text_sha256": frozen.canonical_text_sha256,
                        "snapshot": {
                            key: value
                            for key, value in frozen.source_snapshot.items()
                            if key not in _OTHER_REPAIRS | {DUPLICATE_REPAIR_KEY}
                        },
                    }
                    for row, frozen in sorted(
                        rows, key=lambda pair: (pair[0].position, pair[0].id)
                    )
                ],
            }
        ),
        byte_count=asset.byte_count,
    )


def validate_item_duplicate_repair_audit(
    session: Session,
    *,
    item: RegulatoryLabelingItem,
    run: RegulatoryLabelingRun,
    audit: DuplicateLabelRepairAudit | Mapping[str, object] | None = None,
    labels: Sequence[str] | None = None,
    assignments: Sequence[Mapping[str, object]] | None = None,
) -> DuplicateLabelRepairAudit | None:
    present = DUPLICATE_REPAIR_KEY in item.source_snapshot
    if audit is None and not present:
        return None
    if (labels is None) != (assignments is None):
        raise ValueError("Duplicate-label prospective assignments are incomplete")
    stored = DuplicateLabelRepairAudit.model_validate(
        audit if audit is not None else item.source_snapshot[DUPLICATE_REPAIR_KEY]
    )
    if (
        audit is not None
        and present
        and item.source_snapshot[DUPLICATE_REPAIR_KEY] != stored.model_dump(mode="json")
    ):
        raise ValueError("A different duplicate-label selection attempt is recorded")
    source = require_current_duplicate_source(session, item=item, run=run)
    taxonomy_row = session.get_one(
        RegulatoryLabelTaxonomy, run.taxonomy_id, populate_existing=True
    )
    taxonomy = TaxonomyDefinition.model_validate(taxonomy_row.definition)
    binding = LabelingProviderBinding.model_validate(run.provider_binding)
    if (
        taxonomy.version_hash != taxonomy_row.version_hash
        or binding.transport != "vertex_direct_v1"
        or binding.model != run.model
        or not item.request_payload
        or item.context_snapshot is None
        or not item.context_sha256
        or stored.source_binding != source
    ):
        raise ValueError(
            "The duplicate-label source, model or taxonomy authority changed"
        )
    base = VertexBatchRequest.model_validate(item.request_payload)
    expected = build_labeling_request(
        chunk_id=item.regulatory_chunk_id,
        text=item.text_snapshot,
        context=item.context_snapshot,
        taxonomy=taxonomy,
        source_hash=item.canonical_text_sha256 or "",
    )
    if base != expected or base.request_hash != item.request_hash:
        raise ValueError("The duplicate-label frozen base request changed")
    for evidence in (stored.base_ack, stored.repair_ack):
        if evidence.binding_fingerprint != binding.fingerprint:
            raise ValueError("The duplicate-label cached provider identity changed")
        validate_duplicate_ack_header(session, evidence)
    repair = build_duplicate_label_repair(
        base_request=base,
        rejected_response=stored.rejected_response,
        text=item.text_snapshot,
        chunk_id=item.regulatory_chunk_id,
        source_hash=item.canonical_text_sha256 or "",
        taxonomy=taxonomy,
        provider_binding=run.provider_binding,
    )
    rebuilt = build_duplicate_repair_audit(
        repair=repair,
        selection_response=stored.selection_response,
        base_ack=stored.base_ack,
        repair_ack=stored.repair_ack,
        source_binding=source,
        text=item.text_snapshot,
        taxonomy=taxonomy,
    )
    expected_labels = [entry.label_id for entry in rebuilt.outcome.labels]
    expected_assignments = [
        entry.model_dump(mode="json") for entry in rebuilt.outcome.labels
    ]
    if (
        rebuilt != stored
        or list(item.labels if labels is None else labels) != expected_labels
        or list(item.assignments if assignments is None else assignments)
        != expected_assignments
    ):
        raise ValueError("The duplicate-label receipt or selected assignments changed")
    return stored


@dataclass(frozen=True)
class DuplicateRepairAuthority:
    base_request: VertexBatchRequest
    source_binding: DuplicateSourceBinding
    provider_binding: dict[str, JsonValue]
    binding_fingerprint: str
    base_submission_key: str


def load_current_duplicate_repair_authority(
    session: Session, *, lease: RunLease, shard_id: UUID, item_id: UUID
) -> DuplicateRepairAuthority:
    from onyx.db.document_intake_label_repair import (
        load_current_intake_label_repair_request,
    )

    base = load_current_intake_label_repair_request(session, lease, shard_id, item_id)
    item = session.get_one(RegulatoryLabelingItem, item_id, populate_existing=True)
    run = session.get_one(RegulatoryLabelingRun, lease.run_id, populate_existing=True)
    shard = session.get_one(RegulatoryLabelingShard, shard_id, populate_existing=True)
    if DUPLICATE_REPAIR_KEY in item.source_snapshot:
        raise ValueError("A duplicate-label selection attempt is already recorded")
    binding = LabelingProviderBinding.model_validate(run.provider_binding)
    if binding.transport != "vertex_direct_v1" or binding.model != run.model:
        raise ValueError("The current duplicate-label provider identity changed")
    return DuplicateRepairAuthority(
        base_request=base,
        source_binding=require_current_duplicate_source(session, item=item, run=run),
        provider_binding=cast(dict[str, JsonValue], dict(run.provider_binding)),
        binding_fingerprint=binding.fingerprint,
        base_submission_key=shard.submission_key,
    )
