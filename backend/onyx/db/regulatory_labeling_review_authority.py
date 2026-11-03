"""Bind verified intake reader/repair evidence into review and approval identity."""

import json
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from onyx.db.models import (
    RegulatoryDerivedLabelProjection,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryLabelTaxonomy,
)
from onyx.db.regulatory_labeling_readers import (
    STRUCTURAL_READER_KEY,
    validated_reader_proof,
)
from onyx.db.regulatory_labeling_self_repeal import validate_item_source_only_receipt
from onyx.db.regulatory_labeling_semantic_repair import (
    SEMANTIC_REPAIR_KEY,
    validate_item_semantic_repair_audit,
)
from onyx.document_index.publication_models import publication_digest
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.amendments.annexes.models import AnnexCanonicalSnapshot
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from onyx.regulatory.labeling.intake_abstention_repair import AbstentionRepairAudit
from onyx.regulatory.labeling.provider import (
    TaxonomyDefinition,
    build_labeling_request,
    validate_labeling_response,
)
from onyx.regulatory.labeling.self_repeal import (
    SOURCE_ONLY_KEY,
    SourceOnlySelfRepealReceipt,
)


def _receipt_state(
    item: RegulatoryLabelingItem, current: frozenset[str]
) -> dict[str, object]:
    return {
        "chunk_id": item.regulatory_chunk_id,
        "status": item.status,
        "current": item.regulatory_chunk_id in current,
        "request_hash": item.request_hash,
        "canonical_text_sha256": item.canonical_text_sha256,
        "context_sha256": item.context_sha256,
        "context_snapshot_sha256": publication_digest(item.context_snapshot),
        "request_payload_sha256": publication_digest(item.request_payload),
        "source_snapshot_sha256": publication_digest(item.source_snapshot),
        "labels_sha256": publication_digest(item.labels),
        "assignments_sha256": publication_digest(item.assignments),
        "error_sha256": publication_digest(item.error),
    }


def _validate_completed_leaf(
    item: RegulatoryLabelingItem, taxonomy: TaxonomyDefinition
) -> None:
    if item.status != "completed" or SOURCE_ONLY_KEY in item.source_snapshot:
        return
    if (
        not item.request_payload
        or item.context_snapshot is None
        or item.canonical_text_sha256 != context_hash(item.text_snapshot)
    ):
        raise ValueError("A native reader's accepted child receipt is incomplete")
    request = VertexBatchRequest.model_validate(item.request_payload)
    expected = build_labeling_request(
        chunk_id=item.regulatory_chunk_id,
        text=item.text_snapshot,
        context=item.context_snapshot,
        taxonomy=taxonomy,
        source_hash=context_hash(item.text_snapshot),
    )
    if request != expected or item.request_hash != request.request_hash:
        raise ValueError("A native reader's accepted child request changed")
    outcome = validate_labeling_response(
        json.dumps({"labels": item.assignments, "abstained": False}),
        text=item.text_snapshot,
        taxonomy=taxonomy,
    )
    if item.labels != [assignment.label_id for assignment in outcome.labels]:
        raise ValueError("A native reader's accepted child assignments changed")


def labeling_review_authority(
    session: Session, run_id: UUID
) -> dict[str, object] | None:
    """Legacy receipts keep their bytes; new evidence is always validated first."""
    repairs = list(
        session.scalars(
            select(RegulatoryLabelingItem)
            .where(
                RegulatoryLabelingItem.run_id == run_id,
                or_(
                    RegulatoryLabelingItem.source_snapshot.op("?")(SEMANTIC_REPAIR_KEY),
                    RegulatoryLabelingItem.source_snapshot.op("?")(SOURCE_ONLY_KEY),
                ),
            )
            .order_by(RegulatoryLabelingItem.regulatory_chunk_id)
            .execution_options(populate_existing=True)
        )
    )
    readers = list(
        session.scalars(
            select(RegulatoryDerivedLabelProjection)
            .where(
                RegulatoryDerivedLabelProjection.run_id == run_id,
                RegulatoryDerivedLabelProjection.source_snapshot.op("?")(
                    STRUCTURAL_READER_KEY
                ),
            )
            .order_by(RegulatoryDerivedLabelProjection.regulatory_chunk_id)
            .execution_options(populate_existing=True)
        )
    )
    if not repairs and not readers:
        return None
    run = session.get(RegulatoryLabelingRun, run_id, populate_existing=True)
    if run is None:
        raise ValueError("The intake labeling evidence has no frozen run")
    from onyx.db.regulatory_labeling import (
        current_derived_label_sources_for_search,
        current_labeling_items_for_search,
        derived_label_provenance,
    )

    semantic: list[dict[str, object]] = []
    source_only: list[dict[str, object]] = []
    for item in repairs:
        if SOURCE_ONLY_KEY in item.source_snapshot:
            validate_item_source_only_receipt(session, item=item, run=run)
            source_only.append(
                {
                    "chunk_id": item.regulatory_chunk_id,
                    "receipt_sha256": publication_digest(
                        SourceOnlySelfRepealReceipt.model_validate(
                            item.source_snapshot[SOURCE_ONLY_KEY]
                        ).model_dump(mode="json")
                    ),
                }
            )
            continue
        validate_item_semantic_repair_audit(session, item=item, run=run)
        audit = AbstentionRepairAudit.model_validate(
            item.source_snapshot[SEMANTIC_REPAIR_KEY]
        )
        semantic.append(
            {"chunk_id": item.regulatory_chunk_id, "audit_sha256": audit.audit_sha256}
        )
    if current_labeling_items_for_search(session, run=run, items=repairs) != {
        item.regulatory_chunk_id for item in repairs
    }:
        raise ValueError("A semantic repair is no longer current for review")
    taxonomy_row = session.get_one(
        RegulatoryLabelTaxonomy, run.taxonomy_id, populate_existing=True
    )
    taxonomy = TaxonomyDefinition.model_validate(taxonomy_row.definition)
    if taxonomy.version_hash != taxonomy_row.version_hash:
        raise ValueError("The intake reader frozen taxonomy changed")
    structural: list[dict[str, object]] = []
    scopes: dict[UUID, list[AnnexCanonicalSnapshot]] = {}
    for target in readers:
        proof = validated_reader_proof(session, target, scopes=scopes)
        leaves = list(
            session.scalars(
                select(RegulatoryLabelingItem)
                .where(
                    RegulatoryLabelingItem.run_id == run_id,
                    RegulatoryLabelingItem.regulatory_chunk_id.in_(
                        proof.governed_chunk_ids
                    ),
                )
                .order_by(RegulatoryLabelingItem.regulatory_chunk_id)
                .execution_options(populate_existing=True)
            )
        )
        if {leaf.regulatory_chunk_id for leaf in leaves} != set(
            proof.governed_chunk_ids
        ) or any(leaf.user_file_id != target.user_file_id for leaf in leaves):
            raise ValueError("A native reader's frozen child scope changed")
        for leaf in leaves:
            _validate_completed_leaf(leaf, taxonomy)
            validate_item_semantic_repair_audit(session, item=leaf, run=run)
            validate_item_source_only_receipt(session, item=leaf, run=run)
        current = current_labeling_items_for_search(session, run=run, items=leaves)
        lineage = current_derived_label_sources_for_search(
            session, run=run, target=target
        )
        if target.resolution == "lineage":
            if tuple(lineage) != proof.governed_chunk_ids or current != set(
                proof.governed_chunk_ids
            ):
                raise ValueError(
                    "A native reader's resolved receipts are no longer current"
                )
            by_id = {leaf.regulatory_chunk_id: leaf for leaf in leaves}
            labels, provenance = derived_label_provenance(
                [by_id[identifier] for identifier in lineage]
            )
            if target.labels != labels or target.provenance != provenance:
                raise ValueError("A native reader's projected labels changed")
        elif (
            target.resolution not in {"pending", "unresolved"}
            or run.status == "completed"
            or target.labels
            or target.provenance
        ):
            raise ValueError("A native reader has no resolved label authority")
        structural.append(
            {
                "chunk_id": target.regulatory_chunk_id,
                "proof_sha256": publication_digest(proof.model_dump(mode="json")),
                "lineage_sha256": publication_digest(list(lineage)),
                "resolution": target.resolution,
                "unresolved_reason_sha256": publication_digest(
                    target.unresolved_reason
                ),
                "leaf_states_sha256": publication_digest(
                    [_receipt_state(leaf, current) for leaf in leaves]
                ),
                "labels_sha256": publication_digest(target.labels),
                "provenance_sha256": publication_digest(target.provenance),
            }
        )
    return {
        "version": 1,
        "run_id": str(run_id),
        "run_status": run.status,
        "provider_binding_sha256": publication_digest(run.provider_binding),
        "taxonomy_id": str(run.taxonomy_id),
        "model": run.model,
        "semantic_repairs": semantic,
        **({"source_only_dispositions": source_only} if source_only else {}),
        "structural_readers": structural,
    }
