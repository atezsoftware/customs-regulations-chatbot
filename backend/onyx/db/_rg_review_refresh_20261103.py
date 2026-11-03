"""Verify one worker-created DEV labeling retry; refresh review through normal GET."""

from __future__ import annotations

import argparse
import json
from typing import Any
from uuid import UUID, uuid5

import _new_document_rg_operator as scope
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from onyx.configs.app_configs import POSTGRES_DB
from onyx.db import document_intake as intake
from onyx.db.document_intake_structural_readers import (
    load_intake_structural_reader_proofs,
)
from onyx.db.engine.sql_engine import SqlEngine, get_session_with_current_tenant
from onyx.db.labeling_configuration import resolve_intake_labeling_provider_binding
from onyx.db.models import (
    RegulatoryDerivedLabelProjection,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    User,
)
from onyx.db.regulatory_annex_changes import capture_canonical_scope
from onyx.db.regulatory_labeling import (
    current_derived_label_sources_for_search,
    current_labeling_items_for_search,
    derived_label_provenance,
)
from onyx.db.regulatory_labeling_readers import STRUCTURAL_READER_KEY
from onyx.db.regulatory_labeling_review_authority import labeling_review_authority
from onyx.db.regulatory_labeling_semantic_repair import (
    validate_item_semantic_repair_audit,
)
from onyx.document_index.publication_models import publication_digest
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from onyx.regulatory.labeling.intake_abstention_repair import ABSTENTION_REPAIR_KEY
from onyx.regulatory.labeling.provider import (
    TaxonomyDefinition,
    build_labeling_request,
    validate_labeling_response,
)
from onyx.utils.variable_functionality import set_is_ee_based_on_env_variable
from shared_configs.contextvars import get_current_tenant_id

FINAL_IMAGE_SHA = "2fc69ee1ad9ca619c4266a8b2c08a2ef44523fe3"
PINNED_REVIEW_RUN_ID: UUID | None = UUID("b6fd5ae5-5b51-4c8a-a332-910e86732b0a")
REFRESH_REVIEW_ENABLED = False
PINNED_REVIEW_SOURCE_SHA256: str | None = (
    "de0ffa6d4ee73b11499026af190690fe29760d77a51cfd8bf069601dfa96fd89"
)
PDF_METADATA_SHA = "360736824a7f7588a2ae032d4901238d5e3f9870f01cbe3c1246ed485927d5b1"


def source_digest(session: Session) -> str:
    return publication_digest(
        [
            row.model_dump(mode="json")
            for file_id in sorted([scope.MAIN_FILE, scope.PDF_FILE])
            for row in capture_canonical_scope(session, UUID(file_id))
        ]
    )


def validate_completed_run(run: RegulatoryLabelingRun, batch: Any) -> None:
    scope.require(
        run.id != scope.LABELING_RUN
        and run.id == batch.labeling_run_id
        and run.idempotency_key == uuid5(scope.BATCH, str(scope.LABELING_RUN))
        and run.document_set_id == 15
        and str(run.requested_by_id) == scope.OWNER
        and run.model_configuration_id == batch.labeling_model_configuration_id
        and set(run.file_ids) == {scope.MAIN_FILE, scope.PDF_FILE}
        and (run.status, run.stage) == ("completed", "finished")
        and (
            run.total_chunks,
            run.completed_chunks,
            run.failed_chunks,
            run.stale_chunks,
        )
        == (651, 651, 0, 0)
        and (run.derived_chunks, run.unresolved_derived_chunks) == (46, 0)
        and run.error is None
        and not run.cancel_requested
        and run.lease_token is None
        and run.lease_expires_at is None,
        "Only the exact completed worker-created replacement may refresh review",
    )


def review_plan(session: Session, batch: Any) -> dict[str, Any]:
    scope.require(
        batch.labeling_run_id is not None
        and batch.labeling_run_id != scope.LABELING_RUN
        and (batch.status, batch.stage)
        in {("preparing", "labeling"), ("ready", "review")}
        and batch.error is None
        and batch.lease_token is None
        and batch.lease_expires_at is None,
        "The batch is not a completed unapproved replacement labeling stage",
    )
    files = scope.rows(
        session,
        """SELECT u.id,u.status,u.user_id,m.approval_attempt_id
        FROM regulatory_document_intake_file m JOIN user_file u ON u.id=m.user_file_id
        WHERE m.intake_id=:batch ORDER BY u.id""",
        batch=scope.BATCH,
    )
    scope.require(
        {str(item["id"]) for item in files} == {scope.MAIN_FILE, scope.PDF_FILE}
        and all(
            item["status"] == "CHUNKED"
            and str(item["user_id"]) == scope.OWNER
            and item["approval_attempt_id"] is None
            for item in files
        ),
        "Review requires both original owned chunked and never-approved files",
    )
    run = session.get_one(
        RegulatoryLabelingRun, batch.labeling_run_id, populate_existing=True
    )
    validate_completed_run(run, batch)
    if PINNED_REVIEW_RUN_ID is not None:
        scope.require(
            run.id == PINNED_REVIEW_RUN_ID, "The pinned replacement run changed"
        )
    old = session.get_one(
        RegulatoryLabelingRun, scope.LABELING_RUN, populate_existing=True
    )
    scope.require(
        (old.status, old.stage) == ("completed_with_errors", "finished")
        and (
            old.total_chunks,
            old.completed_chunks,
            old.failed_chunks,
            old.stale_chunks,
        )
        == (662, 619, 43, 0)
        and old.file_ids == run.file_ids
        and old.provider_binding == run.provider_binding
        and old.taxonomy_id == run.taxonomy_id
        and old.model == run.model,
        "The original failed receipt or replacement provider/taxonomy identity changed",
    )
    retained = scope.review_receipt_proof(
        session, batch, run, verify_completed_labels=False
    )
    scope.require(
        {
            item["user_file_id"]: item["metadata_input_sha256"]
            for item in retained["documents"]
        }
        == {scope.MAIN_FILE: scope.MAIN_METADATA_SHA, scope.PDF_FILE: PDF_METADATA_SHA},
        "The complete original metadata source inputs changed",
    )
    current_source_digest = source_digest(session)
    if PINNED_REVIEW_SOURCE_SHA256 is not None:
        scope.require(
            current_source_digest == PINNED_REVIEW_SOURCE_SHA256,
            "The pinned complete canonical source changed",
        )
    binding = resolve_intake_labeling_provider_binding(
        session,
        batch.labeling_model_configuration_id,
        user=session.get_one(User, batch.created_by),
    )
    scope.require(
        binding.model_dump(mode="json") == run.provider_binding
        and run.model == binding.model,
        "The current configured labeling binding changed",
    )
    taxonomy = TaxonomyDefinition.model_validate(run.taxonomy.definition)
    scope.require(
        taxonomy.version_hash == run.taxonomy.version_hash,
        "The frozen taxonomy changed",
    )
    proofs = load_intake_structural_reader_proofs(session, intake_id=batch.id)
    scope.require(len(proofs) == 11, "The fresh native chapter reader scope changed")
    items = list(
        session.scalars(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == run.id
            )
        )
    )
    old_items = {
        item.regulatory_chunk_id: item
        for item in session.scalars(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == old.id
            )
        )
    }
    by_id = {item.regulatory_chunk_id: item for item in items}
    scope.require(
        len(old_items) == 662
        and len(items) == len(by_id) == 651
        and set(old_items) - set(proofs) == set(by_id)
        and all(item.status == "completed" and item.error is None for item in items),
        "The replacement providers do not cover the exact original ordinary source set",
    )
    scope.require(
        current_labeling_items_for_search(session, run=run, items=items) == set(by_id),
        "A completed replacement source/context receipt is stale",
    )
    reused = [
        item
        for item in items
        if old_items[item.regulatory_chunk_id].status == "completed"
    ]
    remaining = [
        item for item in items if old_items[item.regulatory_chunk_id].status == "failed"
    ]
    scope.require(
        len(reused) == 609 and len(remaining) == 42,
        "The verified reuse and reconsidered source sets changed",
    )
    copied_fields = (
        "source_snapshot",
        "text_snapshot",
        "context_snapshot",
        "context_sha256",
        "canonical_text_sha256",
        "request_hash",
        "request_payload",
        "labels",
        "assignments",
    )
    scope.require(
        all(
            all(
                getattr(item, key) == getattr(old_items[item.regulatory_chunk_id], key)
                for key in copied_fields
            )
            for item in reused
        ),
        "An exact compatible receipt changed during retry",
    )
    base_fields = (
        "text_snapshot",
        "context_snapshot",
        "canonical_text_sha256",
        "request_hash",
        "request_payload",
    )
    scope.require(
        all(
            all(
                getattr(item, key) == getattr(old_items[item.regulatory_chunk_id], key)
                for key in base_fields
            )
            for item in remaining
        ),
        "A remaining target's original frozen base/context changed",
    )
    semantic_count = semantic_empty = direct_empty = 0
    for item in items:
        expected = build_labeling_request(
            chunk_id=item.regulatory_chunk_id,
            text=item.text_snapshot,
            context=item.context_snapshot or "",
            taxonomy=taxonomy,
            source_hash=item.canonical_text_sha256 or "",
        )
        scope.require(
            item.request_payload is not None
            and VertexBatchRequest.model_validate(item.request_payload) == expected
            and item.request_hash == expected.request_hash,
            "An accepted frozen base request changed",
        )
        outcome = validate_labeling_response(
            json.dumps({"labels": item.assignments, "abstained": False}),
            text=item.text_snapshot,
            taxonomy=taxonomy,
        )
        scope.require(
            item.labels == [assignment.label_id for assignment in outcome.labels],
            "Accepted labels differ from their strict source quotes",
        )
        validate_item_semantic_repair_audit(session, item=item, run=run)
        if ABSTENTION_REPAIR_KEY in item.source_snapshot:
            semantic_count += 1
            semantic_empty += not item.labels
        elif not item.labels:
            direct_empty += 1
    targets = list(
        session.scalars(
            select(RegulatoryDerivedLabelProjection).where(
                RegulatoryDerivedLabelProjection.run_id == run.id
            )
        )
    )
    scope.require(
        len(targets) == 46
        and len({item.regulatory_chunk_id for item in targets}) == 46,
        "The derived scope is not exactly46 targets",
    )
    native_count = 0
    for target in targets:
        scope.require(
            target.resolution == "lineage" and target.unresolved_reason is None,
            "A derived source is unresolved",
        )
        lineage = current_derived_label_sources_for_search(
            session, run=run, target=target
        )
        scope.require(
            bool(lineage)
            and set(lineage) <= set(by_id)
            and all(
                by_id[identifier].user_file_id == target.user_file_id
                for identifier in lineage
            ),
            "A derived target lacks complete same-file ordinary source authority",
        )
        labels, provenance = derived_label_provenance(
            [by_id[identifier] for identifier in lineage]
        )
        scope.require(
            target.labels == labels and target.provenance == provenance,
            "A derived label or child evidence provenance changed",
        )
        if STRUCTURAL_READER_KEY in target.source_snapshot:
            native_count += 1
            proof = proofs.get(target.regulatory_chunk_id)
            scope.require(
                proof is not None
                and target.source_snapshot[STRUCTURAL_READER_KEY]
                == proof.model_dump(mode="json")
                and tuple(lineage) == proof.governed_chunk_ids,
                "A native chapter's frozen proof or child scope changed",
            )
    scope.require(
        native_count == 11,
        "Not every fresh native reader has a matching frozen projection",
    )
    authority = labeling_review_authority(session, run.id)
    native_authority = authority.get("structural_readers") if authority else None
    semantic_authority = authority.get("semantic_repairs") if authority else None
    scope.require(
        isinstance(native_authority, list)
        and isinstance(semantic_authority, list)
        and len(native_authority) == 11
        and len(semantic_authority) == semantic_count,
        "Current review authority is incomplete",
    )
    budget = scope.review_budget_canary()
    scope.require(
        budget["success"] and budget["validated_chunks"] == 697,
        "The complete unchanged review no longer fits its edit budget",
    )
    return {
        "operator_action": "labeling_review_plan_read_only",
        "full_source_chunks": 697,
        "review_edit_budget": budget,
        "labeling_run_id": str(run.id),
        "canonical_source_sha256": current_source_digest,
        "canonical_metadata_receipts": 662,
        "current_provider_receipts": 651,
        "exact_reused_receipts": 609,
        "remaining_original_base_receipts": 42,
        "resolved_derived_receipts": 46,
        "fresh_native_reader_receipts": 11,
        "accepted_semantic_repair_receipts": semantic_count,
        "semantic_no_applicable_labels": semantic_empty,
        "direct_no_applicable_labels": direct_empty,
        "review_authority_sha256": publication_digest(authority),
        "review_sha256": intake.review_digest(session, batch),
        "source_and_metadata_proof": retained,
        "approval_publication_index_or_provider_requested": False,
    }


def refresh_review(session: Session, batch: Any) -> dict[str, Any]:
    proof = review_plan(session, batch)
    scope.require(
        REFRESH_REVIEW_ENABLED
        and PINNED_REVIEW_RUN_ID is not None
        and PINNED_REVIEW_SOURCE_SHA256 is not None
        and batch.labeling_run_id == PINNED_REVIEW_RUN_ID
        and proof["canonical_source_sha256"] == PINNED_REVIEW_SOURCE_SHA256,
        "Review refresh is disabled until the exact replacement/source witnesses are approved",
    )
    epochs = scope.unpublished_epoch_scope(session)
    result = intake.intake_snapshot(session, batch)
    scope.require(
        batch.status == "ready"
        and batch.stage == "review"
        and batch.error is None
        and batch.review_sha256
        == proof["review_sha256"]
        == intake.review_digest(session, batch),
        "Normal GET did not expose the exact current ready review",
    )
    scope.require(
        result["status"] == "ready"
        and result["stage"] == "review"
        and result["labeling_failed_count"] == 0
        and result["active_document_count"] == 2
        and {item["user_file_id"] for item in result["documents"]}
        == {scope.MAIN_FILE, scope.PDF_FILE},
        "Normal ready review has an incomplete document scope",
    )
    by_document = {item["id"]: item for item in result["documents"]}
    scope.require(
        by_document[scope.MAIN_DOCUMENT]["role"] == "root"
        and by_document[scope.PDF_DOCUMENT]["role"] == "annex"
        and by_document[scope.PDF_DOCUMENT]["parent_document_id"] == scope.MAIN_DOCUMENT
        and any(
            link["target_document_id"] == scope.PDF_DOCUMENT
            for link in by_document[scope.MAIN_DOCUMENT]["document_links"]
        ),
        "The ready attachment navigation changed",
    )
    scope.verify_scope(session, batch)
    scope.require(
        source_digest(session) == PINNED_REVIEW_SOURCE_SHA256
        and scope.unpublished_epoch_scope(session) == epochs,
        "Normal GET changed canonical or publication/index authority",
    )
    proof.update(
        operator_action="normal_get_review_refreshed",
        status=batch.status,
        stage=batch.stage,
        review_sha256=batch.review_sha256,
        edit_review_sha256=result["edit_review_sha256"],
        publication_review_sha256=result["publication_review_sha256"],
    )
    return proof


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-image-sha", required=True)
    parser.add_argument("--observed-image-sha", required=True)
    parser.add_argument("--refresh-review", action="store_true")
    args = parser.parse_args()
    scope.require(
        args.expected_image_sha == args.observed_image_sha == FINAL_IMAGE_SHA,
        "Only the exact final deployed DEV image can inspect this review",
    )
    scope.require(
        not args.refresh_review
        or (
            REFRESH_REVIEW_ENABLED
            and PINNED_REVIEW_RUN_ID is not None
            and PINNED_REVIEW_SOURCE_SHA256 is not None
        ),
        "Review refresh is disabled pending exact replacement/source approval",
    )
    scope.require(
        POSTGRES_DB == "customs-regulations-dev"
        and scope.config.REGULATORY_ANNEX_ENVIRONMENT == "dev"
        and get_current_tenant_id() == "public",
        "Only the frozen DEV tenant may inspect this review",
    )
    set_is_ee_based_on_env_variable()
    SqlEngine.init_engine(pool_size=2, max_overflow=0)
    with get_session_with_current_tenant() as session:
        if not args.refresh_review:
            session.execute(text("SET TRANSACTION READ ONLY"))
        batch = session.get_one(
            scope.RegulatoryDocumentIntake,
            scope.BATCH,
            with_for_update=args.refresh_review,
            populate_existing=True,
        )
        report = (
            refresh_review(session, batch)
            if args.refresh_review
            else review_plan(session, batch)
        )
        if args.refresh_review:
            session.commit()
        else:
            session.rollback()
    print(json.dumps(report, ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
