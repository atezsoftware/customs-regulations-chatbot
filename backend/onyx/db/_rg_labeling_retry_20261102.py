"""Guarded normal retry for the exact unpublished Resmi Gazete labeling run."""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any
from uuid import UUID

import _new_document_rg_operator as scope
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from onyx.configs.app_configs import POSTGRES_DB
from onyx.db import document_intake as intake
from onyx.db.document_intake_metadata import metadata_chunks
from onyx.db.engine.sql_engine import SqlEngine, get_session_with_current_tenant
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatorySourceAsset,
)
from onyx.db.regulatory_annex_changes import capture_canonical_scope
from onyx.db.regulatory_labeling import current_labeling_items_for_search
from onyx.document_index.publication_models import publication_digest
from onyx.regulatory.document_intake.structural_readers import (
    build_native_chapter_reader_proofs,
)
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from onyx.regulatory.labeling.provider import (
    TaxonomyDefinition,
    build_labeling_request,
    validate_labeling_response,
)
from onyx.utils.variable_functionality import set_is_ee_based_on_env_variable
from shared_configs.contextvars import get_current_tenant_id

PDF_METADATA_SHA = "360736824a7f7588a2ae032d4901238d5e3f9870f01cbe3c1246ed485927d5b1"
IBRAZ = "rc_9445ee9075213f582ea39b7124dc6275a32793c6"
RETRY_IMAGE_SHA: str | None = None  # Pinned only after the final release is approved.


def require_failed_run(run: RegulatoryLabelingRun) -> None:
    scope.require(
        run.id == scope.LABELING_RUN
        and run.document_set_id == 15
        and set(run.file_ids) == {scope.MAIN_FILE, scope.PDF_FILE}
        and run.status == "completed_with_errors"
        and run.stage == "finished"
        and (run.total_chunks, run.completed_chunks, run.failed_chunks)
        == (662, 619, 43)
        and (run.stale_chunks, run.derived_chunks, run.unresolved_derived_chunks)
        == (0, 35, 2)
        and not run.cancel_requested
        and run.lease_token is None
        and run.lease_expires_at is None,
        "The original terminal labeling receipt changed",
    )


def retry_plan(
    session: Session, batch: scope.RegulatoryDocumentIntake
) -> dict[str, Any]:
    scope.require(
        batch.labeling_run_id == scope.LABELING_RUN
        and (batch.status, batch.stage)
        in {("preparing", "labeling"), ("partial", "labeling")}
        and batch.review_sha256 is None
        and batch.lease_token is None
        and batch.lease_expires_at is None,
        "Only the original idle failed labeling stage may be retried",
    )
    run = session.get_one(
        RegulatoryLabelingRun, scope.LABELING_RUN, populate_existing=True
    )
    require_failed_run(run)
    source = scope.review_receipt_proof(
        session, batch, run, verify_completed_labels=False
    )
    scope.require(
        {
            item["user_file_id"]: item["metadata_input_sha256"]
            for item in source["documents"]
        }
        == {scope.MAIN_FILE: scope.MAIN_METADATA_SHA, scope.PDF_FILE: PDF_METADATA_SHA},
        "The post-metadata canonical input receipts changed",
    )
    all_rows = list(
        session.scalars(
            select(RegulatoryChunk).where(
                RegulatoryChunk.user_file_id.in_(
                    [UUID(scope.MAIN_FILE), UUID(scope.PDF_FILE)]
                )
            )
        )
    )
    canonical = {chunk.chunk_id for chunk in metadata_chunks(all_rows)}
    items = list(
        session.scalars(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == run.id
            )
        )
    )
    scope.require(
        len(items) == 662 and {item.regulatory_chunk_id for item in items} == canonical,
        "The original item source scope changed",
    )
    completed = [item for item in items if item.status == "completed"]
    failed = [item for item in items if item.status == "failed"]
    scope.require(
        len(completed) == 619
        and len(failed) == 43
        and all(
            item.error == "Model abstained: insufficient evidence"
            and item.labels == []
            and item.assignments == []
            for item in failed
        ),
        "The exact abstention-only failure set changed",
    )
    current = current_labeling_items_for_search(session, run=run, items=completed)
    scope.require(
        current == {item.regulatory_chunk_id for item in completed},
        "An old completed source or context receipt is stale",
    )
    taxonomy = TaxonomyDefinition.model_validate(run.taxonomy.definition)
    scope.require(
        taxonomy.version_hash == run.taxonomy.version_hash,
        "The selected taxonomy changed",
    )
    for item in completed:
        request = build_labeling_request(
            chunk_id=item.regulatory_chunk_id,
            text=item.text_snapshot,
            context=item.context_snapshot or "",
            taxonomy=taxonomy,
            source_hash=item.canonical_text_sha256 or "",
        )
        scope.require(
            item.request_payload is not None
            and VertexBatchRequest.model_validate(item.request_payload) == request
            and item.request_hash == request.request_hash,
            "An original completed base request changed",
        )
        outcome = validate_labeling_response(
            json.dumps({"labels": item.assignments, "abstained": False}),
            text=item.text_snapshot,
            taxonomy=taxonomy,
        )
        scope.require(
            item.labels == [assignment.label_id for assignment in outcome.labels],
            "An original completed accepted assignment changed",
        )
    asset = session.get_one(
        RegulatorySourceAsset, UUID("ed12363b-36f7-4d93-a377-1ae01e3c6d2b")
    )
    with scope.retained_source_stream(asset.file_id, session) as stream:
        html = stream.read(2_000_001)
    scope.require(
        len(html) <= 2_000_000 and hashlib.sha256(html).hexdigest() == asset.sha256,
        "The native main HTML stream changed",
    )
    main_rows = capture_canonical_scope(session, UUID(scope.MAIN_FILE))
    proofs = build_native_chapter_reader_proofs(
        raw_html=html,
        expected_sha256=asset.sha256,
        user_file_id=scope.MAIN_FILE,
        source_asset_id=asset.id,
        rows=main_rows,
    )
    caption_ids = set(proofs)
    failed_ids = {item.regulatory_chunk_id for item in failed}
    scope.require(
        len(proofs) == 11
        and caption_ids <= canonical
        and caption_ids & failed_ids == {IBRAZ}
        and len(proofs[IBRAZ].governed_chunk_ids) == 26,
        "The fresh native chapter-caption plan differs from the verified source",
    )
    provider_ids = canonical - caption_ids
    reused = current - caption_ids
    pending = failed_ids - caption_ids
    scope.require(
        len(provider_ids) == 651
        and len(reused) == 609
        and len(pending) == 42
        and provider_ids == reused | pending,
        "The exact compatible reuse and pending target plan changed",
    )
    scope.require(
        all(set(proof.governed_chunk_ids) <= provider_ids for proof in proofs.values()),
        "A reader depends on a non-provider or cross-scope source",
    )
    return {
        "operator_action": "labeling_retry_plan_read_only",
        "old_run_id": str(run.id),
        "old_run_status": run.status,
        "source_proof": source,
        "current_old_completed_receipts": len(current),
        "old_receipts_sha256": publication_digest(
            [
                {
                    key: getattr(item, key)
                    for key in (
                        "regulatory_chunk_id",
                        "source_snapshot",
                        "context_snapshot",
                        "request_hash",
                        "request_payload",
                        "labels",
                        "assignments",
                    )
                }
                for item in sorted(completed, key=lambda item: item.regulatory_chunk_id)
            ]
        ),
        "canonical_scope_sha256": publication_digest(
            [row.model_dump(mode="json") for row in main_rows]
        ),
        "native_readers": [
            {
                "chunk_id": identifier,
                "heading": proof.heading_text,
                "chapter": proof.chapter_label,
                "governed_children": len(proof.governed_chunk_ids),
                "scope_sha256": proof.scope_sha256,
                "proof_sha256": publication_digest(proof.model_dump(mode="json")),
            }
            for identifier, proof in sorted(proofs.items())
        ],
        "expected_provider_targets": 651,
        "expected_derived_targets": 46,
        "expected_reused_receipts": 609,
        "expected_pending_targets": 42,
        "provider_ids_sha256": publication_digest(sorted(provider_ids)),
        "pending_ids_sha256": publication_digest(sorted(pending)),
        "approval_publication_index_or_provider_requested": False,
    }


def request_retry(
    session: Session, batch: scope.RegulatoryDocumentIntake
) -> dict[str, Any]:
    proof = retry_plan(session, batch)
    epochs = scope.unpublished_epoch_scope(session)
    result = intake.intake_snapshot(session, batch)
    scope.require(
        result["status"] == "partial"
        and result["stage"] == "labeling"
        and result["labeling_failed_count"] == 43,
        "The normal GET did not expose the exact terminal labeling failure",
    )
    deliveries = intake.retry_intake(session, batch)
    scope.require(
        deliveries == []
        and (batch.status, batch.stage) == ("preparing", "chunking")
        and batch.error is None
        and batch.labeling_run_id == scope.LABELING_RUN,
        "The normal intake retry returned an unexpected state",
    )
    scope.require(
        scope.unpublished_epoch_scope(session) == epochs,
        "The retry changed publication or indexing authority",
    )
    scope.verify_scope(session, batch)
    proof["operator_action"] = "normal_labeling_retry_requested"
    proof["normal_get_status"] = result["status"]
    proof["delivery"] = "standard_enqueue_intake_batch_without_reprepare_request"
    return proof


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-image-sha", required=True)
    parser.add_argument("--observed-image-sha", required=True)
    parser.add_argument("--retry-labeling-v4", action="store_true")
    args = parser.parse_args()
    scope.require(
        args.expected_image_sha == args.observed_image_sha
        and bool(scope.re.fullmatch(r"[0-9a-f]{40}", args.expected_image_sha)),
        "The ready deployed image differs from the approved exact revision",
    )
    scope.require(
        not args.retry_labeling_v4
        or (RETRY_IMAGE_SHA is not None and args.expected_image_sha == RETRY_IMAGE_SHA),
        "Labeling retry is disabled until its exact final deployed release is approved",
    )
    scope.require(
        POSTGRES_DB == "customs-regulations-dev"
        and scope.config.REGULATORY_ANNEX_ENVIRONMENT == "dev"
        and get_current_tenant_id() == "public",
        "Only the frozen DEV tenant may run this incident action",
    )
    set_is_ee_based_on_env_variable()
    SqlEngine.init_engine(pool_size=2, max_overflow=0)
    with get_session_with_current_tenant() as session:
        if not args.retry_labeling_v4:
            session.execute(text("SET TRANSACTION READ ONLY"))
        batch = session.get_one(
            scope.RegulatoryDocumentIntake,
            scope.BATCH,
            with_for_update=args.retry_labeling_v4,
            populate_existing=True,
        )
        report = (
            request_retry(session, batch)
            if args.retry_labeling_v4
            else retry_plan(session, batch)
        )
        if args.retry_labeling_v4:
            session.commit()
        else:
            session.rollback()
    if args.retry_labeling_v4:
        from onyx.background.celery.tasks.regulatory_amendments.document_intake import (
            enqueue_intake_batch,
        )

        enqueue_intake_batch(batch_id=scope.BATCH, tenant_id=get_current_tenant_id())
    report["expected_image_sha"] = args.expected_image_sha
    print(json.dumps(report, ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
