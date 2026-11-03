"""Guarded normal retry for the exact unpublished Resmi Gazete labeling run."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from typing import Any
from uuid import UUID

import _new_document_rg_operator as scope
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from onyx.configs.app_configs import POSTGRES_DB
from onyx.db import document_intake as intake
from onyx.db import regulatory_labeling as labeling
from onyx.db.document_intake_metadata import metadata_chunks
from onyx.db.document_intake_review import parent_labeling_contexts
from onyx.db.engine.sql_engine import SqlEngine, get_session_with_current_tenant
from onyx.db.labeling_configuration import (
    LabelingProviderBinding,
    resolve_intake_labeling_provider_binding,
)
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryLabelingShard,
    RegulatorySourceAsset,
    User,
)
from onyx.db.regulatory_annex_changes import capture_canonical_scope
from onyx.db.regulatory_labeling import current_labeling_items_for_search
from onyx.db.regulatory_labeling_semantic_repair import (
    require_current_v4_intake_repair_item,
)
from onyx.document_index.publication_models import publication_digest
from onyx.file_store.postgres_file_store import (
    POSTGRES_BUCKET_SENTINEL,
    PostgresBackedFileStore,
)
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.document_intake.structural_readers import (
    build_native_chapter_reader_proofs,
)
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from onyx.regulatory.labeling.intake_abstention_repair import build_abstention_repair
from onyx.regulatory.labeling.provider import (
    TaxonomyDefinition,
    build_labeling_request,
    parse_labeling_batch_output,
    validate_labeling_response,
)
from onyx.regulatory.labeling.vertex_direct import _StoredResult
from onyx.utils.variable_functionality import set_is_ee_based_on_env_variable
from shared_configs.contextvars import get_current_tenant_id

PDF_METADATA_SHA = "360736824a7f7588a2ae032d4901238d5e3f9870f01cbe3c1246ed485927d5b1"
IBRAZ = "rc_9445ee9075213f582ea39b7124dc6275a32793c6"
RETRY_IMAGE_SHA: str | None = None  # Pinned only after the final release is approved.


def canonical_source_sha256(session: Session) -> str:
    return publication_digest(
        [
            row.model_dump(mode="json")
            for file_id in sorted([scope.MAIN_FILE, scope.PDF_FILE])
            for row in capture_canonical_scope(session, UUID(file_id))
        ]
    )


def complete_ack_context(line: str, request_hash: str) -> str:
    payload = json.loads(line)
    response = payload.get("response") if isinstance(payload, dict) else None
    candidates = response.get("candidates") if isinstance(response, dict) else None
    scope.require(
        isinstance(candidates, list)
        and len(candidates) == 1
        and isinstance(candidates[0], dict)
        and candidates[0].get("finishReason") == "STOP",
        "A retained failed ACK requires exactly one complete STOP candidate",
    )
    parsed = parse_labeling_batch_output(iter([line]), {request_hash})[request_hash]
    scope.require(
        parsed.error is None and parsed.context is not None,
        "A retained failed ACK is not a complete STOP result",
    )
    assert parsed.context is not None
    return parsed.context


def retained_ack_bytes(session: Session, file_id: str, binding: str) -> bytes:
    records = scope.rows(
        session,
        """
        SELECT r.file_metadata,r.bucket_name,c.file_size
        FROM file_record r JOIN file_content c ON c.file_id=r.file_id
        WHERE r.file_id=:id
    """,
        id=file_id,
    )
    scope.require(
        len(records) == 1
        and records[0]["bucket_name"] == POSTGRES_BUCKET_SENTINEL
        and 0 < records[0]["file_size"] <= 2_000_000
        and records[0]["file_metadata"].get("binding_sha256") == binding,
        "A retained direct ACK has no exact bounded PostgreSQL cache record",
    )
    with PostgresBackedFileStore().read_file(file_id, db_session=session) as stream:
        raw = stream.read(2_000_001)
    scope.require(
        len(raw) == records[0]["file_size"]
        and hashlib.sha256(raw).hexdigest()
        == records[0]["file_metadata"].get("sha256"),
        "A retained PostgreSQL ACK changed",
    )
    return raw


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


def failed_request_is_current(
    session: Session,
    *,
    run: RegulatoryLabelingRun,
    item: RegulatoryLabelingItem,
    frozen: dict[str, RegulatoryLabelingItem],
) -> None:
    """Read the source predicate directly; never claim or synthesize a run lease."""
    require_current_v4_intake_repair_item(session, item=item, run=run)
    lower, upper = labeling._snapshot_context_window(item.source_snapshot)
    rows, boundaries, complete = labeling._current_file_context_candidates(
        session,
        document_set_id=run.document_set_id,
        user_file_id=item.user_file_id,
        lower=lower,
        upper=upper,
    )
    advisories = labeling._latest_context_advisories(
        session, [item.regulatory_chunk_id]
    )
    scope.require(
        labeling._item_snapshot_is_current(
            item,
            snapshot_items=frozen,
            current_rows={row.id: row for row in rows},
            current_file_rows={item.user_file_id: rows},
            current_boundary_ids={item.user_file_id: boundaries},
            current_windows_complete={item.user_file_id: complete},
            parent_identities=parent_labeling_contexts(session, [item.user_file_id]),
            advisory_hashes={
                key: context_hash(value) for key, value in advisories.items()
            },
        ),
        "A failed target's original frozen interpretation context changed",
    )


def pure_repair_plan(
    session: Session,
    *,
    batch: scope.RegulatoryDocumentIntake,
    run: RegulatoryLabelingRun,
    items: list[RegulatoryLabelingItem],
    pending_ids: set[str],
    taxonomy: TaxonomyDefinition,
) -> dict[str, Any]:
    binding = LabelingProviderBinding.model_validate(run.provider_binding)
    current = resolve_intake_labeling_provider_binding(
        session,
        batch.labeling_model_configuration_id,
        user=session.get_one(User, batch.created_by),
    )
    scope.require(
        binding.transport == "vertex_direct_v1"
        and run.model == binding.model
        and current.model_dump(mode="json") == run.provider_binding,
        "The current direct provider/model binding differs from the original frozen run",
    )
    frozen = {item.regulatory_chunk_id: item for item in items}
    shards = {
        shard.id: shard
        for shard in session.scalars(
            select(RegulatoryLabelingShard).where(
                RegulatoryLabelingShard.run_id == run.id
            )
        )
    }
    total_bytes = 0
    records = []
    for identifier in sorted(pending_ids):
        item = frozen[identifier]
        failed_request_is_current(session, run=run, item=item, frozen=frozen)
        base = build_labeling_request(
            chunk_id=identifier,
            text=item.text_snapshot,
            context=item.context_snapshot or "",
            taxonomy=taxonomy,
            source_hash=item.canonical_text_sha256 or "",
        )
        scope.require(
            item.request_payload is not None
            and VertexBatchRequest.model_validate(item.request_payload) == base
            and item.request_hash == base.request_hash,
            "A failed target's exact base request changed",
        )
        shard = shards.get(item.shard_id) if item.shard_id is not None else None
        scope.require(
            shard is not None
            and str(item.id) in shard.item_ids
            and shard.status == "succeeded",
            "A failed target no longer has its complete provider shard",
        )
        assert shard is not None
        submission = shard.submission_key.removeprefix("regulatory-labeling-")
        scope.require(
            bool(scope.re.fullmatch(r"[0-9a-f]{64}", submission)),
            "A retained ACK submission identity is malformed",
        )
        file_id = f"regulatory-labeling-direct/{binding.fingerprint}/{submission}/{base.request_hash}.json"
        raw = retained_ack_bytes(session, file_id, binding.fingerprint)
        total_bytes += len(raw)
        scope.require(
            total_bytes <= 16_000_000,
            "A retained ACK changed or exceeded its bounded read budget",
        )
        ack = _StoredResult.model_validate_json(raw)
        line = json.loads(ack.line)
        scope.require(
            ack.binding_sha256 == binding.fingerprint
            and ack.submission_key == shard.submission_key
            and ack.request_hash == base.request_hash
            and VertexBatchRequest.model_validate(ack.request_payload) == base
            and hashlib.sha256(ack.line.encode()).hexdigest() == ack.line_sha256
            and line.get("key") == base.request_hash
            and line.get("request") == base.to_generate_content_request(),
            "A retained failed ACK lost exact request correlation",
        )
        response_context = complete_ack_context(ack.line, base.request_hash)
        repair = build_abstention_repair(
            base_request=base,
            rejected_response=response_context,
            text=item.text_snapshot,
            chunk_id=identifier,
            source_hash=item.canonical_text_sha256 or "",
            taxonomy=taxonomy,
            provider_binding=run.provider_binding,
        )
        records.append(
            {
                "chunk_id": identifier,
                "base_request_hash": base.request_hash,
                "context_sha256": repair.context_text_sha256,
                "rejected_response_sha256": repair.rejected_response_sha256,
                "retained_ack_sha256": hashlib.sha256(raw).hexdigest(),
                "repair_request_hash": repair.request.request_hash,
                "repair_request_bytes": len(
                    json.dumps(
                        repair.request.to_generate_content_request(), ensure_ascii=False
                    ).encode()
                ),
                "eligible": True,
                "verified_sole_candidate_finish_reason": "STOP",
            }
        )
    return {
        "eligible_failed_requests": len(records),
        "provider_calls": 0,
        "ack_bytes_read": total_bytes,
        "binding_keys": sorted(run.provider_binding),
        "provider_binding_fingerprint": binding.fingerprint,
        "transport": binding.transport,
        "model": run.model,
        "current_binding_matches": True,
        "requests": records,
    }


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
    files = scope.rows(
        session,
        """
        SELECT u.id,u.status,u.user_id,m.approval_attempt_id
        FROM regulatory_document_intake_file m JOIN user_file u ON u.id=m.user_file_id
        WHERE m.intake_id=:batch ORDER BY u.id
    """,
        batch=scope.BATCH,
    )
    scope.require(
        {str(file["id"]) for file in files} == {scope.MAIN_FILE, scope.PDF_FILE}
        and all(
            file["status"] == "CHUNKED"
            and str(file["user_id"]) == scope.OWNER
            and file["approval_attempt_id"] is None
            for file in files
        ),
        "Both owned files must remain chunked and never approved",
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
    semantic_plan = pure_repair_plan(
        session,
        batch=batch,
        run=run,
        items=items,
        pending_ids=pending,
        taxonomy=taxonomy,
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
        "all_canonical_source_sha256": canonical_source_sha256(session),
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
        "pure_semantic_repair_plan": semantic_plan,
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
    scope.require(
        canonical_source_sha256(session) == proof["all_canonical_source_sha256"],
        "The normal retry changed canonical source data",
    )
    scope.verify_scope(session, batch)
    proof["operator_action"] = "normal_labeling_retry_requested"
    proof.pop("approval_publication_index_or_provider_requested", None)
    proof["approval_publication_or_index_requested"] = False
    proof["provider_requested_via_standard_intake_retry"] = True
    proof["normal_get_status"] = result["status"]
    proof["delivery"] = "standard_enqueue_intake_batch_without_reprepare_request"
    return proof


def observe_replacement(expected_source_sha256: str) -> dict[str, Any]:
    """Watch worker-created rows; never prepare a run or dispatch provider work."""
    for _ in range(120):
        with get_session_with_current_tenant() as session:
            session.execute(text("SET TRANSACTION READ ONLY"))
            batch = session.get_one(scope.RegulatoryDocumentIntake, scope.BATCH)
            if batch.labeling_run_id != scope.LABELING_RUN:
                run = session.get_one(RegulatoryLabelingRun, batch.labeling_run_id)
                old = session.get_one(RegulatoryLabelingRun, scope.LABELING_RUN)
                require_failed_run(old)
                scope.require(
                    (run.total_chunks, run.derived_chunks) == (651, 46)
                    and run.file_ids == old.file_ids
                    and run.provider_binding == old.provider_binding
                    and run.taxonomy_id == old.taxonomy_id,
                    "The worker-created replacement differs from the approved plan",
                )
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
                reused = [
                    item
                    for item in items
                    if old_items[item.regulatory_chunk_id].status == "completed"
                ]
                pending = [
                    item
                    for item in items
                    if old_items[item.regulatory_chunk_id].status == "failed"
                ]
                scope.require(
                    len(items) == 651 and len(reused) == 609 and len(pending) == 42,
                    "The worker changed the approved reuse/remaining source sets",
                )
                fields = (
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
                        item.status == "completed"
                        and all(
                            getattr(item, key)
                            == getattr(old_items[item.regulatory_chunk_id], key)
                            for key in fields
                        )
                        for item in reused
                    ),
                    "A compatible completed receipt was not copied immutably",
                )
                scope.require(
                    all(
                        item.status == "pending"
                        and item.request_hash is None
                        and item.request_payload is None
                        for item in pending
                    )
                    and run.completed_chunks == 609,
                    "The initial remaining42 pending state was not observed before labeling advanced",
                )
                scope.require(
                    canonical_source_sha256(session) == expected_source_sha256,
                    "The worker retry changed canonical source data",
                )
                scope.unpublished_epoch_scope(session)
                return {
                    "replacement_run_id": str(run.id),
                    "provider_targets": 651,
                    "derived_targets": 46,
                    "observed_exact_reused_receipts": 609,
                    "observed_initial_pending_targets": 42,
                    "canonical_source_sha256": expected_source_sha256,
                    "old_run_unchanged": True,
                    "provider_dispatched_by_operator": False,
                }
            session.rollback()
        time.sleep(0.5)
    raise RuntimeError(
        "The normal worker did not create the replacement within the bounded observation window; inspect without reenqueuing"
    )


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
        print(
            json.dumps(
                {
                    "operator_event": "normal_retry_committed_and_standard_delivery_sent",
                    "batch_id": str(scope.BATCH),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        report["observed_worker_replacement"] = observe_replacement(
            report["all_canonical_source_sha256"]
        )
    report["expected_image_sha"] = args.expected_image_sha
    print(json.dumps(report, ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
