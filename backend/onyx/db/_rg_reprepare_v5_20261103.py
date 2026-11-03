"""One guarded normal reprepare; observe the autonomous pipeline without a GET."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

import _new_document_rg_operator as scope
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from onyx.configs.app_configs import POSTGRES_DB
from onyx.db import document_intake as intake
from onyx.db.document_intake_edit_budget import validate_intake_edit_budgets
from onyx.db.document_intake_metadata import (
    metadata_chunks,
    validate_intake_metadata_receipts,
)
from onyx.db.document_intake_reprepare import (
    latest_reprepare,
    reprepare_snapshot,
    request_reprepare,
)
from onyx.db.document_intake_structural_readers import (
    load_intake_structural_reader_proofs,
)
from onyx.db.engine.sql_engine import SqlEngine, get_session_with_current_tenant
from onyx.db.labeling_configuration import resolve_intake_labeling_provider_binding
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryDerivedLabelProjection,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeReprepare,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    User,
    UserFile,
)
from onyx.db.regulatory_annex_changes import capture_canonical_scope
from onyx.db.regulatory_labeling import (
    current_derived_label_sources_for_search,
    current_labeling_items_for_search,
)
from onyx.db.regulatory_labeling_review_authority import labeling_review_authority
from onyx.document_index.publication_models import publication_digest
from onyx.regulatory.document_intake.metadata import (
    METADATA_AUDIT_KEY,
    METADATA_MODEL,
    METADATA_VERSION,
    metadata_input_hash,
)
from onyx.regulatory.document_intake.models import IntakePreparationPolicy
from onyx.utils.variable_functionality import set_is_ee_based_on_env_variable
from shared_configs.contextvars import get_current_tenant_id

OLD_RUN = UUID("b6fd5ae5-5b51-4c8a-a332-910e86732b0a")
KEY = uuid5(scope.BATCH, "logical-matrix-v5-20261103")
OLD_SOURCE_SHA = "de0ffa6d4ee73b11499026af190690fe29760d77a51cfd8bf069601dfa96fd89"
OLD_PDF_METADATA_SHA = (
    "360736824a7f7588a2ae032d4901238d5e3f9870f01cbe3c1246ed485927d5b1"
)
FINAL_IMAGE_SHA = "eb54c6215dc7dbb9f3ca9029afc11d09830ad77a"
APPROVED_PLAN_SHA = "acca7ee08b6ec9c8e9e30730eb6218d12c201e8225c15f6e3b0dcbba80767d8f"
REPREPARE_ENABLED = True
# Fill only from the source replay made with both real incident assets/file IDs.
V5_PDF_ALL = 197
V5_PDF_CANONICAL = 192
V5_PDF_METADATA_SHA = "116fc65181ed9f62310f42312d28b80a45cae0481d54f18f99caed80a3ffb62a"
V5_PDF_TABLES = 61


def policy(version: str) -> dict[str, Any]:
    value = IntakePreparationPolicy.model_validate(
        {"chunking_version": version}
    ).model_dump(mode="json")
    scope.require(
        value["body_token_limit"] == 512
        and value["table_token_limit"] == 2048
        and value["provider"] == "google"
        and value["model_name"] == "gemini-embedding-2",
        "The frozen scoped embedding policy changed",
    )
    return value


def source_rows(session: Session) -> list[RegulatoryChunk]:
    return list(
        session.scalars(
            select(RegulatoryChunk)
            .where(RegulatoryChunk.user_file_id.in_([scope.MAIN_FILE, scope.PDF_FILE]))
            .order_by(
                RegulatoryChunk.user_file_id,
                RegulatoryChunk.position,
                RegulatoryChunk.id,
            )
        )
    )


def source_digest(session: Session) -> str:
    return publication_digest(
        [
            row.model_dump(mode="json")
            for file_id in sorted([scope.MAIN_FILE, scope.PDF_FILE])
            for row in capture_canonical_scope(session, UUID(file_id))
        ]
    )


def require_old_run(
    run: RegulatoryLabelingRun, batch: RegulatoryDocumentIntake
) -> None:
    scope.require(
        run.id == OLD_RUN
        and run.document_set_id == batch.document_set_id == 15
        and str(run.requested_by_id) == str(batch.created_by) == scope.OWNER
        and run.model_configuration_id == batch.labeling_model_configuration_id
        and set(run.file_ids) == {scope.MAIN_FILE, scope.PDF_FILE}
        and (run.status, run.stage) == ("completed_with_errors", "finished")
        and (run.total_chunks, run.completed_chunks, run.failed_chunks)
        == (651, 640, 11)
        and (run.stale_chunks, run.derived_chunks, run.unresolved_derived_chunks)
        == (0, 46, 0)
        and not run.cancel_requested
        and run.lease_token is None
        and run.lease_expires_at is None,
        "The exact terminal v4 replacement receipt changed",
    )


def require_files(session: Session, batch: RegulatoryDocumentIntake) -> None:
    scope.verify_scope(session, batch)
    scope.unpublished_epoch_scope(session)
    files = scope.rows(
        session,
        """SELECT u.id,u.status,u.user_id,m.approval_attempt_id
        FROM regulatory_document_intake_file m JOIN user_file u ON u.id=m.user_file_id
        WHERE m.intake_id=:batch""",
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
        "The original unpublished owned file scope changed",
    )


def metadata_proof(
    session: Session, batch: RegulatoryDocumentIntake, *, version: str, complete: bool
) -> dict[str, Any]:
    rows = source_rows(session)
    expected = policy(version)
    expected_counts = {
        scope.MAIN_FILE: (365, 342, scope.MAIN_METADATA_SHA),
        scope.PDF_FILE: (332, 320, OLD_PDF_METADATA_SHA)
        if version == "4"
        else (V5_PDF_ALL, V5_PDF_CANONICAL, V5_PDF_METADATA_SHA),
    }
    output = {}
    for file_id, (all_count, canonical_count, sha) in expected_counts.items():
        scoped = [row for row in rows if str(row.user_file_id) == file_id]
        chunks = metadata_chunks(scoped)
        scope.require(
            all_count is not None
            and canonical_count is not None
            and sha is not None
            and (len(scoped), len(chunks)) == (all_count, canonical_count)
            and metadata_input_hash(chunks) == sha
            and all(
                row.status == "active"
                and row.source == "indexed"
                and row.chunk_metadata.get("intake_chunking_version") == version
                and row.chunk_metadata.get("intake_preparation_policy") == expected
                for row in scoped
            ),
            "The frozen source shape, metadata input or scoped policy changed",
        )
        audits = [
            row.chunk_metadata.get(METADATA_AUDIT_KEY, {})
            for row in scoped
            if row.id in {chunk.chunk_id for chunk in chunks}
        ]
        valid = sum(
            audit.get("input_sha256") == sha
            and audit.get("input_stale") is False
            and audit.get("version") == METADATA_VERSION
            and audit.get("model") == METADATA_MODEL
            and isinstance(audit.get("request_hash"), str)
            and bool(re.fullmatch(r"[0-9a-f]{64}", audit["request_hash"]))
            and isinstance(audit.get("fields"), dict)
            and not any(field.get("stale") for field in audit["fields"].values())
            for audit in audits
        )
        scope.require(
            not complete or valid == canonical_count,
            "Metadata evidence is incomplete or stale",
        )
        output[file_id] = {
            "all": len(scoped),
            "canonical": len(chunks),
            "metadata_current": valid,
            "metadata_input_sha256": sha,
            "tables": sum(row.chunk_type == "table" for row in scoped),
        }
    if complete:
        validate_intake_metadata_receipts(session, batch)
    return output


def existing_request(session: Session) -> RegulatoryDocumentIntakeReprepare | None:
    request = session.scalar(
        select(RegulatoryDocumentIntakeReprepare).where(
            RegulatoryDocumentIntakeReprepare.idempotency_key == KEY
        )
    )
    if request is not None:
        scope.require(
            request.intake_id == scope.BATCH
            and request.previous_labeling_run_id == OLD_RUN
            and request.expected_sha256 == APPROVED_PLAN_SHA
            and set(request.documents) == {scope.MAIN_DOCUMENT, scope.PDF_DOCUMENT}
            and all(
                entry.get("target_chunking_version") == "5"
                and entry.get("target_formatter_version") == 4
                and entry.get("target_preparation_policy") == policy("5")
                for entry in request.documents.values()
            ),
            "The existing logical-matrix request differs from its reviewed identity",
        )
    return request


def reprepare_plan(session: Session, batch: RegulatoryDocumentIntake) -> dict[str, Any]:
    require_files(session, batch)
    run = session.get_one(RegulatoryLabelingRun, OLD_RUN)
    require_old_run(run, batch)
    scope.require(
        batch.labeling_run_id == OLD_RUN
        and (batch.status, batch.stage) == ("partial", "labeling")
        and batch.review_sha256 is None
        and batch.lease_token is None
        and batch.lease_expires_at is None
        and batch.error == (run.error or "Labeling has unresolved or failed chunks"),
        "Normal autonomous recovery has not reconciled the original terminal failure",
    )
    scope.require(
        source_digest(session) == OLD_SOURCE_SHA,
        "The original complete canonical source changed",
    )
    old = latest_reprepare(session, batch.id)
    scope.require(
        old is not None
        and str(old.id) == scope.REPREPARE_REQUEST
        and old.status == "prepared"
        and old.expected_sha256 == scope.REPREPARE_REVIEW_SHA
        and old.error is None
        and all(
            entry.get("status") == "prepared"
            and entry.get("target_chunking_version") == "4"
            and entry.get("target_formatter_version") == 4
            and entry.get("target_preparation_policy") == policy("4")
            for entry in old.documents.values()
        ),
        "The existing completed v4 preparation checkpoint changed",
    )
    assert old is not None
    metadata = metadata_proof(session, batch, version="4", complete=True)
    binding = resolve_intake_labeling_provider_binding(
        session,
        batch.labeling_model_configuration_id,
        user=session.get_one(User, batch.created_by),
    )
    scope.require(
        binding.model_dump(mode="json") == run.provider_binding
        and run.model == binding.model,
        "The configured provider binding changed",
    )
    view = reprepare_snapshot(session, batch)
    scope.require(
        view.get("target_preparation_policy") == policy("5"),
        "Normal reprepare did not select v5",
    )
    scope.require(
        set(old.documents) == {scope.MAIN_DOCUMENT, scope.PDF_DOCUMENT}
        and all(
            all(
                current.get(key) == old.documents[current["document_id"]].get(key)
                for key in (
                    "original_file_id",
                    "source_asset_id",
                    "source_sha256",
                    "source_receipt",
                    "parent_document_id",
                    "role",
                    "relationships",
                )
            )
            for current in view["documents"]
        ),
        "The original raw file IDs, ingestion receipt or source relationships changed",
    )
    return {
        "action": "v5_reprepare_plan_rollback_only",
        "expected_sha256": view["expected_sha256"],
        "old_source_sha256": OLD_SOURCE_SHA,
        "old_run_id": str(run.id),
        "metadata": metadata,
        "target_preparation_policy": policy("5"),
        "provider_binding_sha256": publication_digest(run.provider_binding),
        "documents_before_sha256": publication_digest(view["documents"]),
        "provider_calls": 0,
    }


def request_once(
    session: Session, batch: RegulatoryDocumentIntake
) -> tuple[dict[str, Any], UUID | None]:
    scope.require(
        REPREPARE_ENABLED
        and FINAL_IMAGE_SHA is not None
        and APPROVED_PLAN_SHA is not None,
        "Mutation disabled pending final image, reviewed plan and explicit GO",
    )
    assert APPROVED_PLAN_SHA is not None
    existing = existing_request(session)
    if existing is not None:
        return {
            "action": "existing_v5_request_inspected_only",
            "request_id": str(existing.id),
            "status": existing.status,
            "enqueue_requested": False,
        }, None
    report = reprepare_plan(session, batch)
    scope.require(
        report["expected_sha256"] == APPROVED_PLAN_SHA,
        "The approved reprepare snapshot changed",
    )
    request = request_reprepare(
        session, batch, expected_sha256=APPROVED_PLAN_SHA, idempotency_key=KEY
    )
    scope.require(
        request.intake_id == scope.BATCH
        and request.previous_labeling_run_id == OLD_RUN
        and request.status == "queued"
        and request.expected_sha256 == APPROVED_PLAN_SHA
        and (batch.status, batch.stage) == ("preparing", "reprepare")
        and batch.labeling_run_id is None
        and batch.review_sha256 is None,
        "The normal explicit request did not produce its standard queued intent",
    )
    existing_request(session)
    scope.require(
        source_digest(session) == OLD_SOURCE_SHA,
        "The explicit request changed source content",
    )
    return {
        **report,
        "action": "normal_v5_reprepare_requested",
        "request_id": str(request.id),
        "before_audit_sha256": publication_digest(request.documents),
        "enqueue_requested": True,
    }, request.id


def progress(
    session: Session, batch: RegulatoryDocumentIntake, *, acceptance: bool
) -> dict[str, Any]:
    require_files(session, batch)
    request = existing_request(session)
    scope.require(request is not None, "The exact v5 request does not exist")
    assert request is not None
    report: dict[str, Any] = {
        "action": "v5_acceptance_read_only" if acceptance else "v5_progress_read_only",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "request_id": str(request.id),
        "request_status": request.status,
        "documents": {
            key: {
                "status": entry.get("status"),
                "target_chunking_version": entry.get("target_chunking_version"),
                "before_revision_count": len(entry.get("before_revision_ids", [])),
                "after_revision_count": len(entry.get("after_revision_ids", [])),
                "document_audit_sha256": publication_digest(entry),
            }
            for key, entry in request.documents.items()
        },
        "batch": {
            key: getattr(batch, key)
            for key in ("status", "stage", "labeling_run_id", "review_sha256", "error")
        },
        "source_sha256": source_digest(session),
        "provider_calls": 0,
        "approval_or_index_requested": False,
        "metadata_cache": scope.metadata_cache_progress(session, batch),
    }
    if request.status == "prepared":
        report["metadata"] = metadata_proof(
            session, batch, version="5", complete=acceptance
        )
    if batch.labeling_run_id is not None:
        run = session.get_one(RegulatoryLabelingRun, batch.labeling_run_id)
        old = session.get_one(RegulatoryLabelingRun, OLD_RUN)
        require_old_run(old, batch)
        scope.require(
            run.id not in {OLD_RUN, scope.LABELING_RUN}
            and run.idempotency_key
            == uuid5(scope.BATCH, "reprepare:" + str(request.id))
            and run.document_set_id == batch.document_set_id
            and run.model_configuration_id == batch.labeling_model_configuration_id
            and run.requested_by_id == batch.created_by
            and set(run.file_ids) == {scope.MAIN_FILE, scope.PDF_FILE}
            and run.provider_binding == old.provider_binding
            and run.taxonomy_id == old.taxonomy_id
            and run.model == old.model,
            "The observed run is not the exact normal worker-created v5 scope",
        )
        report["run"] = {
            key: getattr(run, key)
            for key in (
                "id",
                "status",
                "stage",
                "total_chunks",
                "completed_chunks",
                "failed_chunks",
                "stale_chunks",
                "derived_chunks",
                "unresolved_derived_chunks",
                "error",
            )
        }
        report["items"] = scope.rows(
            session,
            "SELECT status,count(*) AS count FROM regulatory_labeling_item WHERE run_id=:run GROUP BY status",
            run=run.id,
        )
        report["derived"] = scope.rows(
            session,
            "SELECT resolution,count(*) AS count FROM regulatory_derived_label_projection WHERE run_id=:run GROUP BY resolution",
            run=run.id,
        )
        if acceptance:
            scope.require(
                request.status == "prepared"
                and all(
                    entry.get("status") == "prepared"
                    for entry in request.documents.values()
                ),
                "The normal reprepare checkpoint is incomplete",
            )
            report["review"] = acceptance_proof(session, batch, run)
    scope.require(
        not acceptance or "review" in report, "No autonomous complete review exists yet"
    )
    return report


def acceptance_proof(
    session: Session, batch: RegulatoryDocumentIntake, run: RegulatoryLabelingRun
) -> dict[str, Any]:
    scope.require(
        (batch.status, batch.stage) == ("ready", "review")
        and batch.error is None
        and bool(batch.review_sha256)
        and run.id not in {OLD_RUN, scope.LABELING_RUN}
        and (run.status, run.stage) == ("completed", "finished")
        and run.error is None
        and run.failed_chunks == run.stale_chunks == run.unresolved_derived_chunks == 0
        and run.completed_chunks == run.total_chunks
        and not run.cancel_requested
        and run.lease_token is None
        and run.lease_expires_at is None,
        "Autonomous preparation has not reached a complete valid review",
    )
    binding = resolve_intake_labeling_provider_binding(
        session,
        batch.labeling_model_configuration_id,
        user=session.get_one(User, batch.created_by),
    )
    scope.require(
        binding.model_dump(mode="json") == run.provider_binding
        and run.model == binding.model,
        "The final configured provider binding changed",
    )
    rows = source_rows(session)
    canonical = {
        chunk.chunk_id
        for file in (scope.MAIN_FILE, scope.PDF_FILE)
        for chunk in metadata_chunks(
            [row for row in rows if str(row.user_file_id) == file]
        )
    }
    proofs = load_intake_structural_reader_proofs(session, intake_id=batch.id)
    scope.require(len(proofs) == 11, "Native chapter proof scope changed")
    provider_ids = canonical - set(proofs)
    items = list(
        session.scalars(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == run.id
            )
        )
    )
    scope.require(
        {item.regulatory_chunk_id for item in items} == provider_ids
        and all(item.status == "completed" and item.error is None for item in items)
        and current_labeling_items_for_search(session, run=run, items=items)
        == provider_ids
        and (run.total_chunks, run.derived_chunks)
        == (len(provider_ids), len(rows) - len(provider_ids)),
        "Current provider source/context coverage is incomplete",
    )
    derived = list(
        session.scalars(
            select(RegulatoryDerivedLabelProjection).where(
                RegulatoryDerivedLabelProjection.run_id == run.id
            )
        )
    )
    scope.require(
        len(derived) == run.derived_chunks
        and {target.regulatory_chunk_id for target in derived}
        == {row.id for row in rows} - provider_ids
        and all(
            target.resolution == "lineage"
            and str(target.user_file_id) in {scope.MAIN_FILE, scope.PDF_FILE}
            and target.unresolved_reason is None
            and current_derived_label_sources_for_search(
                session, run=run, target=target
            )
            for target in derived
        ),
        "Derived current source authority is incomplete",
    )
    authority = labeling_review_authority(session, run.id)
    native_authority = authority.get("structural_readers") if authority else None
    digest = intake.review_digest(session, batch)
    scope.require(
        digest == batch.review_sha256
        and isinstance(native_authority, list)
        and len(native_authority) == 11,
        "The persisted autonomous review digest or authority is stale",
    )
    files = {
        UUID(key): session.get_one(UserFile, UUID(key))
        for key in (scope.MAIN_FILE, scope.PDF_FILE)
    }
    validate_intake_edit_budgets(
        session,
        files,
        {key: [row for row in rows if row.user_file_id == key] for key in files},
    )
    scope.require(
        V5_PDF_TABLES is not None
        and sum(
            row.chunk_type == "table" and str(row.user_file_id) == scope.PDF_FILE
            for row in rows
        )
        == V5_PDF_TABLES,
        "The frozen logical whole-table coverage changed",
    )
    return {
        "review_sha256": digest,
        "authority_sha256": publication_digest(authority),
        "current_provider_receipts": len(items),
        "resolved_derived_receipts": len(derived),
        "current_canonical_metadata_receipts": len(canonical),
        "all_source_chunks": len(rows),
        "native_chapter_readers": len(proofs),
        "edit_budget_validated": True,
    }


def terminal_status(
    session: Session, batch: RegulatoryDocumentIntake
) -> dict[str, Any]:
    """Observe the existing periodic handoff without calling its transition."""
    require_files(session, batch)
    old = session.get_one(RegulatoryLabelingRun, OLD_RUN)
    require_old_run(old, batch)
    scope.require(
        batch.labeling_run_id == OLD_RUN and source_digest(session) == OLD_SOURCE_SHA,
        "The original frozen failure scope changed",
    )
    return {
        "action": "terminal_handoff_status_read_only",
        "batch": {
            key: getattr(batch, key)
            for key in ("status", "stage", "labeling_run_id", "review_sha256", "error")
        },
        "autonomous_terminal_reconciled": (batch.status, batch.stage)
        == ("partial", "labeling"),
        "metadata": metadata_proof(session, batch, version="4", complete=True),
        "provider_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-image-sha", required=True)
    parser.add_argument("--observed-image-sha", required=True)
    parser.add_argument(
        "--operation",
        choices=("status", "plan", "request", "progress", "acceptance", "canary"),
        required=True,
    )
    parser.add_argument("--input-path", type=Path)
    parser.add_argument("--expected-input-sha")
    args = parser.parse_args()
    scope.require(
        bool(re.fullmatch(r"[0-9a-f]{40}", args.expected_image_sha))
        and args.expected_image_sha == args.observed_image_sha
        and (
            args.operation == "status"
            or (
                FINAL_IMAGE_SHA is not None
                and args.expected_image_sha == FINAL_IMAGE_SHA
            )
        ),
        "Only the pinned final deployed DEV image is eligible",
    )
    scope.require(
        POSTGRES_DB == "customs-regulations-dev"
        and scope.config.REGULATORY_ANNEX_ENVIRONMENT == "dev"
        and get_current_tenant_id() == "public",
        "Only the exact DEV tenant is eligible",
    )
    set_is_ee_based_on_env_variable()
    SqlEngine.init_engine(pool_size=2, max_overflow=0)
    if args.operation == "canary":
        scope.require(
            args.input_path is not None and args.expected_input_sha is not None,
            "The strict v5 canary requires its exact fixture",
        )
        report = scope.strict_v4_canary(
            args.input_path, args.expected_input_sha, chunking_version="5"
        )
        print(json.dumps(report, ensure_ascii=False, default=str, indent=2))
        return
    enqueue_id = None
    with get_session_with_current_tenant() as session:
        # Normal preview uses FOR UPDATE. Roll back its transaction, never a GET.
        if args.operation in {"status", "progress", "acceptance"}:
            session.execute(text("SET TRANSACTION READ ONLY"))
        session.execute(text("SET LOCAL statement_timeout = '20000'"))
        session.execute(text("SET LOCAL lock_timeout = '3000'"))
        batch = session.get_one(RegulatoryDocumentIntake, scope.BATCH)
        if args.operation == "request":
            report, enqueue_id = request_once(session, batch)
            session.commit() if enqueue_id is not None else session.rollback()
        elif args.operation == "plan":
            report = reprepare_plan(session, batch)
            session.rollback()
        elif args.operation == "status":
            report = terminal_status(session, batch)
            session.rollback()
        else:
            report = progress(session, batch, acceptance=args.operation == "acceptance")
            session.rollback()
    if enqueue_id is not None:
        from onyx.background.celery.tasks.regulatory_amendments.document_intake import (
            enqueue_intake_batch,
        )

        enqueue_intake_batch(
            batch_id=scope.BATCH, tenant_id="public", reprepare_request_id=enqueue_id
        )
    for item in [report.get("batch", {}), report.get("run", {})]:
        if isinstance(item.get("error"), str) and scope.SECRET_LINE.search(
            item["error"]
        ):
            item["error"] = "[redacted diagnostic]"
    print(json.dumps(report, ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
