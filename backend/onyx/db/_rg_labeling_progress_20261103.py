"""One bounded read-only progress snapshot for the source-verified intake retry."""

import argparse
import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid5

import _new_document_rg_operator as scope
import _rg_labeling_retry_20261102 as retry
from sqlalchemy import text

from onyx.configs.app_configs import POSTGRES_DB
from onyx.db.engine.sql_engine import SqlEngine, get_session_with_current_tenant
from onyx.db.models import RegulatoryLabelingRun
from onyx.utils.variable_functionality import set_is_ee_based_on_env_variable
from shared_configs.contextvars import get_current_tenant_id

EXPECTED_IMAGE_SHA = "2fc69ee1ad9ca619c4266a8b2c08a2ef44523fe3"
EXPECTED_SOURCE_SHA256 = "de0ffa6d4ee73b11499026af190690fe29760d77a51cfd8bf069601dfa96fd89"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-image-sha", required=True)
    parser.add_argument("--observed-image-sha", required=True)
    args = parser.parse_args()
    scope.require(
        args.expected_image_sha == args.observed_image_sha == EXPECTED_IMAGE_SHA,
        "Progress requires the approved exact release",
    )
    scope.require(
        EXPECTED_SOURCE_SHA256 is not None,
        "Progress requires the fresh reviewed source fingerprint",
    )
    scope.require(
        POSTGRES_DB == "customs-regulations-dev"
        and scope.config.REGULATORY_ANNEX_ENVIRONMENT == "dev"
        and get_current_tenant_id() == "public",
        "Only the exact DEV tenant is eligible",
    )
    set_is_ee_based_on_env_variable()
    SqlEngine.init_engine(pool_size=2, max_overflow=0)
    with get_session_with_current_tenant() as session:
        session.execute(text("SET TRANSACTION READ ONLY"))
        session.execute(text("SET LOCAL statement_timeout = '20000'"))
        batch = session.get_one(scope.RegulatoryDocumentIntake, scope.BATCH)
        scope.verify_scope(session, batch)
        scope.unpublished_epoch_scope(session)
        scope.require(
            retry.canonical_source_sha256(session) == EXPECTED_SOURCE_SHA256,
            "The monitored canonical source changed",
        )
        old = session.get_one(RegulatoryLabelingRun, scope.LABELING_RUN)
        retry.require_failed_run(old)
        run = session.get_one(RegulatoryLabelingRun, batch.labeling_run_id)
        if run.id != old.id:
            scope.require(
                run.idempotency_key == uuid5(scope.BATCH, str(old.id))
                and (run.total_chunks, run.derived_chunks) == (651, 46)
                and run.provider_binding == old.provider_binding
                and run.taxonomy_id == old.taxonomy_id
                and set(run.file_ids) == {scope.MAIN_FILE, scope.PDF_FILE},
                "The worker-created labeling scope changed",
            )
        report: dict[str, Any] = {
            "operator_action": "labeling_progress_read_only",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "expected_image_sha": EXPECTED_IMAGE_SHA,
            "canonical_source_sha256": EXPECTED_SOURCE_SHA256,
            "batch": {
                key: getattr(batch, key)
                for key in (
                    "status",
                    "stage",
                    "labeling_run_id",
                    "review_sha256",
                    "error",
                )
            },
            "run": {
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
                    "updated_at",
                    "error",
                )
            },
            "items": scope.rows(
                session,
                "SELECT status,count(*) AS count FROM regulatory_labeling_item WHERE run_id=:run GROUP BY status ORDER BY status",
                run=run.id,
            ),
            "derived": scope.rows(
                session,
                "SELECT resolution,count(*) AS count FROM regulatory_derived_label_projection WHERE run_id=:run GROUP BY resolution ORDER BY resolution",
                run=run.id,
            ),
            "semantic_receipts": scope.rows(
                session,
                "SELECT status,count(*) AS count FROM regulatory_labeling_item WHERE run_id=:run AND source_snapshot ? 'intake_abstention_repair' GROUP BY status ORDER BY status",
                run=run.id,
            ),
            "failed_items": scope.rows(
                session,
                "SELECT i.regulatory_chunk_id,i.user_file_id,c.position,c.chunk_type,left(i.error,500) AS error FROM regulatory_labeling_item i JOIN regulatory_chunk c ON c.id=i.regulatory_chunk_id WHERE i.run_id=:run AND i.status IN ('failed','stale') ORDER BY c.user_file_id,c.position LIMIT 64",
                run=run.id,
            ),
            "approval_publication_index_or_provider_requested": False,
        }
        session.rollback()
    for item in [report["batch"], report["run"], *report["failed_items"]]:
        error = item.get("error")
        if isinstance(error, str) and scope.SECRET_LINE.search(error):
            item["error"] = "[redacted diagnostic]"
    print(json.dumps(report, ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
