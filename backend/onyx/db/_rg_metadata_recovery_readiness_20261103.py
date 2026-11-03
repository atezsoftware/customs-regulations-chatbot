"""Read-only post-release proof for recovery schema and the completed incident."""

import argparse
import json
import re
from typing import Any

import _new_document_rg_operator as scope
import _rg_reprepare_v5_20261103 as ready
from sqlalchemy import text
from sqlalchemy.orm import Session

from onyx.configs.app_configs import POSTGRES_DB
from onyx.db.engine.sql_engine import SqlEngine, get_session_with_current_tenant
from onyx.utils.variable_functionality import set_is_ee_based_on_env_variable
from shared_configs.contextvars import get_current_tenant_id

RELEASE_SHA = "f582d8de3752b393baefd832ef48ab677c820466"
SCHEMA_HEAD = "c3b4ea5ee175"
TASK = "regulatory_document_intake_metadata_recover"
SOURCE_SHA = "d76ea12a2a3e2755f5c2ef81ca0e10598b67acce2c157df5afef480eabdca06c"
REVIEW_SHA = "805db0ed77c41a4027b7b35d1bc61dbcff73b70931261111282dc204f248ea72"


def validate_schema_shape(
    columns: list[dict[str, Any]], constraints: list[dict[str, Any]]
) -> None:
    actual = {
        row["column_name"]: (row["udt_name"], row["is_nullable"], row["column_default"])
        for row in columns
    }
    scope.require(
        actual
        == {
            "metadata_checkpoint": ("jsonb", "YES", None),
            "metadata_recovery_count": ("int4", "NO", "0"),
        },
        "Recovery column types, nullability or server defaults differ",
    )
    expression = (
        re.sub(r"[\s()]", "", constraints[0]["expression"]).lower()
        if len(constraints) == 1
        else ""
    )
    scope.require(
        len(constraints) == 1
        and constraints[0]["validated"] is True
        and expression == "metadata_recovery_count>=0andmetadata_recovery_count<=3",
        "Recovery count requires the validated exact inclusive 0..3 check",
    )


def schema_proof(session: Session) -> dict[str, Any]:
    heads = scope.rows(session, "SELECT version_num FROM alembic_version")
    scope.require(
        heads == [{"version_num": SCHEMA_HEAD}], "Unexpected post-release schema head"
    )
    columns = scope.rows(
        session,
        """SELECT column_name,udt_name,is_nullable,column_default
        FROM information_schema.columns WHERE table_schema=current_schema()
        AND table_name='regulatory_document_intake'
        AND column_name IN ('metadata_checkpoint','metadata_recovery_count') ORDER BY column_name""",
    )
    constraints = scope.rows(
        session,
        """SELECT c.convalidated AS validated,
        pg_get_expr(c.conbin,c.conrelid) AS expression FROM pg_constraint c
        JOIN pg_class t ON t.oid=c.conrelid JOIN pg_namespace n ON n.oid=t.relnamespace
        WHERE n.nspname=current_schema() AND t.relname='regulatory_document_intake'
        AND c.conname='ck_regulatory_intake_metadata_recovery_count' AND c.contype='c'""",
    )
    validate_schema_shape(columns, constraints)
    legacy = scope.rows(
        session,
        """SELECT metadata_checkpoint IS NULL AS checkpoint_absent,
        metadata_recovery_count FROM regulatory_document_intake WHERE id=:batch""",
        batch=scope.BATCH,
    )
    scope.require(
        legacy == [{"checkpoint_absent": True, "metadata_recovery_count": 0}],
        "The completed legacy incident was changed by recovery deployment",
    )
    return {
        "schema_head": SCHEMA_HEAD,
        "columns": columns,
        "count_constraint": constraints[0],
        "completed_legacy_checkpoint_absent": True,
        "completed_legacy_recovery_count": 0,
    }


def registered_runtime(observed_pod: str) -> dict[str, Any]:
    from onyx.background.celery.versioned_apps.client import app

    inspector = app.control.inspect(timeout=3.0)
    queues, registered = inspector.active_queues() or {}, inspector.registered() or {}
    expected = scope.config.source_queue_name()
    consumers = []
    for worker, worker_queues in queues.items():
        if worker.rsplit("@", 1)[-1] != observed_pod or expected not in {
            row.get("name") for row in worker_queues
        }:
            continue
        names = {str(name).split(" ", 1)[0] for name in registered.get(worker, [])}
        consumers.append(
            {
                "worker": worker,
                "consumes_expected_queue": True,
                "metadata_recovery_task_registered": TASK in names,
            }
        )
    scope.require(
        bool(consumers)
        and all(row["metadata_recovery_task_registered"] for row in consumers),
        "The exact ready source worker does not advertise metadata recovery",
    )
    return {
        "observed_pod": observed_pod,
        "expected_source_queue": expected,
        "task_name": TASK,
        "consumers": consumers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-image-sha", required=True)
    parser.add_argument("--observed-image-sha", required=True)
    parser.add_argument("--observed-pod", required=True)
    args = parser.parse_args()
    scope.require(
        RELEASE_SHA is not None
        and args.expected_image_sha == args.observed_image_sha == RELEASE_SHA,
        "Post-deploy acceptance requires the approved exact recovery release",
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
        report = ready.progress(session, batch, acceptance=True)
        scope.require(
            report["source_sha256"] == SOURCE_SHA
            and report["review"]["review_sha256"] == REVIEW_SHA,
            "The completed source or autonomous review changed during deployment",
        )
        report["recovery_schema"] = schema_proof(session)
        session.rollback()
    report["recovery_worker_runtime"] = registered_runtime(args.observed_pod)
    report["action"] = "post_deploy_recovery_acceptance_read_only"
    print(json.dumps(report, ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
