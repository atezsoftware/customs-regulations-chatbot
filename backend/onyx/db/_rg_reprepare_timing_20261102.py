"""Read only the exact incident's retained revision timing from DEV."""

import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

ROOT = Path("/Users/kubilaypayci/Desktop/atez/customs-regulations-chatbot")
REQUEST = "a03a7101-f9a7-46cf-bbda-831d96e64865"
BATCH = "f32df3a9-4644-4fd3-be07-6aaa966d3107"
config = dotenv_values(ROOT / "deployment/docker_compose/.env.remote-db.local")
assert config["REMOTE_POSTGRES_DB"] == "customs-regulations-dev"
engine = create_engine(
    URL.create(
        "postgresql+psycopg2",
        username=config["REMOTE_POSTGRES_USER"],
        password=config["REMOTE_POSTGRES_PASSWORD"],
        host=config["REMOTE_POSTGRES_HOST"],
        port=int(config["REMOTE_POSTGRES_PORT"] or "5432"),
        database="customs-regulations-dev",
    ),
    connect_args={
        "connect_timeout": 5,
        "sslmode": config.get("REMOTE_POSTGRES_SSLMODE") or "require",
        "options": "-c default_transaction_read_only=on -c search_path=public -c statement_timeout=10000 -c lock_timeout=2000 -c application_name=rg_intake_timing_readonly_20261102",
    },
    pool_size=1,
    max_overflow=0,
)
report: dict[str, object] = {
    "observed_at": datetime.now(timezone.utc).isoformat(),
    "request_id": REQUEST,
    "database_writes": False,
}
with engine.connect() as connection:
    assert (
        connection.execute(text("SELECT current_database()")).scalar_one()
        == "customs-regulations-dev"
    )
    track = connection.execute(text("SHOW track_commit_timestamp")).scalar_one()
    report["track_commit_timestamp"] = track
    report["request"] = dict(
        connection.execute(
            text(
                "SELECT id,intake_id,status,error,created_at,xmin::text AS last_update_xid FROM regulatory_document_intake_reprepare WHERE id=:request AND intake_id=:batch"
            ),
            {"request": REQUEST, "batch": BATCH},
        )
        .mappings()
        .one()
    )
    documents = [
        dict(row)
        for row in connection.execute(
            text("""
        SELECT entry.key AS document_id,entry.value->>'user_file_id' AS user_file_id,
               entry.value->>'status' AS checkpoint_status,count(revision.id) AS after_revision_count,
               min(revision.created_at) AS earliest_revision_transaction_started_at,
               max(revision.created_at) AS latest_revision_transaction_started_at,
               array_agg(DISTINCT revision.xmin::text) FILTER (WHERE revision.id IS NOT NULL) AS revision_xids
        FROM regulatory_document_intake_reprepare request
        CROSS JOIN LATERAL jsonb_each(request.documents) entry
        LEFT JOIN LATERAL jsonb_array_elements_text(entry.value->'after_revision_ids') id ON true
        LEFT JOIN regulatory_canonical_revision revision ON revision.id=id.value::uuid
             AND revision.user_file_id=(entry.value->>'user_file_id')::uuid
        WHERE request.id=:request AND request.intake_id=:batch
        GROUP BY entry.key,entry.value->>'user_file_id',entry.value->>'status'
        ORDER BY entry.key
    """),
            {"request": REQUEST, "batch": BATCH},
        ).mappings()
    ]
    report["documents"] = documents
    if track == "on":
        xids = {value for row in documents for value in (row["revision_xids"] or [])}
        report["revision_commit_timestamps"] = {
            value: connection.execute(
                text("SELECT pg_xact_commit_timestamp(CAST(:xid AS xid))"),
                {"xid": value},
            ).scalar_one()
            for value in sorted(xids)
        }
    report["timestamp_semantics"] = (
        "Revision created_at is transaction start, not commit; no commit time is inferred if tracking is disabled."
    )
    connection.rollback()
engine.dispose()
output = ROOT / "plans/new-document-rg-20261102/reprepare-v4-revision-timing.json"
output.write_text(json.dumps(report, ensure_ascii=False, default=str, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, default=str, indent=2))
