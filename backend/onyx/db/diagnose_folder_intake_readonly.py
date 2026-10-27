"""Bounded, transaction-enforced read-only DEV intake delivery inventory."""

import json
import os
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


def main() -> None:
    if os.environ.get("POSTGRES_DB") != "customs-regulations-dev":
        raise RuntimeError("Expected the explicitly named DEV database")
    engine = create_engine(
        URL.create(
            "postgresql+psycopg2",
            username=os.environ.get("POSTGRES_USER", "postgres"),
            password=os.environ.get("POSTGRES_PASSWORD"),
            host=os.environ.get("POSTGRES_HOST"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            database="customs-regulations-dev",
        ),
        connect_args={
            "connect_timeout": 5,
            "options": "-c default_transaction_read_only=on -c search_path=public "
            "-c statement_timeout=30000 -c application_name=folder_intake_readonly",
            **(
                {"sslmode": os.environ["POSTGRES_SSLMODE"]}
                if os.environ.get("POSTGRES_SSLMODE")
                else {}
            ),
        },
        pool_size=1,
        max_overflow=0,
    )
    queries = {
        "PROTECTED_CORPUS": """
            WITH protected AS (
        SELECT f.id FROM user_file f JOIN document_set__user_file m
          ON m.user_file_id = f.id AND m.document_set_id = 15
         WHERE f.created_at < '2026-10-27T14:49:00Z'
           AND f.id NOT IN ('10e927dc-a124-4c6a-85b1-e3619b61b6d3',
                            'a3c6e8cf-93a9-460b-9edf-a9cdd4ba05a9',
                            '16620cfa-0bf4-4540-b8c3-26ad5210c6b5')
    )
            SELECT (SELECT count(*) FROM protected) AS files,
                   (SELECT md5(string_agg(md5(to_jsonb(f)::text), '' ORDER BY f.id))
                      FROM user_file f JOIN protected p ON p.id=f.id) AS file_rows_md5,
                   count(c.id) AS canonical_rows,
                   md5(string_agg(md5(to_jsonb(c)::text), '' ORDER BY c.id)) AS canonical_rows_md5
              FROM regulatory_chunk c JOIN protected p ON p.id=c.user_file_id
        """,
        "PROTECTED_DERIVED_LABELS": """
            WITH protected AS (
        SELECT f.id FROM user_file f JOIN document_set__user_file m
          ON m.user_file_id = f.id AND m.document_set_id = 15
         WHERE f.created_at < '2026-10-27T14:49:00Z'
           AND f.id NOT IN ('10e927dc-a124-4c6a-85b1-e3619b61b6d3',
                            'a3c6e8cf-93a9-460b-9edf-a9cdd4ba05a9',
                            '16620cfa-0bf4-4540-b8c3-26ad5210c6b5')
    )
            SELECT count(*) AS rows,
                   md5(string_agg(md5(to_jsonb(d)::text), '' ORDER BY d.id)) AS rows_md5
              FROM regulatory_derived_label_projection d
              JOIN protected p ON p.id=d.user_file_id
        """,
        "SOURCE_PACKAGES": """
            SELECT id, status, environment, created_at, updated_at, asset_count,
                   total_bytes, lease_token IS NOT NULL AS claimed, lease_expires_at,
                   input_spec, issues,
                   (SELECT count(*) FROM regulatory_source_asset a
                    WHERE a.package_id = p.id) AS persisted_assets
              FROM amendment_source_package p
             WHERE document_set_id = 15 AND created_at >= '2026-10-27T14:00:00Z'
             ORDER BY created_at LIMIT 30
        """,
        "EXISTING_BATCH": """
            SELECT id, source_package_id, status, stage, error, labeling_run_id,
                   created_at, updated_at, lease_token IS NOT NULL AS claimed,
                   lease_expires_at
              FROM regulatory_document_intake
             WHERE id = '7b7a8362-f9a7-465a-a582-d09042165728'
        """,
        "SOURCE_RECOVERY_SCHEDULER": """
            SELECT application_name, state, wait_event_type, wait_event,
                   now() - query_start AS query_age,
                   left(query, 160) AS statement_prefix
              FROM pg_stat_activity
             WHERE datname = current_database()
               AND application_name ILIKE '%annex%'
             ORDER BY query_start DESC LIMIT 20
        """,
    }
    with engine.connect() as connection:
        connection.execute(text("SET TRANSACTION READ ONLY"))
        for label, query in queries.items():
            rows: list[dict[str, Any]] = [
                dict(row._mapping) for row in connection.execute(text(query))
            ]
            print(label, json.dumps(rows, ensure_ascii=False, default=str), flush=True)
        connection.rollback()


if __name__ == "__main__":
    main()
