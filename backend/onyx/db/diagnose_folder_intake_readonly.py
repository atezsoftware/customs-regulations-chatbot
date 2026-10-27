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
        "OWNED_INDEX_PROGRESS": """
            SELECT f.id, f.status AS file_status, r.attempt_id,
                   r.status AS repair_status, r.updated_at AS repair_updated_at,
                   r.failure_code, j.id AS job_id, j.status AS job_status,
                   j.stage AS job_stage, j.heartbeat_at, j.next_retry_at,
                   j.error_code, j.error_message,
                   j.remote_vertex_job_name IS NOT NULL AS has_remote_vertex_job,
                   j.remote_openrouter_batch_id IS NOT NULL AS has_remote_openrouter_job
              FROM user_file f LEFT JOIN user_file_projection_repair r ON r.user_file_id=f.id
              LEFT JOIN regulatory_indexing_job j ON j.user_file_id=f.id
             WHERE f.id IN ('10e927dc-a124-4c6a-85b1-e3619b61b6d3',
                            'a3c6e8cf-93a9-460b-9edf-a9cdd4ba05a9')
             ORDER BY f.id,j.created_at DESC LIMIT 12
        """,
        "FROZEN_SOURCE_RAW_DUPLICATES": """
            WITH incoming AS (
                SELECT id, package_id, sha256, display_name FROM regulatory_source_asset
                 WHERE package_id IN (
                    '8a4c1bb2-66f0-4a6d-bb3b-dfb3adf4b73b',
                    '0e8a9538-c59f-4451-a513-3c9ce71971a3',
                    'c85b6a1d-0dca-4968-b4a5-f4011fc32a7c',
                    '08dd6e41-7c67-4dfa-9c64-48a0d3d33a3a',
                    'f71bd843-0827-4264-976b-911d00930022',
                    '93064b06-86f1-4b3d-aaa4-279471e6e1f8',
                    '0d558987-0ebb-49c1-9a35-fe9e619bde8e',
                    'aae7aa45-1a84-4838-bd13-ae3c1ce632db')
            ), old_receipts AS (
                SELECT f.id, f.name, p.original_ingestion_receipt->>'raw_sha256' AS sha256
                  FROM user_file f JOIN document_set__user_file m
                    ON m.user_file_id=f.id AND m.document_set_id=15
                  JOIN regulatory_file_publication p ON p.user_file_id=f.id
                 WHERE f.created_at < '2026-10-27T14:49:00Z'
                   AND p.original_ingestion_receipt->>'raw_sha256' IS NOT NULL
            )
            SELECT a.id AS source_asset_id, a.package_id, a.display_name, a.sha256,
                   (SELECT count(*) FROM old_receipts) AS comparable_old_originals,
                   coalesce(jsonb_agg(jsonb_build_object('file_id',o.id,'name',o.name)
                            ORDER BY o.id) FILTER (WHERE o.id IS NOT NULL),'[]') AS old_matches
              FROM incoming a LEFT JOIN old_receipts o ON o.sha256=a.sha256
             GROUP BY a.id,a.package_id,a.display_name,a.sha256 ORDER BY a.id
        """,
        "RECENT_DIRECT_CACHE_COUNTS": """
            SELECT split_part(file_id, '/', 2) AS provider_binding,
                   count(*) FILTER (WHERE file_id LIKE '%/manifest.json') AS request_groups,
                   count(*) FILTER (WHERE file_id NOT LIKE '%/manifest.json') AS result_files,
                   min(created_at) AS first_created, max(updated_at) AS last_updated
              FROM file_record
             WHERE file_id LIKE 'regulatory-labeling-direct/%'
               AND created_at >= '2026-10-27T15:58:30.872604Z'
             GROUP BY split_part(file_id, '/', 2)
        """,
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
             WHERE id IN ('7b7a8362-f9a7-465a-a582-d09042165728',
                          'f8d84bb5-411a-4f7e-9105-bb7b11f45d34')
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
            if os.environ.get("FOLDER_PROGRESS_ONLY") == "1" and label.startswith(
                "PROTECTED"
            ):
                continue
            rows: list[dict[str, Any]] = [
                dict(row._mapping) for row in connection.execute(text(query))
            ]
            print(label, json.dumps(rows, ensure_ascii=False, default=str), flush=True)
        connection.rollback()


if __name__ == "__main__":
    main()
