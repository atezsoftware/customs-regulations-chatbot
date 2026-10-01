"""Read-only, bounded DEV inventory for the five reported amendment batches."""

import hashlib
import json
import os
from pathlib import Path

from dotenv import dotenv_values
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


def main() -> None:
    local_values = dotenv_values(
        Path("deployment/docker_compose/.env.remote-db.local")
    )
    values = {
        "DB": os.environ.get("POSTGRES_DB") or local_values.get("REMOTE_POSTGRES_DB"),
        "USER": os.environ.get("POSTGRES_USER") or local_values.get("REMOTE_POSTGRES_USER"),
        "PASSWORD": os.environ.get("POSTGRES_PASSWORD") or local_values.get("REMOTE_POSTGRES_PASSWORD"),
        "HOST": os.environ.get("POSTGRES_HOST") or local_values.get("REMOTE_POSTGRES_HOST"),
        "PORT": os.environ.get("POSTGRES_PORT") or local_values.get("REMOTE_POSTGRES_PORT"),
        "SSLMODE": os.environ.get("POSTGRES_SSLMODE") or local_values.get("REMOTE_POSTGRES_SSLMODE"),
    }
    if values["DB"] != "customs-regulations-dev":
        raise RuntimeError("Expected explicit DEV database")
    engine = create_engine(
        URL.create(
            "postgresql+psycopg2",
            username=values["USER"] or "postgres",
            password=values["PASSWORD"],
            host=values["HOST"],
            port=int(values["PORT"] or "5432"),
            database="customs-regulations-dev",
        ),
        connect_args={
            "connect_timeout": 5,
            "options": "-c default_transaction_read_only=on -c search_path=public "
            "-c statement_timeout=30000 -c application_name=amendment_299_310_audit",
            **(
                {"sslmode": values["SSLMODE"]}
                if values["SSLMODE"]
                else {}
            ),
        },
        pool_size=1,
        max_overflow=0,
    )
    with engine.connect() as connection:
        batches = connection.execute(
            text(
                "SELECT id, document_set_id, status, stage, instruction_count, "
                "processed_instruction_count, unmatched_instructions, "
                "segmented_instructions, analysis_log, user_file_ids, raw_text, "
                "source_package_id, created_at, updated_at "
                "FROM amendment_batch WHERE id IN (299,302,306,309,310) ORDER BY id"
            )
        )
        for row in batches:
            batch = row._mapping
            print(
                "BATCH",
                json.dumps(
                    {
                        "id": batch["id"],
                        "set": batch["document_set_id"],
                        "status": batch["status"],
                        "stage": batch["stage"],
                        "instructions": batch["instruction_count"],
                        "processed": batch["processed_instruction_count"],
                        "unmatched": batch["unmatched_instructions"],
                        "segmented_count": len(batch["segmented_instructions"] or []),
                        "log_tail": (batch["analysis_log"] or [])[-4:],
                        "file_count": len(batch["user_file_ids"] or []),
                        "raw_sha256": hashlib.sha256(batch["raw_text"].encode()).hexdigest(),
                        "source_package_id": str(batch["source_package_id"]),
                        "updated_at": str(batch["updated_at"]),
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            )
            proposals = connection.execute(
                text(
                    "SELECT id, instruction_index, instruction_text, status, "
                    "old_chunk_id, applied_new_chunk_id, applied_new_chunk_ids, "
                    "approval_indexing_job_id, approval_error, new_chunk_draft, "
                    "chunk_changes, updated_at FROM amendment_proposal "
                    "WHERE batch_id=:batch_id ORDER BY instruction_index"
                ),
                {"batch_id": batch["id"]},
            )
            for proposal_row in proposals:
                proposal = proposal_row._mapping
                draft = proposal["new_chunk_draft"] or {}
                print(
                    "PROPOSAL",
                    json.dumps(
                        {
                            "batch": batch["id"],
                            "id": proposal["id"],
                            "instruction_index": proposal["instruction_index"],
                            "instruction_excerpt": proposal["instruction_text"][:300],
                            "status": proposal["status"],
                            "old_chunk_id": proposal["old_chunk_id"],
                            "applied_new_chunk_id": proposal["applied_new_chunk_id"],
                            "applied_new_chunk_ids": proposal["applied_new_chunk_ids"],
                            "indexing_job_id": str(proposal["approval_indexing_job_id"]),
                            "approval_error": proposal["approval_error"],
                            "file_id": draft.get("user_file_id"),
                            "draft_text_sha256": hashlib.sha256(
                                str(draft.get("text", "")).encode()
                            ).hexdigest(),
                            "draft_article": (draft.get("metadata") or {}).get("article_no"),
                            "insertion_order": draft.get("insertion_order"),
                            "chunk_change_count": len(proposal["chunk_changes"] or []),
                            "updated_at": str(proposal["updated_at"]),
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                )
    engine.dispose()


if __name__ == "__main__":
    main()
