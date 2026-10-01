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
                if proposal["id"] == 470:
                    from onyx.regulatory.amendments.draft_integrity import (
                        validate_explicit_replacement_texts,
                    )
                    for index, change in enumerate(proposal["chunk_changes"] or []):
                        old = change.get("old_chunk_snapshot") or {}
                        new = change.get("new_chunk_draft") or {}
                        try:
                            validate_explicit_replacement_texts(
                                list(change.get("instruction_texts") or []),
                                str(new.get("text", "")),
                            )
                            integrity = "pass"
                        except ValueError as error:
                            integrity = str(error)
                        print(
                            "CHANGE_470",
                            json.dumps(
                                {
                                    "index": index,
                                    "old_id": change.get("old_chunk_id"),
                                    "old_file": old.get("user_file_id"),
                                    "new_file": new.get("user_file_id"),
                                    "old_position": old.get("position"),
                                    "new_position": new.get("position"),
                                    "old_sha256": hashlib.sha256(
                                        str(old.get("text", "")).encode()
                                    ).hexdigest(),
                                    "new_sha256": hashlib.sha256(
                                        str(new.get("text", "")).encode()
                                    ).hexdigest(),
                                    "heading": new.get("heading_path"),
                                    "replacement_integrity": integrity,
                                    "instruction_text_count": len(
                                        change.get("instruction_texts") or []
                                    ),
                                },
                                ensure_ascii=False,
                                default=str,
                            ),
                        )
        manifest_row = connection.execute(
            text(
                "SELECT writer_manifest, writer_manifest_sha256 FROM "
                "regulatory_file_publication WHERE user_file_id = "
                "CAST(:file_id AS uuid)"
            ),
            {"file_id": "1e110f49-3814-4edd-9b08-f17b9f51eabe"},
        ).one_or_none()
        if manifest_row and manifest_row.writer_manifest:
            from onyx.regulatory.writer_publication_models import WriterPublicationManifest
            from onyx.document_index.publication_models import accepts_publication_projection
            from onyx.regulatory.amendments.annexes.models import AnnexTemporalProjection

            manifest = WriterPublicationManifest.model_validate(
                manifest_row.writer_manifest
            )
            print(
                "MANIFEST_460",
                json.dumps(
                    {
                        "proposal_id": manifest.amendment_proposal_id,
                        "sha256": manifest_row.writer_manifest_sha256,
                        "bindings": len(manifest.bindings),
                        "new_bindings": len(manifest.bindings)
                        - len(manifest.previous_binding_ids),
                        "indexes": [
                            {
                                "uuid": idx.index_uuid,
                                "receipts": len(idx.encoder_receipts),
                                "config_sha256": idx.embedding_config_sha256,
                            }
                            for idx in manifest.indexes
                        ],
                    },
                ),
            )
            dependencies = sorted(
                {
                    identifier
                    for binding in manifest.bindings
                    if binding.derived_role == "hierarchical_aggregate"
                    for identifier in binding.dependency_ids
                }
            )
            if dependencies:
                binding_indexes = {}
                previous_ids = set(manifest.previous_binding_ids)
                for binding in manifest.bindings:
                    idx = binding.index
                    key = (
                        "retained" if binding.id in previous_ids else "new",
                        idx.index_uuid,
                        idx.embedding_config_sha256,
                        len(idx.encoder_receipts),
                        type(binding.projection).__name__,
                        binding.derived_role,
                    )
                    binding_indexes[key] = binding_indexes.get(key, 0) + 1
                print("BINDING_INDEXES_460", json.dumps(list(binding_indexes.items())))
                rows = connection.execute(
                    text(
                        "SELECT canonical_chunk_id, payload FROM "
                        "regulatory_temporal_projection WHERE "
                        "canonical_chunk_id = ANY(:ids) AND retired_at IS NULL"
                    ),
                    {"ids": dependencies},
                )
                failed = 0
                checked = 0
                derived_by_dependency = {}
                for binding in manifest.bindings:
                    if binding.derived_role == "hierarchical_aggregate":
                        for identifier in binding.dependency_ids:
                            derived_by_dependency.setdefault(identifier, []).append(
                                binding
                            )
                for old_row in rows:
                    prior = AnnexTemporalProjection.model_validate(old_row.payload)
                    for parent in derived_by_dependency.get(
                        old_row.canonical_chunk_id, []
                    ):
                        idx = parent.index
                        if prior.index.index_uuid != idx.index_uuid:
                            continue
                        checked += 1
                        if not accepts_publication_projection(idx, prior.projection):
                            failed += 1
                            print(
                                "RECEIPT_MISMATCH_460",
                                json.dumps(
                                    {
                                        "dependency_id": old_row.canonical_chunk_id,
                                        "index_uuid": idx.index_uuid,
                                        "prior_type": type(prior.projection).__name__,
                                        "prior_config_sha256": hashlib.sha256(
                                            getattr(
                                                prior.projection,
                                                "embedding_config_json",
                                                "",
                                            ).encode()
                                        ).hexdigest(),
                                        "accepted_receipts": len(idx.encoder_receipts),
                                        "parent_id": parent.id.hex,
                                        "parent_index_config_sha256": idx.embedding_config_sha256,
                                    }
                                ),
                            )
                            if failed >= 8:
                                break
                    if failed >= 8:
                        break
                print("RECEIPT_CHECK_460", json.dumps({"checked": checked, "failed": failed}))
    engine.dispose()


if __name__ == "__main__":
    main()
