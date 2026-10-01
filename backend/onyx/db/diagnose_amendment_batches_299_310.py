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
            if batch["id"] == 299:
                from onyx.regulatory.amendments.draft_integrity import (
                    explicit_added_article_identity,
                    explicit_added_body,
                )
                from onyx.regulatory.amendments.new_provision_policy import (
                    explicitly_adds_top_level_provision,
                )
                instruction = (batch["segmented_instructions"] or [])[5]
                instruction_text = instruction.get("instruction_text", "")
                print("BATCH_299_6A_PARSE", json.dumps({
                    "length": len(instruction_text),
                    "tail": instruction_text[-360:],
                    "is_top_level": explicitly_adds_top_level_provision(instruction_text),
                    "body_length": len(explicit_added_body(instruction_text) or ""),
                    "identity": explicit_added_article_identity(instruction_text),
                }, ensure_ascii=False))
                for event in batch["analysis_log"] or []:
                    indices = event.get("indices") or []
                    if 5 in indices or event.get("index") == 5:
                        print("BATCH_299_INDEX_5", json.dumps(event, ensure_ascii=False, default=str))
            if batch["id"] == 306:
                print(
                    "BATCH_306_SOURCE_MEMBER",
                    "c7bed4e2-77a2-4609-bb52-3fdffcd15cb1"
                    in batch["user_file_ids"],
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
                manifest_index_by_uuid = {
                    idx.index_uuid: idx for idx in manifest.indexes
                }
                rejected_by_current = []
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
                    if (
                        binding.id not in previous_ids
                        and not accepts_publication_projection(
                            manifest_index_by_uuid[idx.index_uuid],
                            binding.projection,
                        )
                    ):
                        rejected_by_current.append(binding.id.hex)
                print("BINDING_INDEXES_460", json.dumps(list(binding_indexes.items())))
                print(
                    "REJECTED_BY_CURRENT_460",
                    json.dumps(
                        {"count": len(rejected_by_current), "sample": rejected_by_current[:8]}
                    ),
                )
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
                dependency_windows = {}
                active_rows = connection.execute(text(
                    "SELECT canonical_chunk_id, index_identity_sha256, effective_start, "
                    "effective_end FROM regulatory_temporal_projection WHERE "
                    "canonical_chunk_id = ANY(:ids) AND retired_at IS NULL"
                ), {"ids": dependencies})
                for active_row in active_rows:
                    dependency_windows.setdefault(active_row.canonical_chunk_id, []).append({
                        "identity": active_row.index_identity_sha256,
                        "start": active_row.effective_start,
                        "end": active_row.effective_end,
                        "origin": "active",
                    })
                for other in manifest.bindings:
                    if other.id in previous_ids:
                        continue
                    identifier = json.loads(other.projection.source_json)["regulatory_chunk_id"]
                    dependency_windows.setdefault(identifier, []).append({
                        "identity": other.index.temporal_lookup_identity(),
                        "start": other.effective_start,
                        "end": other.effective_end,
                        "origin": "pending",
                    })
                from onyx.regulatory.contextual import context_reference_date
                window_issues = 0
                for parent in manifest.bindings:
                    if parent.id in previous_ids or parent.derived_role != "hierarchical_aggregate":
                        continue
                    identities = set(manifest_index_by_uuid[parent.index.index_uuid].temporal_lookup_identities())
                    for identifier in parent.dependency_ids:
                        windows = [w for w in dependency_windows.get(identifier, []) if w["identity"] in identities]
                        reference_date = context_reference_date(parent.effective_start, parent.effective_end)
                        selected = [w for w in windows if
                            (w["start"] is None or w["start"] <= reference_date)
                            and (w["end"] is None or w["end"] > reference_date)
                        ]
                        noncovering = [w for w in selected if not (
                            (w["start"] is None or parent.effective_start is not None and w["start"] <= parent.effective_start)
                            and (w["end"] is None or parent.effective_end is not None and w["end"] >= parent.effective_end)
                        )]
                        if noncovering:
                            print("WINDOW_MISMATCH_460", json.dumps({
                                "parent": parent.id.hex,
                                "parent_start": str(parent.effective_start),
                                "parent_end": str(parent.effective_end),
                                "dependency": identifier,
                                "reference_date": str(reference_date),
                                "windows": [{**w, "start": str(w["start"]), "end": str(w["end"])} for w in selected[:5]],
                            }))
                            window_issues += 1
                            if window_issues >= 12:
                                break
                    if window_issues >= 12:
                        break
                print("WINDOW_ISSUE_COUNT_460", window_issues)
        targets = [
            ("299_6A", "rc.user_file_id = CAST(:file_id AS uuid) AND "
             "(rc.chunk_metadata->>'article_no' = '6/A' OR "
             "rc.text ILIKE '%MADDE 6/A%')", {"file_id": "014fbc6a-2da2-4845-9df9-f4afc625587a"}),
            ("302_CODE_0703", "(uf.name ILIKE '%İthalat Rejimi%' OR "
             "uf.name ILIKE '%3350%') AND rc.text ILIKE '%0703.10.19.00.11%'", {}),
            ("302_CODE_1206", "(uf.name ILIKE '%İthalat Rejimi%' OR "
             "uf.name ILIKE '%3350%') AND rc.text ILIKE '%1206.00.91.00.19%'", {}),
            ("306_TITLE", "uf.name ILIKE '%Gümrüksüz Satış Mağazaları%' OR "
             "uf.name ILIKE '%2018/13%'", {}),
        ]
        for label, predicate, params in targets:
            rows = connection.execute(
                text(
                    "SELECT rc.id, rc.user_file_id, uf.name, rc.status, "
                    "rc.position, rc.chunk_type, rc.chunk_metadata->>'article_no' "
                    "AS article_no, rc.text, rc.heading_path FROM regulatory_chunk rc "
                    "JOIN user_file uf ON uf.id=rc.user_file_id WHERE "
                    f"({predicate}) LIMIT 30"
                ),
                params,
            )
            for row in rows:
                text_value = row.text
                print(
                    "SOURCE_MATCH",
                    json.dumps(
                        {
                            "target": label,
                            "id": row.id,
                            "file_id": str(row.user_file_id),
                            "file_name": row.name,
                            "status": row.status,
                            "position": row.position,
                            "chunk_type": row.chunk_type,
                            "article_no": row.article_no,
                            "text_sha256": hashlib.sha256(text_value.encode()).hexdigest(),
                            "text_excerpt": text_value[:700],
                            "heading_path": row.heading_path[:2],
                        },
                        ensure_ascii=False,
                    ),
                )
        files = connection.execute(
            text(
                "SELECT id, name, status, chunk_count FROM user_file WHERE "
                "name ILIKE '%İthalat Rejimi%' OR name ILIKE '%3350%' OR "
                "name ILIKE '%Gümrüksüz Satış Mağazaları%' OR "
                "name ILIKE '%2018/13%' LIMIT 100"
            )
        )
        for row in files:
            print(
                "SOURCE_FILE",
                json.dumps(
                    {"id": str(row.id), "name": row.name,
                     "status": row.status, "chunk_count": row.chunk_count},
                    ensure_ascii=False,
                    default=str,
                ),
            )
        for row in connection.execute(
            text(
                "SELECT id, position, status, chunk_metadata->>'article_no' "
                "AS article_no, text, heading_path FROM regulatory_chunk WHERE "
                "user_file_id=CAST(:file_id AS uuid) AND position BETWEEN 12 AND 25 "
                "ORDER BY position, created_at"
            ),
            {"file_id": "014fbc6a-2da2-4845-9df9-f4afc625587a"},
        ):
            print(
                "FILE_299_AROUND_6",
                json.dumps(
                    {"id": row.id, "position": row.position,
                     "status": row.status, "article_no": row.article_no,
                     "text_excerpt": row.text[:350],
                     "heading_path": row.heading_path[:3]},
                    ensure_ascii=False,
                ),
            )
        for row in connection.execute(
            text(
                "SELECT rc.id, rc.user_file_id, uf.name, rc.status, "
                "rc.position, rc.text FROM regulatory_chunk rc JOIN user_file uf "
                "ON uf.id=rc.user_file_id WHERE "
                "rc.chunk_metadata->>'article_no' = '6/A' AND "
                "(uf.name ILIKE '%dovizlerinin_turk_lirasina_donusum%' OR "
                "uf.name ILIKE '%2023-5%') LIMIT 30"
            )
        ):
            print(
                "ARTICLE_6A_OTHER_FILE",
                json.dumps(
                    {"id": row.id, "file_id": str(row.user_file_id),
                     "name": row.name, "status": row.status,
                     "position": row.position,
                     "text_sha256": hashlib.sha256(row.text.encode()).hexdigest(),
                     "text_excerpt": row.text[:700]},
                    ensure_ascii=False,
                ),
            )
        for row in connection.execute(
            text(
                "SELECT id, name, status, chunk_count FROM user_file WHERE "
                "name ILIKE '%2018%13%' OR name ILIKE '%gumruksuz%' OR "
                "name ILIKE '%gümrüksüz%' LIMIT 100"
            )
        ):
            print(
                "FILE_306_CANDIDATE",
                json.dumps(
                    {"id": str(row.id), "name": row.name,
                     "status": row.status, "chunk_count": row.chunk_count},
                    ensure_ascii=False, default=str,
                ),
            )
        for row in connection.execute(
            text(
                "SELECT id, position, status, chunk_metadata->>'article_no' "
                "AS article_no, text, heading_path FROM regulatory_chunk WHERE "
                "user_file_id=CAST(:file_id AS uuid) AND "
                "(chunk_metadata->>'article_no' = '2' OR position < 6) "
                "ORDER BY position LIMIT 20"
            ),
            {"file_id": "c7bed4e2-77a2-4609-bb52-3fdffcd15cb1"},
        ):
            print(
                "FILE_306_ARTICLE_2",
                json.dumps(
                    {"id": row.id, "position": row.position,
                     "status": row.status, "article_no": row.article_no,
                     "text_excerpt": row.text[:500],
                     "heading_path": row.heading_path[:3]},
                    ensure_ascii=False,
                ),
            )
    engine.dispose()


if __name__ == "__main__":
    main()
