"""Bounded, read-only DEV diagnosis of the four reported new-document files."""

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Connection

from onyx.regulatory.document_intake.metadata import (
    MetadataChunk,
    build_metadata_evidence_catalog,
    metadata_input_hash,
    validate_metadata_transport_response,
)
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchRequest,
    parse_vertex_jsonl_output,
)

TARGET_FILES = (
    "2814a713-41f9-4292-aa0d-3cf05104e6ce",
    "38993fc3-fb3c-4eaa-aa0f-576e925b820a",
    "6b1d6522-a26b-4eba-8d82-288feb57f70a",
    "ef8dcdf7-8144-44bb-83cc-6312817b13c1",
)
MAX_CACHE_RECORDS = 512
MAX_CACHE_BYTES = 256 * 1024 * 1024
MAX_RECORD_BYTES = 64 * 1024 * 1024
MAX_REJECTED_FACTS = 32


def emit(kind: str, value: object) -> None:
    print(kind, json.dumps(value, ensure_ascii=False, default=str), flush=True)


def digest(value: str | bytes) -> str:
    return hashlib.sha256(
        value.encode() if isinstance(value, str) else value
    ).hexdigest()


def validate_response(
    response: str, chunks: list[MetadataChunk], targets: list[str]
) -> str | None:
    try:
        validate_metadata_transport_response(
            response, chunks=chunks, target_ids=targets
        )
    except ValueError as error:
        return str(error)[:1000]
    return None


def rejected_facts(response: str, chunks: list[MetadataChunk]) -> list[dict[str, Any]]:
    """Explain individual failures without relaxing whole-response validation."""
    try:
        payload = json.loads(response)
    except ValueError:
        return []
    catalog = {
        entry.evidence_id: entry for entry in build_metadata_evidence_catalog(chunks)
    }
    rejected: list[dict[str, Any]] = []
    for target in payload.get("chunks", []):
        for fact in target.get("fields", []):
            response_one = json.dumps(
                {
                    "chunks": [{"chunk_id": target["chunk_id"], "fields": [fact]}],
                    "warnings": [],
                }
            )
            error = validate_response(response_one, chunks, [target["chunk_id"]])
            if not error:
                continue
            evidence = catalog.get(fact.get("evidence_id"))
            reference = catalog.get(fact.get("reference_evidence_id"))
            rejected.append(
                {
                    "target": target["chunk_id"],
                    "field": fact.get("field"),
                    "value": str(fact.get("value"))[:2000],
                    "basis": fact.get("basis"),
                    "error": error,
                    "evidence_id": fact.get("evidence_id"),
                    "evidence": evidence.model_dump(mode="json") if evidence else None,
                    "reference": reference.model_dump(mode="json")
                    if reference
                    else None,
                }
            )
            if len(rejected) >= MAX_REJECTED_FACTS:
                return rejected
    return rejected


def inspect_cache(
    connection: Connection, members: list[Mapping[str, Any]], since: object
) -> None:
    current: dict[frozenset[str], Mapping[str, Any]] = {}
    for member in members:
        rows = (
            connection.execute(
                text(
                    "SELECT id, text, heading_path, chunk_type, chunk_metadata, validity_start_date, "
                    "validity_end_date FROM regulatory_chunk WHERE user_file_id=CAST(:file_id AS uuid) "
                    "AND chunk_type IS DISTINCT FROM 'hierarchical_aggregate' "
                    "AND chunk_metadata->>'chunk_variant' IS DISTINCT FROM 'hierarchical_aggregate' "
                    "ORDER BY position,id LIMIT 10001"
                ),
                {"file_id": str(member["user_file_id"])},
            )
            .mappings()
            .all()
        )
        if len(rows) > 10000:
            raise RuntimeError("Target file exceeds diagnostic chunk bound")
        # Stored requests, rather than current metadata-enriched rows, are replay authority.
        current[frozenset(row["id"] for row in rows)] = member
        audits = [row["chunk_metadata"].get("intake_legal_metadata") for row in rows]
        emit(
            "INTAKE_FILE",
            {
                **dict(member),
                "chunk_count": len(rows),
                "metadata_audit_count": sum(
                    isinstance(audit, dict) for audit in audits
                ),
                "audit_sample": [audit for audit in audits if isinstance(audit, dict)][
                    :2
                ],
                "chunk_sample": [
                    {
                        "id": row["id"],
                        "heading_path": row["heading_path"],
                        "text_sha256": digest(row["text"]),
                        "text_excerpt": row["text"][:400],
                        "metadata_keys": sorted(row["chunk_metadata"]),
                        "validity_start_date": row["validity_start_date"],
                        "validity_end_date": row["validity_end_date"],
                    }
                    for row in rows[:2]
                ],
            },
        )
    records = (
        connection.execute(
            text(
                "SELECT r.file_id, r.file_metadata, r.bucket_name, c.lobj_oid, c.file_size "
                "FROM file_record r JOIN file_content c ON c.file_id=r.file_id "
                "WHERE r.file_id LIKE 'regulatory-labeling-direct/%' "
                "AND r.file_id NOT LIKE '%/manifest.json' AND r.created_at >= :since "
                "ORDER BY r.created_at DESC LIMIT :limit"
            ),
            {"since": since, "limit": MAX_CACHE_RECORDS + 1},
        )
        .mappings()
        .all()
    )
    if len(records) > MAX_CACHE_RECORDS:
        raise RuntimeError("Cache inventory exceeds diagnostic record bound")
    total_bytes = 0
    matched = 0
    for record in records:
        size = record["file_size"]
        if record["bucket_name"] != "postgres" or not 0 < size <= MAX_RECORD_BYTES:
            raise RuntimeError("Unexpected cache storage or size")
        total_bytes += size
        if total_bytes > MAX_CACHE_BYTES:
            raise RuntimeError("Cache scan exceeds diagnostic byte bound")
        raw = bytes(
            connection.execute(
                text("SELECT pg_catalog.lo_get(CAST(:oid AS oid), 0, :length)"),
                {"oid": record["lobj_oid"], "length": size + 1},
            ).scalar_one()
        )
        if len(raw) != size or digest(raw) != (record["file_metadata"] or {}).get(
            "sha256"
        ):
            raise RuntimeError("Stored cache digest differs")
        stored = json.loads(raw)
        request = VertexBatchRequest.model_validate(stored["request_payload"])
        try:
            payload = json.loads(request.prompt)
        except ValueError:
            continue
        if not isinstance(payload, dict) or not isinstance(
            payload.get("complete_document"), list
        ):
            continue
        chunks = [
            MetadataChunk.model_validate(chunk)
            for chunk in payload["complete_document"]
        ]
        member = current.get(frozenset(chunk.chunk_id for chunk in chunks))
        if member is None:
            continue
        targets = payload["target_chunk_ids"]
        binding = stored["binding_sha256"]
        identity = f"metadata-v{payload['version']}:public:{member['intake_id']}:{member['id']}:{binding}"
        expected_submission = "regulatory-labeling-" + digest(
            f"{identity}:{request.request_hash}"
        )
        expected_file_id = f"regulatory-labeling-direct/{binding}/{expected_submission.removeprefix('regulatory-labeling-')}/{request.request_hash}.json"
        line = json.loads(stored["line"])
        if (
            stored["submission_key"] != expected_submission
            or record["file_id"] != expected_file_id
            or stored["request_hash"] != request.request_hash
            or digest(stored["line"]) != stored["line_sha256"]
            or line.get("key") != request.request_hash
            or line.get("request") != request.to_generate_content_request()
            or metadata_input_hash(chunks) != payload["input_sha256"]
        ):
            raise RuntimeError("Matched metadata cache correlation differs")
        parsed = parse_vertex_jsonl_output(
            iter([stored["line"]]), [request.request_hash]
        )[request.request_hash]
        response = parsed.context
        error = (
            validate_response(response, chunks, targets)
            if response
            else "No complete response body"
        )
        matched += 1
        emit(
            "INTAKE_METADATA_CACHE",
            {
                "file_id": str(member["user_file_id"]),
                "document_id": str(member["id"]),
                "cache_file_id": record["file_id"],
                "request_hash": request.request_hash,
                "targets": targets,
                "validation_feedback": payload.get("validation_feedback"),
                "provider_parse_error": parsed.error,
                "validator_error": error,
                "rejected_facts": rejected_facts(response, chunks)
                if response and error
                else [],
            },
        )
        # The workflow extracts these records into an artifact and suppresses them in logs.
        emit(
            "INTAKE_CACHE_ARTIFACT",
            {
                "file_id": str(member["user_file_id"]),
                "document_id": str(member["id"]),
                "cache_file_id": record["file_id"],
                "stored": stored,
            },
        )
    emit(
        "INTAKE_CACHE_SCAN",
        {"candidate_records": len(records), "bytes": total_bytes, "matched": matched},
    )


def main() -> None:
    local = dotenv_values(Path("deployment/docker_compose/.env.remote-db.local"))

    def setting(name: str) -> str | None:
        return os.environ.get(f"POSTGRES_{name}") or local.get(
            f"REMOTE_POSTGRES_{name}"
        )

    if setting("DB") != "customs-regulations-dev":
        raise RuntimeError("Expected explicit DEV database")
    engine = create_engine(
        URL.create(
            "postgresql+psycopg2",
            username=setting("USER") or "postgres",
            password=setting("PASSWORD"),
            host=setting("HOST"),
            port=int(setting("PORT") or "5432"),
            database="customs-regulations-dev",
        ),
        connect_args={
            "connect_timeout": 5,
            "options": "-c default_transaction_read_only=on -c search_path=public -c statement_timeout=30000 -c application_name=intake_approval_diagnostic",
            **({"sslmode": setting("SSLMODE")} if setting("SSLMODE") else {}),
        },
        pool_size=1,
        max_overflow=0,
    )
    with engine.connect() as connection:
        connection.execute(text("SET TRANSACTION READ ONLY"))
        if (
            connection.execute(text("SELECT current_database()")).scalar_one()
            != "customs-regulations-dev"
        ):
            raise RuntimeError("Database guard failed")
        members = (
            connection.execute(
                text(
                    "SELECT m.id,m.intake_id,m.user_file_id,m.status,m.error,m.role,"
                    "u.name,u.status AS file_status,a.display_name AS source_name,a.sha256 AS source_sha256 "
                    "FROM regulatory_document_intake_file m JOIN user_file u ON u.id=m.user_file_id "
                    "JOIN regulatory_source_asset a ON a.id=m.source_asset_id "
                    "WHERE m.id=ANY(CAST(:ids AS uuid[])) OR m.user_file_id=ANY(CAST(:ids AS uuid[])) ORDER BY m.intake_id,m.id"
                ),
                {"ids": list(TARGET_FILES)},
            )
            .mappings()
            .all()
        )
        if len(members) != 4 or not all(
            any(
                identifier == str(member[key])
                for member in members
                for key in ("id", "user_file_id")
            )
            for identifier in TARGET_FILES
        ):
            raise RuntimeError("Expected all four target intake files")
        batches = (
            connection.execute(
                text(
                    "SELECT id,source_package_id,document_set_id,status,stage,error,review_sha256,"
                    "labeling_run_id,labeling_model_configuration_id,lease_expires_at,created_at,updated_at "
                    "FROM regulatory_document_intake WHERE id=ANY(CAST(:ids AS uuid[])) ORDER BY id"
                ),
                {"ids": list({str(member["intake_id"]) for member in members})},
            )
            .mappings()
            .all()
        )
        for batch in batches:
            emit("INTAKE_BATCH", dict(batch))
        inspect_cache(
            connection, members, min(batch["created_at"] for batch in batches)
        )
        connection.rollback()


if __name__ == "__main__":
    main()
