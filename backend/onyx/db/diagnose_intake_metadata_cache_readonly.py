"""Read only one allowlisted DEV intake's acknowledged metadata cache."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, cast
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session

from onyx.db.document_intake_metadata import metadata_chunks
from onyx.db.enums import UserFileStatus
from onyx.db.models import (
    FileRecord,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    UserFile,
)
from onyx.file_store.file_store import S3BackedFileStore, get_default_file_store
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from onyx.regulatory.document_intake.metadata import (
    MAX_METADATA_RESPONSE_BYTES,
    METADATA_VERSION,
    TARGETS_PER_REQUEST,
    MetadataChunk,
    build_metadata_evidence_catalog,
    build_metadata_repair_request,
    build_metadata_request,
    metadata_input_hash,
    validate_metadata_transport_response,
)
from onyx.regulatory.document_intake.metadata_job import _metadata_response
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchGateway,
    VertexBatchRequest,
)
from onyx.regulatory.labeling.vertex_direct import _Manifest, _StoredResult

BATCH_ID = UUID("f8d84bb5-411a-4f7e-9105-bb7b11f45d34")
MEMBER_ID = UUID("3ce3cc58-d96f-434f-95d7-31cb719479ff")
FILE_ID = UUID("73fd455a-4e58-4966-bfc0-edec5f2d3ca7")
BINDING = "9f04d6faad370b533f335376a2c300044cb51d75ee360b05914950b6b9bea9ea"
DATABASE = "customs-regulations-dev"
MAX_FAILURE_FACTS = 20
MAX_REQUESTS = 128
RepairCategory = Literal[
    "invalid_json_or_schema",
    "invalid_coverage_or_evidence",
    "value_not_covered_by_evidence",
]


def _sha(value: str | bytes) -> str:
    return hashlib.sha256(
        value.encode() if isinstance(value, str) else value
    ).hexdigest()


def _brief_error(error: Exception) -> str:
    if isinstance(error, ValidationError):
        return json.dumps(
            [
                {
                    "location": list(item["loc"]),
                    "message": item["msg"],
                    "type": item["type"],
                }
                for item in error.errors(include_input=False, include_url=False)[:3]
            ],
            ensure_ascii=False,
        )[:500]
    return str(error)[:500]


def _category(error: ValueError) -> RepairCategory:
    return (
        "invalid_json_or_schema"
        if isinstance(error, ValidationError)
        else "value_not_covered_by_evidence"
        if str(error) == "Metadata value contradicts its quoted source wording"
        else "invalid_coverage_or_evidence"
    )


class _ReadResultsOnly:
    """No submit, client, provider, poll, cancellation or file-write interface."""

    def __init__(self, line: str) -> None:
        self._line = line

    def read_results(self, _output_uri: str) -> Iterable[str]:
        yield self._line


class CacheDiagnostic:
    def __init__(self, session: Session, chunks: Sequence[MetadataChunk]) -> None:
        self.session = session
        self.chunks = chunks
        self.by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.catalog = {
            entry.evidence_id: entry
            for entry in build_metadata_evidence_catalog(chunks)
        }
        self.store = get_default_file_store()
        if not isinstance(self.store, (PostgresBackedFileStore, S3BackedFileStore)):
            raise RuntimeError("Unsupported configured read-only diagnostic store")
        self.identity = (
            f"metadata-v{METADATA_VERSION}:public:{BATCH_ID}:{MEMBER_ID}:{BINDING}"
        )
        self.records: set[str] = set()
        self.windows: list[dict[str, Any]] = []
        self.bad_facts: list[dict[str, Any]] = []
        self.total_bad_facts = 0

    def _read(self, identifier: str, limit: int) -> bytes | None:
        if not identifier.startswith(f"regulatory-labeling-direct/{BINDING}/"):
            raise ValueError("Cache read left the allowlisted provider binding")
        record = self.session.scalar(
            select(FileRecord).where(FileRecord.file_id == identifier)
        )
        if record is None:
            return None
        assert isinstance(self.store, (PostgresBackedFileStore, S3BackedFileStore))
        with self.store.read_file(identifier, db_session=self.session) as stream:
            content = stream.read(limit + 1)
        metadata = record.file_metadata
        if not isinstance(metadata, Mapping):
            raise ValueError("Acknowledged cache file metadata is malformed")
        typed_metadata = cast(Mapping[str, object], metadata)
        if (
            len(content) > limit
            or typed_metadata.get("binding_sha256") != BINDING
            or typed_metadata.get("sha256") != _sha(content)
        ):
            raise ValueError("Acknowledged cache file integrity mismatch")
        self.records.add(identifier)
        return content

    def _load(self, request: VertexBatchRequest) -> tuple[str | None, dict[str, Any]]:
        submission_sha = _sha(f"{self.identity}:{request.request_hash}")
        submission_key = f"regulatory-labeling-{submission_sha}"
        prefix = f"regulatory-labeling-direct/{BINDING}/{submission_sha}"
        manifest_bytes = self._read(prefix + "/manifest.json", 64 * 1024)
        detail: dict[str, Any] = {
            "request_hash": request.request_hash,
            "submission_sha256": submission_sha,
        }
        if manifest_bytes is None:
            return None, {**detail, "cache": "no_manifest"}
        manifest = _Manifest.model_validate_json(manifest_bytes)
        if (
            manifest.binding_sha256 != BINDING
            or manifest.submission_key != submission_key
            or manifest.request_hashes != [request.request_hash]
        ):
            raise ValueError("Cached manifest does not match its reconstructed request")
        detail.update(
            {
                "manifest_complete": manifest.complete,
                "manifest_cancelled": manifest.cancelled,
            }
        )
        result_bytes = self._read(
            prefix + f"/{request.request_hash}.json", MAX_METADATA_RESPONSE_BYTES * 4
        )
        if result_bytes is None:
            return None, {**detail, "cache": "manifest_without_output"}
        stored = _StoredResult.model_validate_json(result_bytes)
        if (
            stored.binding_sha256 != BINDING
            or stored.submission_key != submission_key
            or stored.request_hash != request.request_hash
            or _sha(stored.line) != stored.line_sha256
            or len(stored.line.encode()) > MAX_METADATA_RESPONSE_BYTES
            or stored.request_payload != request.model_dump(mode="json")
            or VertexBatchRequest.model_validate(stored.request_payload).request_hash
            != request.request_hash
        ):
            raise ValueError("Cached result does not match its reconstructed request")
        detail.update(
            {"cache": "acknowledged_output", "line_sha256": stored.line_sha256}
        )
        return stored.line, detail

    def _fact_failures(
        self, response: str, targets: Sequence[str], phase: str, request_hash: str
    ) -> None:
        try:
            parsed = json.loads(response)
        except ValueError:
            return
        if not isinstance(parsed, dict) or not isinstance(parsed.get("chunks"), list):
            return
        for target in parsed["chunks"]:
            if not isinstance(target, dict) or not isinstance(
                target.get("fields"), list
            ):
                continue
            target_id = target.get("chunk_id")
            if not isinstance(target_id, str):
                continue
            fields = target["fields"]
            counts = Counter(
                fact.get("field")
                for fact in fields
                if isinstance(fact, dict) and isinstance(fact.get("field"), str)
            )
            for index, fact in enumerate(fields):
                if not isinstance(fact, dict):
                    continue
                # Preserve exact target coverage; isolate one fact without accepting/dropping it.
                isolated = {
                    "chunks": [
                        {
                            "chunk_id": identifier,
                            "fields": [fact] if identifier == target_id else [],
                        }
                        for identifier in targets
                    ],
                    "warnings": [],
                }
                try:
                    validate_metadata_transport_response(
                        json.dumps(isolated, ensure_ascii=False),
                        chunks=self.chunks,
                        target_ids=targets,
                    )
                    error = (
                        "Duplicate field in complete response"
                        if counts.get(fact.get("field"), 0) > 1
                        else None
                    )
                except ValueError as exception:
                    error = _brief_error(exception)
                if error is None:
                    continue
                self.total_bad_facts += 1
                if len(self.bad_facts) >= MAX_FAILURE_FACTS:
                    continue
                evidence = self.catalog.get(str(fact.get("evidence_id", "")))
                reference = self.catalog.get(str(fact.get("reference_evidence_id", "")))
                source = self.by_id.get(target_id)
                self.bad_facts.append(
                    {
                        "phase": phase,
                        "request_hash": request_hash,
                        "target_chunk_id": target_id,
                        "field_index": index,
                        "field": fact.get("field"),
                        "value": str(fact.get("value", ""))[:400],
                        "basis": fact.get("basis"),
                        "evidence_id": fact.get("evidence_id"),
                        "reference_evidence_id": fact.get("reference_evidence_id"),
                        "error": error,
                        "evidence_source_chunk_id": evidence.source_chunk_id
                        if evidence
                        else None,
                        "evidence_scope": evidence.scope if evidence else None,
                        "evidence_text": evidence.text[:400] if evidence else None,
                        "evidence_text_sha256": _sha(evidence.text)
                        if evidence
                        else None,
                        "reference_text": reference.text[:400] if reference else None,
                        "reference_source_chunk_id": reference.source_chunk_id
                        if reference
                        else None,
                        "target_text": source.text[:400] if source else None,
                    }
                )

    def _inspect(
        self, request: VertexBatchRequest, targets: Sequence[str], phase: str
    ) -> tuple[str | None, bool | None, ValueError | None]:
        if len(self.windows) >= MAX_REQUESTS:
            raise ValueError("Diagnostic reconstructed request budget exceeded")
        line, detail = self._load(request)
        detail.update({"phase": phase, "target_ids": list(targets)})
        self.windows.append(detail)
        if line is None:
            return None, None, None
        try:
            context, exhausted = _metadata_response(
                cast(VertexBatchGateway, _ReadResultsOnly(line)),
                request,
                "read-only-cached-result",
            )
        except ValueError as error:
            detail.update(
                {"outcome": "envelope_rejected", "error": _brief_error(error)}
            )
            return None, None, None
        if exhausted:
            detail["outcome"] = "authoritative_MAX_TOKENS"
            return None, True, None
        assert context is not None
        try:
            result = validate_metadata_transport_response(
                context, chunks=self.chunks, target_ids=targets
            )
            detail.update(
                {
                    "outcome": "strict_valid",
                    "fact_count": sum(len(target.fields) for target in result.chunks),
                }
            )
            return context, False, None
        except ValueError as error:
            detail.update(
                {
                    "outcome": "semantic_rejected",
                    "error": _brief_error(error),
                    "repair_category": _category(error),
                }
            )
            self._fact_failures(context, targets, phase, request.request_hash)
            return context, False, error

    def inspect_targets(self, targets: Sequence[str], path: str) -> None:
        request = build_metadata_request(self.chunks, targets)
        context, exhausted, error = self._inspect(request, targets, path + ":base")
        if error is not None and context is not None:
            repair = build_metadata_repair_request(
                request, context, validation_category=_category(error)
            )
            _context, exhausted, _repair_error = self._inspect(
                repair, targets, path + ":repair1"
            )
        if exhausted and len(targets) > 1:
            midpoint = len(targets) // 2
            self.inspect_targets(targets[:midpoint], path + ":left")
            self.inspect_targets(targets[midpoint:], path + ":right")


def main() -> None:
    if os.environ.get("POSTGRES_DB") != DATABASE:
        raise RuntimeError("Expected the explicitly named DEV database")
    engine = create_engine(
        URL.create(
            "postgresql+psycopg2",
            username=os.environ.get("POSTGRES_USER", "postgres"),
            password=os.environ.get("POSTGRES_PASSWORD"),
            host=os.environ.get("POSTGRES_HOST"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            database=DATABASE,
        ),
        connect_args={
            "connect_timeout": 5,
            "options": "-c default_transaction_read_only=on -c search_path=public -c statement_timeout=30000 -c application_name=intake_metadata_cache_readonly",
            **(
                {"sslmode": os.environ["POSTGRES_SSLMODE"]}
                if os.environ.get("POSTGRES_SSLMODE")
                else {}
            ),
        },
        pool_size=1,
        max_overflow=0,
    )
    with engine.connect() as connection, connection.begin():
        connection.execute(text("SET TRANSACTION READ ONLY"))
        guards = connection.execute(
            text(
                "SELECT current_database(), current_schema(), current_setting('transaction_read_only')"
            )
        ).one()
        if tuple(guards) != (DATABASE, "public", "on"):
            raise RuntimeError("Read-only database fence unavailable")
        with Session(bind=connection, autoflush=False) as session:
            batch = session.get_one(RegulatoryDocumentIntake, BATCH_ID)
            member = session.get_one(RegulatoryDocumentIntakeFile, MEMBER_ID)
            file = session.get_one(UserFile, FILE_ID)
            if (
                batch.document_set_id != 15
                or member.intake_id != batch.id
                or member.user_file_id != file.id
                or file.user_id != batch.created_by
                or file.status != UserFileStatus.CHUNKED
                or batch.labeling_run_id is not None
                or batch.stage != "metadata"
            ):
                raise RuntimeError("Allowlisted unpublished metadata scope changed")
            rows = list(
                session.scalars(
                    select(RegulatoryChunk)
                    .where(RegulatoryChunk.user_file_id == FILE_ID)
                    .order_by(RegulatoryChunk.position, RegulatoryChunk.id)
                )
            )
            if len(rows) != 239:
                raise RuntimeError("Allowlisted canonical source count changed")
            chunks = metadata_chunks(rows)
            diagnostic = CacheDiagnostic(session, chunks)
            for offset in range(0, len(chunks), TARGETS_PER_REQUEST):
                diagnostic.inspect_targets(
                    [
                        chunk.chunk_id
                        for chunk in chunks[offset : offset + TARGETS_PER_REQUEST]
                    ],
                    f"window{offset // TARGETS_PER_REQUEST}",
                )
            report = {
                "scope": "read_only_cached_metadata_no_submit_no_provider_no_configuration_reads",
                "database": DATABASE,
                "transaction_read_only": True,
                "batch_id": str(BATCH_ID),
                "document_id": str(MEMBER_ID),
                "user_file_id": str(FILE_ID),
                "batch_status": batch.status,
                "stage": batch.stage,
                "canonical_count": len(rows),
                "metadata_target_count": len(chunks),
                "metadata_version": METADATA_VERSION,
                "input_sha256": metadata_input_hash(chunks),
                "canonical_metadata_audit_count": sum(
                    "intake_legal_metadata" in row.chunk_metadata for row in rows
                ),
                "verified_cache_manifest_count": sum(
                    identifier.endswith("/manifest.json")
                    for identifier in diagnostic.records
                ),
                "verified_cache_output_count": sum(
                    not identifier.endswith("/manifest.json")
                    for identifier in diagnostic.records
                ),
                "requests": diagnostic.windows,
                "total_invalid_facts": diagnostic.total_bad_facts,
                "reported_invalid_facts_limit": MAX_FAILURE_FACTS,
                "invalid_facts": diagnostic.bad_facts,
            }
            print(
                "METADATA_CACHE_DIAGNOSTIC",
                json.dumps(report, ensure_ascii=False, separators=(",", ":")),
                flush=True,
            )
            session.rollback()
    engine.dispose()


if __name__ == "__main__":
    main()
