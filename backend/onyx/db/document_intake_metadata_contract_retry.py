"""Idle owner Retry authority for exact, disproved clause receipt fields."""

import hashlib
import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, cast
from uuid import UUID, uuid4

from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy import select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db.document_intake_metadata import metadata_chunks
from onyx.db.document_intake_metadata_review import canonical_metadata_source_digest
from onyx.db.enums import UserFileStatus
from onyx.db.labeling_configuration import (
    LabelingProviderBinding,
    resolve_intake_labeling_provider_binding,
)
from onyx.db.models import (
    AmendmentSourcePackage,
    DocumentSet__UserFile,
    FileRecord,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryFilePublication,
    RegulatoryIndexingJob,
    RegulatoryLabelingRun,
    RegulatorySourceAsset,
    RegulatoryTemporalProjection,
    User,
    UserFile,
    UserFileProjectionRepair,
)
from onyx.file_processing.original_ingestion import OriginalIngestionReceipt
from onyx.file_store.file_store import get_default_file_store
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from onyx.regulatory.document_intake.metadata import (
    METADATA_AUDIT_KEY,
    METADATA_MODEL,
    METADATA_VERSION,
    TARGETS_PER_REQUEST,
    VALIDITY_FIELDS,
    ChunkLegalMetadata,
    LegalMetadataFact,
    LegalMetadataResult,
    MetadataChunk,
    build_metadata_evidence_catalog,
    build_metadata_output_reserve_request,
    build_metadata_request,
    metadata_input_hash,
    metadata_response_needs_split,
    selected_clause_evidence_contradicts,
    target_clause_evidence_contradicts,
    validate_metadata_response,
)
from onyx.regulatory.document_intake.metadata_checkpoint import (
    MetadataCheckpoint,
    metadata_digest,
)
from onyx.regulatory.document_intake.metadata_contract_retry import (
    CONTRACT_RETRY_KEY,
    MAX_CONTRACT_BYTES,
    MAX_CONTRACT_TARGETS,
    ClauseFailureReason,
    ContractAppliedClause,
    ContractInvalidClause,
    ContractRetryDocument,
    MetadataContractRetry,
    build_contract_retry_request,
    metadata_ack_file_id,
    require_contract_accepted_request,
    validate_contract_retry_result,
    validate_repaired_clause_fact,
)
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchRequest,
    parse_vertex_jsonl_output,
)
from shared_configs.contextvars import get_current_tenant_id


def _contract_retry(batch: RegulatoryDocumentIntake) -> MetadataContractRetry | None:
    raw_checkpoint = getattr(batch, "metadata_checkpoint", None)
    if not isinstance(raw_checkpoint, dict):
        return None
    source = raw_checkpoint.get("source_snapshot", {})
    if not isinstance(source, dict):
        raise intake.IntakeConflictError(
            "The metadata checkpoint source identity changed"
        )
    raw = source.get(CONTRACT_RETRY_KEY)
    if raw is None:
        return None
    checkpoint = MetadataCheckpoint.model_validate(batch.metadata_checkpoint)
    checkpoint.validate_checksum()
    intent = MetadataContractRetry.model_validate(raw)
    previous = intent.previous_checkpoint
    if (
        intent.intake_id != batch.id
        or intent.package_id != batch.source_package_id
        or intent.owner_id != batch.created_by
        or checkpoint.recipe_sha256 != intent.recipe_sha256
        or checkpoint.provider_binding != previous.provider_binding
        or checkpoint.request_hashes != previous.request_hashes
        or checkpoint.predecessor is None
        or checkpoint.predecessor.checkpoint_sha256 != previous.checkpoint_sha256
        or checkpoint.predecessor.recipe_sha256 != previous.recipe_sha256
        or {
            k: v
            for k, v in checkpoint.source_snapshot.items()
            if k != CONTRACT_RETRY_KEY
        }
        != previous.source_snapshot
    ):
        raise intake.IntakeConflictError(
            "The metadata contract repair identity changed"
        )
    return intent


def contract_retry(batch: RegulatoryDocumentIntake) -> MetadataContractRetry | None:
    try:
        return _contract_retry(batch)
    except (ValueError, KeyError, TypeError) as error:
        raise intake.IntakeConflictError(
            "The stored metadata contract repair is invalid or changed"
        ) from error


def _rows(session: Session, file_id: UUID) -> list[RegulatoryChunk]:
    return list(
        session.scalars(
            select(RegulatoryChunk)
            .where(RegulatoryChunk.user_file_id == file_id)
            .order_by(RegulatoryChunk.position, RegulatoryChunk.id)
            .execution_options(populate_existing=True)
        )
    )


def _preserved_metadata(rows: list[RegulatoryChunk], targets: set[str]) -> str:
    values = []
    for row in rows:
        metadata = deepcopy(row.chunk_metadata)
        if row.id in targets:
            metadata.pop("clause_label", None)
            audit = metadata.get(METADATA_AUDIT_KEY)
            if isinstance(audit, dict):
                audit.get("fields", {}).pop("clause_label", None)
                audit.pop(CONTRACT_RETRY_KEY, None)
        values.append(
            {
                "id": row.id,
                "metadata": metadata,
                "start": str(row.validity_start_date)
                if row.validity_start_date
                else None,
                "end": str(row.validity_end_date) if row.validity_end_date else None,
            }
        )
    return metadata_digest(values)


def _owned_scope(
    session: Session,
    batch: RegulatoryDocumentIntake,
    previous: MetadataCheckpoint,
    *,
    previous_run_id: UUID,
    allow_fresh_run: bool = False,
) -> tuple[list[RegulatoryDocumentIntakeFile], dict[str, list[RegulatoryChunk]]]:
    source = previous.source_snapshot
    package = session.get_one(
        AmendmentSourcePackage, batch.source_package_id, populate_existing=True
    )
    old_run = session.get_one(
        RegulatoryLabelingRun, previous_run_id, populate_existing=True
    )
    members = list(
        session.scalars(
            select(RegulatoryDocumentIntakeFile)
            .where(RegulatoryDocumentIntakeFile.intake_id == batch.id)
            .order_by(RegulatoryDocumentIntakeFile.id)
            .execution_options(populate_existing=True)
        )
    )
    files = {str(member.user_file_id) for member in members}
    if (
        source.get("tenant_id") != get_current_tenant_id()
        or source.get("intake_id") != str(batch.id)
        or source.get("package_id") != str(package.id)
        or source.get("created_by") != str(batch.created_by)
        or source.get("document_set_id") != batch.document_set_id
        or source.get("environment") != package.environment
        or package.created_by != batch.created_by
        or package.document_set_id != batch.document_set_id
        or package.status != "ready"
        or package.issues
        or not package.manifest_sha256
        or package.manifest_sha256 != source.get("manifest_sha256")
        or package.manifest_file_id != source.get("manifest_file_id")
        or {str(m.id) for m in members} != set(source.get("documents", {}))
        or len(members) != package.asset_count
        or len(files) != len(members)
        or old_run.status
        not in {"completed", "completed_with_errors", "failed", "cancelled"}
        or old_run.lease_token is not None
        or old_run.lease_expires_at is not None
        or len(old_run.file_ids) != len(files)
        or set(old_run.file_ids) != files
        or old_run.document_set_id != batch.document_set_id
        or old_run.requested_by_id != batch.created_by
        or old_run.provider_binding != previous.provider_binding
        or (not allow_fresh_run and batch.labeling_run_id != previous_run_id)
    ):
        raise intake.IntakeConflictError(
            "Metadata contract repair requires its exact idle owner and source scope"
        )
    for active in session.scalars(
        select(RegulatoryLabelingRun).where(
            RegulatoryLabelingRun.document_set_id == batch.document_set_id,
            RegulatoryLabelingRun.status.in_(("queued", "running")),
        )
    ):
        if active.id != batch.labeling_run_id and set(active.file_ids) & files:
            raise intake.IntakeConflictError(
                "Active labeling overlaps contract metadata repair"
            )
    if batch.labeling_run_id != previous_run_id:
        current = session.get_one(
            RegulatoryLabelingRun, batch.labeling_run_id, populate_existing=True
        )
        if (
            len(current.file_ids) != len(files)
            or set(current.file_ids) != files
            or current.document_set_id != batch.document_set_id
            or current.requested_by_id != batch.created_by
            or current.provider_binding != previous.provider_binding
        ):
            raise intake.IntakeConflictError(
                "The fresh contract labeling run identity changed"
            )
    by_member = {}
    for member in members:
        assert member.user_file_id is not None
        fid = member.user_file_id
        wanted = source["documents"][str(member.id)]
        file = session.get_one(UserFile, fid, populate_existing=True)
        asset = session.get_one(
            RegulatorySourceAsset, member.source_asset_id, populate_existing=True
        )
        publication = session.get(
            RegulatoryFilePublication, fid, populate_existing=True
        )
        if (
            wanted["user_file_id"] != str(fid)
            or wanted["source_asset_id"] != str(asset.id)
            or file.user_id != batch.created_by
            or file.status != UserFileStatus.CHUNKED
            or member.approval_attempt_id is not None
            or asset.package_id != package.id
            or asset.file_id != wanted["source_file_id"]
            or asset.sha256 != wanted["raw_sha256"]
            or asset.text_file_id != wanted["text_file_id"]
            or asset.text_sha256 != wanted["text_sha256"]
            or asset.original_url != wanted["original_url"]
            or asset.final_url != wanted["final_url"]
            or member.role != wanted["role"]
            or member.relationships != wanted["relationships"]
            or (str(member.parent_document_id) if member.parent_document_id else None)
            != wanted["parent_document_id"]
            or session.get(
                DocumentSet__UserFile,
                (batch.document_set_id, fid),
                populate_existing=True,
            )
            is None
            or publication is None
            or publication.epoch != 0
            or publication.gate_closed
            or publication.writer_manifest is not None
            or publication.lease_expires_at > datetime.now(timezone.utc)
            or not publication.original_ingestion_receipt
        ):
            raise intake.IntakeConflictError(
                "Metadata contract repair requires unchanged unpublished raw source ownership"
            )
        original = OriginalIngestionReceipt.model_validate(
            publication.original_ingestion_receipt
        )
        if (
            original.file_id != file.file_id
            or file.file_id != wanted["original_file_id"]
            or original.raw_sha256 != asset.sha256
            or metadata_digest(original.model_dump(mode="json"))
            != wanted["original_receipt_sha256"]
            or file.regulatory_chunk_generation_hash != wanted["generation_hash"]
        ):
            raise intake.IntakeConflictError(
                "Metadata contract repair original source receipt changed"
            )
        for model in (
            RegulatoryIndexingJob,
            RegulatoryTemporalProjection,
            UserFileProjectionRepair,
        ):
            if (
                session.scalar(select(model).where(model.user_file_id == fid).limit(1))
                is not None
            ):
                raise intake.IntakeConflictError(
                    "Publication history prevents metadata contract repair"
                )
        rows = _rows(session, fid)
        if (
            not rows
            or any(
                r.status != "active"
                or r.source != "indexed"
                or r.superseded_by_chunk_id
                or r.supersedes_chunk_id
                for r in rows
            )
            or canonical_metadata_source_digest(rows)
            != wanted["canonical_source_sha256"]
            or metadata_input_hash(metadata_chunks(rows)) != wanted["input_sha256"]
        ):
            raise intake.IntakeConflictError(
                "Metadata contract repair canonical source or revisions changed"
            )
        by_member[str(member.id)] = rows
    return members, by_member


def _fact(row: RegulatoryChunk, field: str, entry: dict[str, Any]) -> LegalMetadataFact:
    value = (
        getattr(row, field)
        if field in VALIDITY_FIELDS
        else row.chunk_metadata.get(field)
    )
    if field in VALIDITY_FIELDS and value is not None:
        value = value.isoformat()
    return LegalMetadataFact.model_validate(
        {
            "field": field,
            "value": value,
            "source_chunk_id": entry.get("source_chunk_id"),
            "evidence_quote": entry.get("evidence_quote"),
            "basis": entry.get("basis", "explicit"),
            "reference_chunk_id": entry.get("reference_chunk_id"),
            "reference_quote": entry.get("reference_quote"),
        }
    )


def _invalid_fields(
    rows: list[RegulatoryChunk],
) -> dict[str, tuple[RegulatoryChunk, dict[str, Any], ClauseFailureReason]]:
    chunks = metadata_chunks(rows)
    sources = {c.chunk_id: c for c in chunks}
    input_hash = metadata_input_hash(chunks)
    invalid = {}
    for row in rows:
        if row.id not in sources:
            continue
        if (
            row.validity_start_date
            and row.validity_end_date
            and row.validity_start_date >= row.validity_end_date
        ):
            raise intake.IntakeConflictError(
                "Unrelated invalid metadata dates prevent contract repair"
            )
        audit = row.chunk_metadata.get(METADATA_AUDIT_KEY)
        if (
            not isinstance(audit, dict)
            or audit.get("model") != METADATA_MODEL
            or audit.get("version") != METADATA_VERSION
            or audit.get("input_stale") is not False
            or audit.get("input_sha256") != input_hash
            or not isinstance(audit.get("fields"), dict)
            or not isinstance(audit.get("request_hash"), str)
            or re.fullmatch(r"[0-9a-f]{64}", audit["request_hash"]) is None
        ):
            raise intake.IntakeConflictError(
                "Unrelated missing or stale metadata prevents contract repair"
            )
        for field, entry in audit.get("fields", {}).items():
            if not isinstance(entry, dict) or entry.get("source") != "llm":
                continue
            if entry.get("stale") is not False:
                raise intake.IntakeConflictError(
                    "Unrelated stale metadata prevents contract repair"
                )
            if (
                not isinstance(entry.get("request_hash"), str)
                or re.fullmatch(r"[0-9a-f]{64}", entry["request_hash"]) is None
            ):
                raise intake.IntakeConflictError(
                    "Unrelated missing metadata request authority prevents contract repair"
                )
            fact = _fact(row, field, entry)
            try:
                validate_metadata_response(
                    LegalMetadataResult(
                        chunks=[ChunkLegalMetadata(chunk_id=row.id, fields=[fact])]
                    ).model_dump_json(),
                    chunks=chunks,
                    target_ids=[row.id],
                )
            except ValueError as error:
                if (
                    str(error)
                    == "Metadata clause locator contradicts its selected source evidence"
                    and field == "clause_label"
                    and isinstance(fact.value, str)
                    and selected_clause_evidence_contradicts(
                        value=fact.value,
                        evidence_quote=fact.evidence_quote,
                        source=sources[fact.source_chunk_id],
                    )
                ):
                    reason: ClauseFailureReason = "selected_clause_contradiction"
                elif (
                    str(error)
                    == "Metadata clause locator contradicts its target source prefix"
                    and field == "clause_label"
                    and isinstance(fact.value, str)
                    and target_clause_evidence_contradicts(
                        value=fact.value, target=sources[row.id]
                    )
                ):
                    reason = "target_clause_prefix_contradiction"
                else:
                    raise intake.IntakeConflictError(
                        "Unrelated metadata validation failure prevents contract repair"
                    ) from error
                invalid[row.id] = (row, deepcopy(entry), reason)
    return invalid


def _read_ack(session: Session, cache_id: str) -> bytes:
    record = session.get(FileRecord, cache_id, populate_existing=True)
    if record is None or not isinstance(record.file_metadata, dict):
        raise intake.IntakeConflictError("The actual prior metadata ACK is unavailable")
    store = get_default_file_store()
    stream = (
        store.read_file(cache_id, db_session=session)
        if isinstance(store, PostgresBackedFileStore)
        else store.read_file(cache_id)
    )
    with stream:
        raw = stream.read(8 * 1024 * 1024 + 1)
    stored_metadata = cast(dict[str, Any], record.file_metadata)
    if not 0 < len(raw) <= 8 * 1024 * 1024 or hashlib.sha256(
        raw
    ).hexdigest() != stored_metadata.get("sha256"):
        raise intake.IntakeConflictError(
            "The actual prior metadata ACK integrity changed"
        )
    return raw


def _ack(
    raw: bytes,
    *,
    binding: LabelingProviderBinding,
    cache_id: str,
    expected_hash: str,
) -> tuple[VertexBatchRequest, dict[str, Any], str]:
    ack = json.loads(raw)
    request = VertexBatchRequest.model_validate(ack["request_payload"])
    line = json.loads(ack["line"])
    if (
        request.request_hash != expected_hash
        or ack["request_hash"] != expected_hash
        or line["key"] != expected_hash
        or ack["binding_sha256"] != binding.fingerprint
        or line["request"] != request.to_generate_content_request()
        or hashlib.sha256(ack["line"].encode()).hexdigest() != ack["line_sha256"]
        or cache_id
        != f"regulatory-labeling-direct/{binding.fingerprint}/{ack['submission_key'].removeprefix('regulatory-labeling-')}/{expected_hash}.json"
        or line.get("error") is not None
        or line.get("status", "") != ""
        or len(line["response"]["candidates"]) != 1
        or line["response"]["candidates"][0]["finishReason"] != "STOP"
    ):
        raise ValueError(
            "Metadata ACK request, provider or completion identity changed"
        )
    parsed = parse_vertex_jsonl_output(iter([ack["line"]]), [expected_hash])[
        expected_hash
    ]
    if parsed.error is not None or parsed.context is None:
        raise ValueError("Metadata ACK has no complete response")
    return request, json.loads(request.prompt), parsed.context


def _previous_lineage(
    raw: bytes,
    *,
    binding: LabelingProviderBinding,
    cache_id: str,
    row: RegulatoryChunk,
    entry: dict[str, Any],
    chunks: list[MetadataChunk],
    previous_hashes: list[str],
) -> tuple[str, int, str]:
    request, payload, response = _ack(
        raw, binding=binding, cache_id=cache_id, expected_hash=entry["request_hash"]
    )
    old_chunks = [MetadataChunk.model_validate(c) for c in payload["complete_document"]]
    if metadata_input_hash(old_chunks) != metadata_input_hash(chunks):
        raise ValueError("Prior metadata ACK source context changed")
    catalog = {e.evidence_id: e for e in build_metadata_evidence_catalog(old_chunks)}
    if payload["evidence_catalog"] != [
        e.model_dump(mode="json") for e in catalog.values()
    ]:
        raise ValueError("Prior metadata ACK evidence catalog changed")
    targets = payload["target_chunk_ids"]
    base = build_metadata_request(old_chunks, targets)
    literal_payload = {k: v for k, v in payload.items() if k != "validation_feedback"}
    literal_base = VertexBatchRequest(
        prompt=json.dumps(
            literal_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
        system_instruction=request.system_instruction,
        generation_config=request.generation_config,
    )
    if literal_base != base and literal_base != build_metadata_output_reserve_request(
        base
    ):
        raise ValueError("Prior metadata base or output reserve lineage changed")
    feedback = payload.get("validation_feedback")
    attempt = 0
    if feedback is not None:
        if (
            feedback.get("original_request_hash") != literal_base.request_hash
            or feedback.get("repair_attempt") != 1
        ):
            raise ValueError("Prior metadata repair lineage changed")
        attempt = 1
    elif request != literal_base:
        raise ValueError("Prior metadata base request changed")
    checkpoint_hash = base.request_hash
    if checkpoint_hash not in previous_hashes:
        # A cache ACK may be a genuine child of the frozen eight-target window.
        # Only the existing deterministic bisection tree authorizes that lineage.
        ancestor = None
        for offset in range(0, len(old_chunks), TARGETS_PER_REQUEST):
            root = [
                c.chunk_id for c in old_chunks[offset : offset + TARGETS_PER_REQUEST]
            ]
            queue = [root]
            while queue:
                candidate = queue.pop()
                if candidate == targets:
                    parent = build_metadata_request(old_chunks, root)
                    if parent.request_hash in previous_hashes:
                        ancestor = parent.request_hash
                if len(candidate) > 1:
                    midpoint = len(candidate) // 2
                    queue.extend((candidate[:midpoint], candidate[midpoint:]))
        checkpoint_hash = ancestor or ""
    if not checkpoint_hash:
        raise ValueError("Prior metadata request is outside its frozen checkpoint")
    result = json.loads(response)
    if metadata_response_needs_split(response, targets):
        raise ValueError(
            "Prior metadata ACK changed its exact target or field coverage"
        )
    selected = [
        f
        for t in result["chunks"]
        if t["chunk_id"] == row.id
        for f in t["fields"]
        if f["field"] == "clause_label"
    ]
    if len(selected) != 1:
        raise ValueError(
            "Prior metadata ACK does not contain its exact defective field"
        )
    fact = selected[0]
    evidence = catalog[fact["evidence_id"]]
    if (
        fact["value"] != row.chunk_metadata["clause_label"]
        or fact["basis"] != entry["basis"]
        or fact.get("reference_evidence_id")
        or evidence.text != entry["evidence_quote"]
        or evidence.source_chunk_id != entry["source_chunk_id"]
        or evidence.scope != entry["evidence_scope"]
    ):
        raise ValueError("Prior metadata selected evidence differs from its saved fact")
    return literal_base.request_hash, attempt, checkpoint_hash


def _request_contract_retry(session: Session, batch: RegulatoryDocumentIntake) -> bool:
    """Only the explicit idle owner Retry may freeze a new repair request lineage."""
    from onyx.db.document_intake_metadata_checkpoint import (
        try_reserve_metadata_execution,
    )
    from onyx.db.document_intake_review import has_active_publication_decisions
    from onyx.regulatory.document_intake.metadata_job import metadata_recipe_sha256

    existing = contract_retry(batch)
    if existing is not None:
        require_current_contract(
            session, batch, require_complete=False, verify_ack_bytes=True
        )
        if batch.stage == "metadata" and batch.status == "preparing":
            return True
        if batch.stage == "metadata" and batch.status == "failed":
            if (
                batch.lease_token is not None
                or batch.lease_expires_at is not None
                or not try_reserve_metadata_execution(session, batch.id)
            ):
                raise intake.IntakeConflictError(
                    "Contract metadata execution is not idle"
                )
            # A duplicate explicit Retry reuses literal requests and their cached attempt budget.
            batch.status, batch.error = "preparing", None
            batch.metadata_recovery_count = 1
            session.flush()
            return True
        return False
    if (
        batch.metadata_checkpoint is None
        or batch.labeling_run_id is None
        or batch.stage not in {"labeling", "review"}
    ):
        return False
    # Unrelated normal labeling retries retain their existing route and checkpoint.
    probe = list(
        session.scalars(
            select(RegulatoryChunk)
            .join(
                RegulatoryDocumentIntakeFile,
                RegulatoryDocumentIntakeFile.user_file_id
                == RegulatoryChunk.user_file_id,
            )
            .where(RegulatoryDocumentIntakeFile.intake_id == batch.id)
        )
    )
    sources = {c.chunk_id: c for c in metadata_chunks(probe)}
    if not any(
        isinstance(value := row.chunk_metadata.get("clause_label"), str)
        and isinstance(
            entry := row.chunk_metadata.get(METADATA_AUDIT_KEY, {})
            .get("fields", {})
            .get("clause_label"),
            dict,
        )
        and entry.get("source") == "llm"
        and (
            selected_clause_evidence_contradicts(
                value=value,
                evidence_quote=entry["evidence_quote"],
                source=sources.get(entry["source_chunk_id"]),
            )
            or target_clause_evidence_contradicts(value=value, target=sources[row.id])
        )
        for row in probe
        if row.id in sources
    ):
        return False
    if (
        batch.lease_token is not None
        or batch.lease_expires_at is not None
        or has_active_publication_decisions(session, batch)
        or not try_reserve_metadata_execution(session, batch.id)
    ):
        raise intake.IntakeConflictError(
            "Metadata contract repair requires idle unpublished execution"
        )
    previous = MetadataCheckpoint.model_validate(batch.metadata_checkpoint)
    previous.validate_checksum()
    members, by_member = _owned_scope(
        session, batch, previous, previous_run_id=batch.labeling_run_id
    )
    binding = resolve_intake_labeling_provider_binding(
        session,
        batch.labeling_model_configuration_id,
        user=session.get_one(User, batch.created_by),
    )
    if binding.model_dump(mode="json") != previous.provider_binding:
        raise intake.IntakeConflictError("Metadata contract repair provider changed")
    invalid = {mid: _invalid_fields(rows) for mid, rows in by_member.items()}
    count = sum(len(v) for v in invalid.values())
    if not count:
        return False
    if count > MAX_CONTRACT_TARGETS:
        raise intake.IntakeConflictError(
            "Metadata contract repair exceeds its bounded target limit"
        )
    intent_id, documents, raw_cache, total_bytes = uuid4(), {}, {}, 0
    for member in members:
        mid = str(member.id)
        if not invalid[mid]:
            continue
        rows = by_member[mid]
        original_chunks = metadata_chunks(rows)
        frozen = [
            c.model_copy(
                update={
                    "structural_metadata": {
                        k: v
                        for k, v in c.structural_metadata.items()
                        if k != "clause_label" or c.chunk_id not in invalid[mid]
                    }
                }
            )
            for c in original_chunks
        ]
        targets = {}
        for target, (row, entry, reason) in invalid[mid].items():
            cache_id = metadata_ack_file_id(
                tenant_id=get_current_tenant_id(),
                intake_id=batch.id,
                document_id=member.id,
                binding_sha256=binding.fingerprint,
                request_hash=entry["request_hash"],
            )
            if cache_id not in raw_cache:
                raw_cache[cache_id] = _read_ack(session, cache_id)
                total_bytes += len(raw_cache[cache_id])
                if total_bytes > MAX_CONTRACT_BYTES:
                    raise intake.IntakeConflictError(
                        "Metadata contract prior ACK proof exceeds its byte budget"
                    )
            raw = raw_cache[cache_id]
            base_hash, attempt, checkpoint_hash = _previous_lineage(
                raw,
                binding=binding,
                cache_id=cache_id,
                row=row,
                entry=entry,
                chunks=original_chunks,
                previous_hashes=previous.request_hashes[mid],
            )
            pending = {**entry, "stale": True}
            targets[target] = ContractInvalidClause(
                reason=reason,
                previous_value=row.chunk_metadata["clause_label"],
                previous_fact=entry,
                previous_request_hash=entry["request_hash"],
                previous_base_request_hash=base_hash,
                previous_checkpoint_request_hash=checkpoint_hash,
                previous_repair_attempt=attempt,
                previous_raw_ack_sha256=hashlib.sha256(raw).hexdigest(),
                previous_cache_file_id=cache_id,
                before_audit_sha256=metadata_digest(entry),
                pending_audit_sha256=metadata_digest(pending),
            )
        documents[mid] = ContractRetryDocument(
            user_file_id=member.user_file_id,
            input_sha256=metadata_input_hash(frozen),
            canonical_source_sha256=canonical_metadata_source_digest(rows),
            preserved_metadata_sha256=_preserved_metadata(rows, set(targets)),
            chunks=frozen,
            targets=targets,
            request_hashes={
                target: build_contract_retry_request(
                    intent_id, frozen, target
                ).request_hash
                for target in targets
            },
        )
    intent = MetadataContractRetry(
        id=intent_id,
        intake_id=batch.id,
        package_id=batch.source_package_id,
        owner_id=batch.created_by,
        previous_labeling_run_id=batch.labeling_run_id,
        previous_checkpoint=previous,
        recipe_sha256=metadata_recipe_sha256(),
        scope_metadata_sha256={
            mid: _preserved_metadata(rows, set(invalid[mid]))
            for mid, rows in by_member.items()
        },
        documents=documents,
    )
    checkpoint = MetadataCheckpoint.build(
        recipe_sha256=intent.recipe_sha256,
        provider_binding=previous.provider_binding,
        source_snapshot={
            **previous.source_snapshot,
            CONTRACT_RETRY_KEY: intent.model_dump(mode="json"),
        },
        request_hashes=previous.request_hashes,
        predecessor=previous,
    )
    if len(checkpoint.model_dump_json().encode()) > MAX_CONTRACT_BYTES:
        raise intake.IntakeConflictError(
            "Metadata contract checkpoint exceeds its byte budget"
        )
    for mid, document in documents.items():
        for row in by_member[mid]:
            if row.id in document.targets:
                metadata = deepcopy(row.chunk_metadata)
                metadata[METADATA_AUDIT_KEY]["fields"]["clause_label"]["stale"] = True
                row.chunk_metadata = metadata
    batch.metadata_checkpoint = checkpoint.model_dump(mode="json")
    batch.status, batch.stage, batch.error = "preparing", "metadata", None
    batch.review_sha256 = None
    batch.metadata_recovery_count = 1
    for member in members:
        member.status, member.error = "chunked", None
    session.flush()
    return True


def request_contract_retry(session: Session, batch: RegulatoryDocumentIntake) -> bool:
    try:
        return _request_contract_retry(session, batch)
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        NoResultFound,
        BotoCoreError,
        ClientError,
    ) as error:
        raise intake.IntakeConflictError(
            "The prior metadata ACK, source or contract proof is unavailable or changed"
        ) from error


def _applied(
    session: Session,
    row: RegulatoryChunk,
    intent: MetadataContractRetry,
    document: ContractRetryDocument,
    member_id: UUID,
    binding: LabelingProviderBinding,
    verify_ack_bytes: bool,
) -> bool:
    audit = row.chunk_metadata[METADATA_AUDIT_KEY]
    stamp = audit.get(CONTRACT_RETRY_KEY)
    target = document.targets[row.id]
    entry = audit["fields"].get("clause_label")
    if stamp is None:
        if (
            metadata_digest(entry) != target.pending_audit_sha256
            or row.chunk_metadata.get("clause_label") != target.previous_value
        ):
            raise intake.IntakeConflictError("Contract metadata pending field changed")
        return False
    receipt = ContractAppliedClause.model_validate(stamp)
    if (
        receipt.intent_id != intent.id
        or receipt.intent_sha256 != intent.intent_sha256
        or receipt.base_request_hash != document.request_hashes[row.id]
        or receipt.policy != intent.policy
    ):
        raise intake.IntakeConflictError("Contract metadata applied authority changed")
    expected_request = build_contract_retry_request(intent.id, document.chunks, row.id)
    if receipt.output_reserve:
        reserved = build_metadata_output_reserve_request(expected_request)
        if reserved is None:
            raise intake.IntakeConflictError("Contract metadata output reserve changed")
        expected_request = reserved
    if receipt.validation_feedback is not None:
        payload = json.loads(expected_request.prompt)
        payload["validation_feedback"] = receipt.validation_feedback
        expected_request = expected_request.model_copy(
            update={
                "prompt": json.dumps(
                    payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
            }
        )
    require_contract_accepted_request(
        expected_request,
        contract_id=intent.id,
        chunks=document.chunks,
        target_id=row.id,
    )
    if expected_request.request_hash != receipt.accepted_request_hash:
        raise intake.IntakeConflictError(
            "Contract metadata accepted request descriptor changed"
        )
    cache_id = metadata_ack_file_id(
        tenant_id=get_current_tenant_id(),
        intake_id=intent.intake_id,
        document_id=member_id,
        binding_sha256=binding.fingerprint,
        request_hash=receipt.accepted_request_hash,
    )
    record = session.get(FileRecord, cache_id, populate_existing=True)
    if (
        record is None
        or not isinstance(record.file_metadata, dict)
        or cast(dict[str, Any], record.file_metadata).get("sha256")
        != receipt.raw_ack_sha256
    ):
        raise intake.IntakeConflictError(
            "Contract metadata accepted ACK identity changed"
        )
    if receipt.omitted:
        if entry is not None or "clause_label" in row.chunk_metadata:
            raise intake.IntakeConflictError(
                "An omitted contract clause value was restored"
            )
        result = LegalMetadataResult(
            chunks=[ChunkLegalMetadata(chunk_id=row.id, fields=[])]
        )
    else:
        if (
            not isinstance(entry, dict)
            or entry.get("source") != "llm"
            or entry.get("stale") is not False
            or entry.get("evidence_scope") != "body"
            or entry.get("request_hash") != receipt.accepted_request_hash
        ):
            raise intake.IntakeConflictError("Contract metadata accepted fact changed")
        fact = _fact(row, "clause_label", entry)
        validate_repaired_clause_fact(
            fact, source=next(c for c in document.chunks if c.chunk_id == row.id)
        )
        result = LegalMetadataResult(
            chunks=[ChunkLegalMetadata(chunk_id=row.id, fields=[fact])]
        )
    if (
        metadata_digest(result.model_dump(mode="json", exclude={"warnings"}))
        != receipt.result_sha256
    ):
        raise intake.IntakeConflictError("Contract metadata accepted result changed")
    if verify_ack_bytes:
        raw = _read_ack(session, cache_id)
        actual, _payload, response = _ack(
            raw,
            binding=binding,
            cache_id=cache_id,
            expected_hash=receipt.accepted_request_hash,
        )
        from onyx.regulatory.document_intake.metadata_contract_retry import (
            validate_contract_retry_response,
        )

        received = validate_contract_retry_response(
            response, chunks=document.chunks, target_id=row.id
        )
        if (
            actual != expected_request
            or hashlib.sha256(raw).hexdigest() != receipt.raw_ack_sha256
            or metadata_digest(received.model_dump(mode="json", exclude={"warnings"}))
            != receipt.result_sha256
        ):
            raise intake.IntakeConflictError(
                "Contract metadata accepted ACK no longer proves its field"
            )
    return True


def _require_current_contract(
    session: Session,
    batch: RegulatoryDocumentIntake,
    *,
    require_complete: bool,
    verify_ack_bytes: bool,
    require_fresh_labeling: bool,
) -> tuple[MetadataContractRetry, list[tuple[UUID, str]]]:
    from onyx.regulatory.document_intake.metadata_job import metadata_recipe_sha256

    intent = contract_retry(batch)
    if intent is None or intent.recipe_sha256 != metadata_recipe_sha256():
        raise intake.IntakeConflictError("Contract metadata validator changed")
    if (
        require_fresh_labeling
        and batch.labeling_run_id == intent.previous_labeling_run_id
    ):
        raise intake.IntakeConflictError(
            "Contract metadata repair requires its fresh labeling generation"
        )
    if require_fresh_labeling:
        current = session.get_one(
            RegulatoryLabelingRun, batch.labeling_run_id, populate_existing=True
        )
        if (
            current.status != "completed"
            or current.failed_chunks
            or current.stale_chunks
            or current.unresolved_derived_chunks
        ):
            raise intake.IntakeConflictError(
                "Contract metadata repair labeling is not complete"
            )
    binding = resolve_intake_labeling_provider_binding(
        session,
        batch.labeling_model_configuration_id,
        user=session.get_one(User, batch.created_by),
    )
    if binding.model_dump(mode="json") != intent.previous_checkpoint.provider_binding:
        raise intake.IntakeConflictError("Contract metadata provider changed")
    members, by_member = _owned_scope(
        session,
        batch,
        intent.previous_checkpoint,
        previous_run_id=intent.previous_labeling_run_id,
        allow_fresh_run=True,
    )
    pending = []
    for member in members:
        mid, rows = str(member.id), by_member[str(member.id)]
        chunks = metadata_chunks(rows)
        canonical_ids = {c.chunk_id for c in chunks}
        input_hash = metadata_input_hash(chunks)
        document = intent.documents.get(mid)
        targets = set(document.targets) if document else set()
        if _preserved_metadata(rows, targets) != intent.scope_metadata_sha256[mid]:
            raise intake.IntakeConflictError(
                "Unreviewed metadata changed during contract repair"
            )
        if document is not None:
            frozen_current = [
                c.model_copy(
                    update={
                        "structural_metadata": {
                            k: v
                            for k, v in c.structural_metadata.items()
                            if k != "clause_label" or c.chunk_id not in targets
                        }
                    }
                )
                for c in chunks
            ]
            if (
                document.user_file_id != member.user_file_id
                or document.chunks != frozen_current
                or document.input_sha256 != input_hash
                or document.canonical_source_sha256
                != canonical_metadata_source_digest(rows)
                or document.preserved_metadata_sha256
                != intent.scope_metadata_sha256[mid]
            ):
                raise intake.IntakeConflictError(
                    "Contract metadata document identity changed"
                )
            for row in rows:
                if row.id in targets and not _applied(
                    session, row, intent, document, member.id, binding, verify_ack_bytes
                ):
                    pending.append((member.id, row.id))
        for row in rows:
            if row.id not in canonical_ids:
                continue
            audit = row.chunk_metadata.get(METADATA_AUDIT_KEY)
            if (
                not isinstance(audit, dict)
                or audit.get("input_stale") is not False
                or audit.get("input_sha256") != input_hash
                or audit.get("model") != METADATA_MODEL
                or audit.get("version") != METADATA_VERSION
            ):
                raise intake.IntakeConflictError(
                    "Contract metadata source receipt changed"
                )
            # Only the explicitly pending clause is excluded, never another field on its row.
            facts = [
                _fact(row, k, e)
                for k, e in audit["fields"].items()
                if e.get("source") == "llm"
                and not (k == "clause_label" and (member.id, row.id) in pending)
            ]
            if any(
                e.get("source") == "llm"
                and e.get("stale") is not False
                and not (k == "clause_label" and (member.id, row.id) in pending)
                for k, e in audit["fields"].items()
            ):
                raise intake.IntakeConflictError(
                    "Unrelated stale metadata prevents contract repair"
                )
            validate_metadata_response(
                LegalMetadataResult(
                    chunks=[ChunkLegalMetadata(chunk_id=row.id, fields=facts)]
                ).model_dump_json(),
                chunks=chunks,
                target_ids=[row.id],
            )
    if require_complete and pending:
        raise intake.IntakeConflictError("Contract metadata repair is incomplete")
    return intent, pending


def require_current_contract(
    session: Session,
    batch: RegulatoryDocumentIntake,
    *,
    require_complete: bool,
    verify_ack_bytes: bool = False,
    require_fresh_labeling: bool = False,
) -> tuple[MetadataContractRetry, list[tuple[UUID, str]]]:
    try:
        return _require_current_contract(
            session,
            batch,
            require_complete=require_complete,
            verify_ack_bytes=verify_ack_bytes,
            require_fresh_labeling=require_fresh_labeling,
        )
    except (ValueError, KeyError, TypeError, NoResultFound) as error:
        raise intake.IntakeConflictError(
            "Contract metadata evidence or identity changed"
        ) from error


def contract_retry_snapshot(
    session: Session, batch: RegulatoryDocumentIntake
) -> dict[str, Any] | None:
    intent = contract_retry(batch)
    if intent is None:
        return None
    proof: dict[str, Any] = {
        "version": 1,
        "id": str(intent.id),
        "policy": intent.policy,
        "intent_sha256": intent.intent_sha256,
        "recipe_sha256": intent.recipe_sha256,
        "previous_checkpoint_sha256": intent.previous_checkpoint.checkpoint_sha256,
        "previous_recipe_sha256": intent.previous_checkpoint.recipe_sha256,
        "previous_labeling_run_id": str(intent.previous_labeling_run_id),
        "historical_request_count": sum(
            map(len, intent.previous_checkpoint.request_hashes.values())
        ),
        "document_count": len(intent.documents),
        "target_count": sum(len(d.targets) for d in intent.documents.values()),
        "documents": {
            mid: {
                "user_file_id": str(d.user_file_id),
                "input_sha256": d.input_sha256,
                "canonical_source_sha256": d.canonical_source_sha256,
                "requests": d.request_hashes,
            }
            for mid, d in intent.documents.items()
        },
        "complete": False,
        "current_strict_postcondition": False,
        "ack_authority": "frozen_request_descriptor_and_cache_sha256",
        "ack_bytes_replayed": False,
        "phase": "current_unpublished",
        "current_repair_authority": True,
    }
    if batch.stage in {"indexing", "completed", "lineage"} or batch.status in {
        "approving",
        "completed",
    }:
        proof.update(
            {
                "phase": "historical_after_approval",
                "complete": None,
                "current_strict_postcondition": None,
                "current_repair_authority": False,
                "validation_basis": "Bind the preapproval contract ACK proof to unchanged postpublication canonical source and checkpoint hashes",
            }
        )
        recorded = {}
        for document in intent.documents.values():
            for target in document.targets:
                row = session.get(RegulatoryChunk, target)
                recorded[target] = (
                    row.chunk_metadata.get(METADATA_AUDIT_KEY, {}).get(
                        CONTRACT_RETRY_KEY
                    )
                    if row is not None
                    else None
                )
        proof["recorded_target_receipts"] = recorded
        return proof
    try:
        _intent, pending = require_current_contract(
            session, batch, require_complete=False
        )
        proof.update(
            {
                "complete": not pending,
                "current_strict_postcondition": not pending,
                "pending_count": len(pending),
            }
        )
    except intake.IntakeConflictError:
        proof["current_repair_authority"] = False
        proof["error"] = (
            "The contract repair is not current under its unpublished source authority"
        )
    return proof


def apply_contract_result(
    session: Session,
    *,
    intake_id: UUID,
    lease_token: UUID,
    document_id: UUID,
    target_id: str,
    request: VertexBatchRequest,
    result: LegalMetadataResult,
) -> None:
    from onyx.db.document_intake_metadata_checkpoint import require_metadata_lease

    batch = require_metadata_lease(session, intake_id, lease_token)
    intent, pending = require_current_contract(session, batch, require_complete=False)
    document = intent.documents[str(document_id)]
    require_contract_accepted_request(
        request, contract_id=intent.id, chunks=document.chunks, target_id=target_id
    )
    validate_contract_retry_result(result, chunks=document.chunks, target_id=target_id)
    validate_metadata_response(
        result.model_dump_json(), chunks=document.chunks, target_ids=[target_id]
    )
    binding = LabelingProviderBinding.model_validate(
        intent.previous_checkpoint.provider_binding
    )
    cache_id = metadata_ack_file_id(
        tenant_id=get_current_tenant_id(),
        intake_id=batch.id,
        document_id=document_id,
        binding_sha256=binding.fingerprint,
        request_hash=request.request_hash,
    )
    raw = _read_ack(session, cache_id)
    actual, _payload, response = _ack(
        raw, binding=binding, cache_id=cache_id, expected_hash=request.request_hash
    )
    from onyx.regulatory.document_intake.metadata_contract_retry import (
        validate_contract_retry_response,
    )

    if (
        actual != request
        or validate_contract_retry_response(
            response, chunks=document.chunks, target_id=target_id
        )
        != result
    ):
        raise intake.IntakeConflictError(
            "Contract metadata accepted ACK differs from its result"
        )
    row = session.get_one(
        RegulatoryChunk, target_id, with_for_update=True, populate_existing=True
    )
    if (document_id, target_id) not in pending:
        receipt = ContractAppliedClause.model_validate(
            row.chunk_metadata[METADATA_AUDIT_KEY][CONTRACT_RETRY_KEY]
        )
        if (
            receipt.accepted_request_hash != request.request_hash
            or receipt.raw_ack_sha256 != hashlib.sha256(raw).hexdigest()
            or receipt.result_sha256
            != metadata_digest(result.model_dump(mode="json", exclude={"warnings"}))
        ):
            raise intake.IntakeConflictError(
                "A duplicate contract apply changed its accepted receipt"
            )
        return
    metadata = deepcopy(row.chunk_metadata)
    audit = metadata[METADATA_AUDIT_KEY]
    fields = result.chunks[0].fields
    if fields:
        fact = fields[0]
        metadata["clause_label"] = fact.value
        audit["fields"]["clause_label"] = {
            "source": "llm",
            "request_hash": request.request_hash,
            "basis": fact.basis,
            "evidence_quote": fact.evidence_quote,
            "source_chunk_id": fact.source_chunk_id,
            "evidence_scope": "body",
            "reference_quote": fact.reference_quote,
            "reference_chunk_id": fact.reference_chunk_id,
            "stale": False,
        }
    else:
        metadata.pop("clause_label", None)
        audit["fields"].pop("clause_label", None)
    audit[CONTRACT_RETRY_KEY] = {
        "version": 1,
        "intent_id": str(intent.id),
        "intent_sha256": intent.intent_sha256,
        "base_request_hash": document.request_hashes[target_id],
        "accepted_request_hash": request.request_hash,
        "raw_ack_sha256": hashlib.sha256(raw).hexdigest(),
        "result_sha256": metadata_digest(
            result.model_dump(mode="json", exclude={"warnings"})
        ),
        "omitted": not fields,
        "policy": intent.policy,
        "output_reserve": request.generation_config
        != build_contract_retry_request(
            intent.id, document.chunks, target_id
        ).generation_config,
        **(
            {"validation_feedback": json.loads(request.prompt)["validation_feedback"]}
            if "validation_feedback" in json.loads(request.prompt)
            else {}
        ),
    }
    row.chunk_metadata = metadata
    session.flush()
    require_current_contract(session, batch, require_complete=False)
