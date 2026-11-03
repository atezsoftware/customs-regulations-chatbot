"""Inspect or retry only the frozen failed DEV Resmi Gazete intake."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any
from uuid import UUID, uuid5

from sqlalchemy import text
from sqlalchemy.orm import Session

from onyx.configs.app_configs import POSTGRES_DB
from onyx.db import document_intake as repository
from onyx.db.engine.sql_engine import SqlEngine, get_session_with_current_tenant
from onyx.db.models import (
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeReprepare,
    RegulatoryLabelingRun,
)
from onyx.db.search_settings import get_current_search_settings
from onyx.indexing.chunker import DEFAULT_CONTEXTUAL_RAG_RESERVED_TOKENS
from onyx.natural_language_processing.utils import get_tokenizer
from onyx.regulatory.amendments.annexes import config
from onyx.regulatory.contextual import contextual_reserve_for_embedding_text
from onyx.utils.variable_functionality import set_is_ee_based_on_env_variable
from shared_configs.configs import DOC_EMBEDDING_CONTEXT_SIZE
from shared_configs.contextvars import get_current_tenant_id

BATCH = UUID("f32df3a9-4644-4fd3-be07-6aaa966d3107")
PACKAGE = "d75a8cb8-b268-4d24-a230-cfa9bf301189"
OWNER = "9ea33f67-5578-4dc4-bbea-e32b16f3f1dc"
MAIN_FILE = "6cb352ab-1331-445d-8dee-7d2c2fbe8a87"
PDF_FILE = "8cbb0ca5-7e37-4880-aef6-72f816af7539"
MAIN_DOCUMENT = "f8d4be65-ec82-4011-a4c9-da10c8998e2e"
PDF_DOCUMENT = "fd1fefc8-1e2f-41f7-a09f-d36c3f6152e2"
TARGET_ROW_SHA256 = "fe6b6725b8104825be247dc3fcdb8122659030f108e845dc0b8ae21b303b004a"
REPREPARE_KEY = uuid5(BATCH, "explicit-whole-table-v4-reprepare-20261102")
REPREPARE_REQUEST = "a03a7101-f9a7-46cf-bbda-831d96e64865"
REPREPARE_REVIEW_SHA = "1188de8875d91317e9f9cb9bc9cbf1057e9a4a47b40092de755bb8235b0b1462"
MAIN_CANONICAL_SHA = "c987526ff76ea81f1cee005397a45282c7ecb718d34c836b8ecf7367773210f2"
MAIN_METADATA_SHA = "21b05187e74424039e086570c512e55b0e20abce4c43afb06e383c86857b9ea0"
REFRESH_IMAGE_SHA = "ff935c6d49088617d680e7d5d03763155fc04a99"
LABELING_RUN = UUID("8620ccb4-c0c0-4a37-a403-5027d75d6f69")
FK_INDEXES = {
    "ix_regulatory_chunk_supersedes_chunk_id": ("regulatory_chunk", "supersedes_chunk_id"),
    "ix_regulatory_chunk_superseded_by_chunk_id": ("regulatory_chunk", "superseded_by_chunk_id"),
    "ix_user_file_batch_item_regulatory_chunk_id": ("user_file_batch_item", "regulatory_chunk_id"),
    "ix_regulatory_indexing_item_regulatory_chunk_id": ("regulatory_indexing_item", "regulatory_chunk_id"),
    "ix_amendment_proposal_applied_new_chunk_id": ("amendment_proposal", "applied_new_chunk_id"),
    "ix_regulatory_annex_element_chunk_chunk_id": ("regulatory_annex_element_chunk", "chunk_id"),
}
MAIN_DIGEST = "f70e13400f34aec9878368a92a040d76b6a6d9363f70c7f06e6898b878f16ba3"
ORIGINAL_ERROR = (
    "An intake table row exceeds the embedding content token limit; "
    "only a single source cell with safe line breaks can continue"
)
EXPECTED_ASSETS = {
    "7743688a-0cdf-4b8a-8315-b292b21724ba": (
        "ab52bd164678654c6ce02d3f7c4d1409d56426269c03c36b5192211f2e1b8e43",
        "65bbac4dbdc075ca78b5e917069af8bb4d01112bf9cc71ef294aab87672baa58",
        PDF_FILE,
    ),
    "ed12363b-36f7-4d93-a377-1ae01e3c6d2b": (
        "d0196f35a106128da7c68761c7f9e8aeb0de143a7dffde863d4ed9db8edbece7",
        "24c2109a77e06700ca14809a2452d3b442b7a38480cb69d788a690de5a79c82d",
        MAIN_FILE,
    ),
}
SECRET_LINE = re.compile(r"api[ _-]?key|password|secret|authorization|bearer|access[ _-]?token|credential|vault", re.I)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def retained_source_stream(file_id: str, session: Session) -> IO[bytes]:
    from onyx.file_store.file_store import S3BackedFileStore, get_default_file_store
    from onyx.file_store.postgres_file_store import PostgresBackedFileStore

    store = get_default_file_store()
    if isinstance(store, (PostgresBackedFileStore, S3BackedFileStore)):
        return store.read_file(file_id, db_session=session)
    return store.read_file(file_id)


def rows(session: Session, query: str, **params: object) -> list[dict[str, Any]]:
    return [dict(row) for row in session.execute(text(query), params).mappings()]


def main_identity(session: Session) -> tuple[str, int]:
    chunks = rows(session, """
        SELECT id,user_file_id,chunk_type,position,heading_path,text,source
        FROM regulatory_chunk WHERE user_file_id=:file ORDER BY user_file_id,position
    """, file=MAIN_FILE)
    serialized = json.dumps(chunks, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode()).hexdigest(), len(chunks)


def verify_scope(session: Session, batch: RegulatoryDocumentIntake) -> list[dict[str, Any]]:
    require(str(batch.source_package_id) == PACKAGE, "Frozen intake package changed")
    require(str(batch.created_by) == OWNER, "Frozen intake owner changed")
    require(batch.document_set_id == 15, "Frozen intake document set changed")
    package = rows(session, """
        SELECT environment,status,manifest_sha256,issues FROM amendment_source_package WHERE id=:id
    """, id=PACKAGE)
    require(len(package) == 1 and package[0]["environment"] == "dev", "Source package is not DEV")
    require(package[0]["status"] == "ready" and package[0]["issues"] == [], "Source package is not ready")
    require(package[0]["manifest_sha256"] == "a410f20b802e5e664e307915295f561a633954eed41a7fbaa4b66f25f8ce5fb9", "Frozen source manifest changed")
    assets = rows(session, """
        SELECT a.id,a.sha256,a.text_sha256,a.text_file_id,a.display_name,a.original_url,m.user_file_id,
               m.id AS document_id,m.role,m.parent_document_id,m.relationships,
               m.status AS intake_file_status,m.error AS intake_file_error,
               u.status AS user_file_status,u.chunk_count,u.regulatory_chunk_generation_hash
        FROM regulatory_source_asset a
        JOIN regulatory_document_intake_file m ON m.source_asset_id=a.id AND m.intake_id=:batch
        JOIN user_file u ON u.id=m.user_file_id
        WHERE a.package_id=:package ORDER BY a.id
    """, batch=BATCH, package=PACKAGE)
    actual = {str(a["id"]): (a["sha256"], a["text_sha256"], str(a["user_file_id"])) for a in assets}
    require(actual == EXPECTED_ASSETS, "Frozen source assets or owned files changed")
    by_file = {str(asset["user_file_id"]): asset for asset in assets}
    require(str(by_file[MAIN_FILE]["document_id"]) == MAIN_DOCUMENT and by_file[MAIN_FILE]["role"] == "root" and by_file[MAIN_FILE]["parent_document_id"] is None, "Main source relationship changed")
    require(str(by_file[PDF_FILE]["document_id"]) == PDF_DOCUMENT and by_file[PDF_FILE]["role"] == "annex" and str(by_file[PDF_FILE]["parent_document_id"]) == MAIN_DOCUMENT, "PDF attachment relationship changed")
    digest, count = main_identity(session)
    require(digest == MAIN_DIGEST and count == 365, "Existing main HTML chunk identity changed")
    return assets


def pdf_chunk_verification(session: Session) -> dict[str, Any]:
    settings = get_current_search_settings(session)
    tokenizer = get_tokenizer(model_name=settings.model_name, provider_type=settings.provider_type)
    report: dict[str, Any] = {
        "search_settings_id": settings.id,
        "model_name": settings.model_name,
        "provider_type": settings.provider_type.value if settings.provider_type else None,
        "tokenizer_class": type(tokenizer).__name__,
        "embedding_token_limit": DOC_EMBEDDING_CONTEXT_SIZE,
        "preferred_context_reserve": DEFAULT_CONTEXTUAL_RAG_RESERVED_TOKENS,
        "contextual_enabled": settings.enable_contextual_rag,
    }
    chunks = rows(session, """
        SELECT id,chunk_type,position,text,chunk_metadata
        FROM regulatory_chunk WHERE user_file_id=:file ORDER BY position,id
    """, file=PDF_FILE)
    if not chunks:
        report["state"] = "pdf_not_chunked_yet"
        return report
    require(DOC_EMBEDDING_CONTEXT_SIZE == 512 and DEFAULT_CONTEXTUAL_RAG_RESERVED_TOKENS == 200, "Production embedding/context budgets changed")
    require(settings.enable_contextual_rag, "Production contextual enrichment is disabled")
    atomic = [chunk for chunk in chunks if chunk["chunk_type"] != "hierarchical_aggregate" and chunk["chunk_metadata"].get("chunk_variant") != "hierarchical_aggregate"]
    profiles = {chunk["chunk_metadata"].get("intake_chunking_version") for chunk in chunks}
    require(profiles in ({"3"}, {"4"}), "PDF chunk profiles are inconsistent")
    profile = next(iter(profiles))
    expected = (332, 320) if profile == "4" else (406, 395)
    require((len(chunks), len(atomic)) == expected, "PDF chunk counts differ from the verified exact-source replay")
    matches = [(chunk, line) for chunk in atomic for line in chunk["text"].splitlines() if hashlib.sha256(line.encode()).hexdigest() == TARGET_ROW_SHA256]
    require(len(matches) == 1, "Exact tariff row is missing, split or duplicated")
    target, target_line = matches[0]
    tokens = len(tokenizer.encode(target_line))
    require(tokens == 314 and target_line.startswith("| 8544 ila 8548 |") and target_line.count("|") == 4, "Tariff row source or three-cell relationship changed")
    if profile == "4":
        from onyx.regulatory.document_intake.embedding_policy import (
            embedding_token_limit_for_metadata,
        )
        limits = [embedding_token_limit_for_metadata(chunk["chunk_metadata"], chunk["chunk_type"], provider=settings.provider_type.value if settings.provider_type else None, model_name=settings.model_name) for chunk in chunks]
        require(sum(chunk["chunk_type"] == "table" for chunk in atomic) == 68, "Whole-table source block count changed")
    else:
        limits = [DOC_EMBEDDING_CONTEXT_SIZE] * len(chunks)
    target_limit = limits[chunks.index(target)]
    reserve = contextual_reserve_for_embedding_text(target["text"], tokenizer=tokenizer, embedding_token_limit=target_limit, requested_reserve=DEFAULT_CONTEXTUAL_RAG_RESERVED_TOKENS)
    require(reserve >= 198, "Target source lost its useful contextual capacity")
    content_token_counts = [len(tokenizer.encode(chunk["text"])) for chunk in chunks]
    require(all(count <= limit for count, limit in zip(content_token_counts, limits, strict=True)), "A persisted PDF chunk exceeds its frozen embedding budget")
    report.update({"state": "verified", "chunking_version": profile, "chunk_count": len(chunks), "canonical_chunk_count": len(atomic), "target_chunk_id": target["id"], "target_row_sha256": TARGET_ROW_SHA256, "target_content_tokens": tokens, "target_chunk_content_tokens": len(tokenizer.encode(target["text"])), "target_final_input_limit": target_limit, "target_context_reserve": reserve, "maximum_content_tokens": max(content_token_counts), "target_three_cells_preserved": True})
    return report


def metadata_cache_progress(session: Session, batch: RegulatoryDocumentIntake) -> dict[str, Any]:
    # Submission names encode the full intake/document identity and permit exact
    # correlation without opening provider response blobs or invoking a model.
    from onyx.regulatory.document_intake.metadata import METADATA_VERSION
    inventory = rows(session, """
        SELECT file_id,created_at,updated_at,file_metadata->>'binding_sha256' AS binding
        FROM file_record
        WHERE file_id LIKE 'regulatory-labeling-direct/%'
          AND file_id NOT LIKE '%/manifest.json'
          AND created_at>=:since
        ORDER BY created_at DESC,file_id DESC LIMIT 1001
    """, since=batch.created_at)
    bounded = len(inventory) > 1000
    inventory = inventory[:1000]
    by_document: dict[str, list[dict[str, Any]]] = {MAIN_DOCUMENT: [], PDF_DOCUMENT: []}
    for record in inventory:
        match = re.fullmatch(r"regulatory-labeling-direct/([0-9a-f]{64})/([0-9a-f]{64})/([0-9a-f]{64})\.json", record["file_id"])
        if match is None:
            continue
        binding, submission, request_hash = match.groups()
        if record["binding"] != binding:
            continue
        for document in by_document:
            identity = f"metadata-v{METADATA_VERSION}:public:{BATCH}:{document}:{binding}"
            if hashlib.sha256(f"{identity}:{request_hash}".encode()).hexdigest() == submission:
                by_document[document].append(record)
                break
    return {
        "inventory_rows_scanned": len(inventory),
        "inventory_limit_reached_counts_are_lower_bounds": bounded,
        "source": "exact-correlated metadata response cache records; repairs/splits also count as requests",
        "documents": [{"document_id": document, "correlated_cached_requests": len(records), "latest_cached_at": max((record["updated_at"] for record in records), default=None)} for document, records in by_document.items()],
    }


def snapshot(session: Session, batch: RegulatoryDocumentIntake) -> dict[str, Any]:
    assets = verify_scope(session, batch)
    report: dict[str, Any] = {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "database": session.execute(text("SELECT current_database()")).scalar_one(),
        "batch": {key: getattr(batch, key) for key in ("id", "source_package_id", "document_set_id", "status", "stage", "error", "labeling_run_id", "review_sha256", "created_at", "updated_at")},
        "main_chunk_sha256": MAIN_DIGEST,
        "pdf_chunk_verification": pdf_chunk_verification(session),
        "metadata_cache_progress": metadata_cache_progress(session, batch),
        "assets": assets,
        "chunks": rows(session, """
            SELECT c.user_file_id,c.chunk_type,c.status,count(*) AS count,
                   count(*) FILTER (WHERE c.chunk_metadata ? 'intake_legal_metadata') AS metadata_count,
                   count(*) FILTER (WHERE c.chunk_metadata ? 'intake_table_row') AS continued_table_rows
            FROM regulatory_chunk c WHERE c.user_file_id IN (:main,:pdf)
            GROUP BY c.user_file_id,c.chunk_type,c.status ORDER BY c.user_file_id,c.chunk_type,c.status
        """, main=MAIN_FILE, pdf=PDF_FILE),
        "labeling": rows(session, """
            SELECT id,status,stage,total_chunks,completed_chunks,failed_chunks,stale_chunks,
                   derived_chunks,unresolved_derived_chunks,error,updated_at
            FROM regulatory_labeling_run WHERE id=:id
        """, id=batch.labeling_run_id) if batch.labeling_run_id else [],
        "indexing": rows(session, """
            SELECT id,user_file_id,status,stage,error_code,error_message,search_settings_id,created_at,updated_at
            FROM regulatory_indexing_job WHERE user_file_id IN (:main,:pdf) ORDER BY user_file_id,created_at
        """, main=MAIN_FILE, pdf=PDF_FILE),
    }
    return report



def embedding_canary() -> dict[str, Any]:
    """Exercise the configured encoder without publishing or indexing a document."""
    import math

    from onyx.indexing.embedder import DefaultIndexingEmbedder
    from onyx.regulatory.contextual import fit_context_fields_to_embedding_budget
    from shared_configs.enums import EmbedTextType

    with get_session_with_current_tenant() as session:
        session.execute(text("SET TRANSACTION READ ONLY"))
        candidates = rows(session, "SELECT text FROM regulatory_chunk WHERE user_file_id=:file AND text LIKE :prefix", file=PDF_FILE, prefix="| 8544 ila 8548 |%")
        require(len(candidates) == 1, "Expected one exact row for the embedding canary")
        source = candidates[0]["text"]
        require(hashlib.sha256(source.encode()).hexdigest() == TARGET_ROW_SHA256, "Canary source changed")
        settings = get_current_search_settings(session)
        embedder = DefaultIndexingEmbedder.from_db_search_settings(settings)
        model_name, dimension = settings.model_name, settings.reduced_dimension
        session.rollback()
    tokenizer = embedder.embedding_model.tokenizer
    summary, context = fit_context_fields_to_embedding_budget(
        title_prefix="", content=source, metadata_suffix="", doc_summary="Ek II.",
        chunk_context="8544 ila 8548 kod aralığı, ürün tanımları ve aynı satırdaki imalat koşulu. " * 80,
        tokenizer=tokenizer, embedding_token_limit=512,
    )
    enriched = summary + source + context
    require(source in enriched and len(tokenizer.encode(enriched)) <= 512, "Final canary input exceeded its budget or lost source text")
    inputs = [source, enriched]
    vectors = embedder.embedding_model.encode(inputs, text_type=EmbedTextType.PASSAGE, tenant_id="public", request_id="rg-f32df3a9-20261102-canary")
    require(len(vectors) == 2 and all(vector and all(math.isfinite(value) for value in vector) for vector in vectors), "Embedding provider did not return two finite vectors")
    require(dimension is None or all(len(vector) == dimension for vector in vectors), "Embedding dimensions changed")
    return {"success": True, "model": model_name, "input_tokens": [len(tokenizer.encode(value)) for value in inputs], "vector_dimensions": [len(vector) for vector in vectors], "source_row_sha256": TARGET_ROW_SHA256, "enriched_input_sha256": hashlib.sha256(enriched.encode()).hexdigest(), "whole_source_row_preserved": True, "document_or_index_writes": False}



def review_budget_canary() -> dict[str, Any]:
    """Validate the persisted review inputs without saving an edit."""
    from sqlalchemy import select

    from onyx.db.document_intake_edit_budget import validate_intake_edit_budgets
    from onyx.db.models import RegulatoryChunk, UserFile

    with get_session_with_current_tenant() as session:
        session.execute(text("SET TRANSACTION READ ONLY"))
        identifiers = [UUID(MAIN_FILE), UUID(PDF_FILE)]
        files = {identifier: session.get_one(UserFile, identifier) for identifier in identifiers}
        prospective = {identifier: list(session.scalars(select(RegulatoryChunk).where(RegulatoryChunk.user_file_id == identifier).order_by(RegulatoryChunk.position))) for identifier in identifiers}
        counts = [len(prospective[identifier]) for identifier in identifiers]
        require(counts in ([365, 406], [365, 332]), "Review scope changed")
        validate_intake_edit_budgets(session, files, prospective)
        session.rollback()
    return {"success": True, "files": 2, "validated_chunks": sum(counts), "document_or_index_writes": False}


def request_v4_reprepare(session: Session, batch: RegulatoryDocumentIntake) -> tuple[dict[str, Any], UUID | None]:
    """Use the normal reviewed request only for this idle, unpublished incident."""
    from onyx.db.document_intake_reprepare import (
        latest_reprepare,
        reprepare_snapshot,
        request_reprepare,
    )
    from onyx.regulatory.document_intake.models import IntakePreparationPolicy

    previous = latest_reprepare(session, batch.id)
    if previous is not None:
        require(previous.idempotency_key == REPREPARE_KEY, "A different reprepare request exists")
        require(previous.status in {"queued", "running", "prepared"}, "Existing reprepare needs the standard retry/reconciliation flow")
        return {"operator_action": "existing_reprepare_inspected_only", "request_id": previous.id, "status": previous.status}, None
    require(batch.status == "failed" and batch.stage == "metadata", "Only the original failed-metadata intake may be upgraded")
    require(batch.error == "New-document metadata failed strict validation after bounded repair attempts", "The metadata failure signature changed")
    require(batch.review_sha256 is None and batch.labeling_run_id is None, "A later review or labeling stage already exists")
    require(pdf_chunk_verification(session)["chunking_version"] == "3", "The original v3 source revision is no longer current")
    preview = reprepare_snapshot(session, batch)
    policy = IntakePreparationPolicy().model_dump(mode="json")
    require(preview.get("target_preparation_policy") == policy, "The reviewed request does not target the exact whole-table v4 policy")
    require({entry["document_id"] for entry in preview["documents"]} == {MAIN_DOCUMENT, PDF_DOCUMENT}, "The reviewed document scope changed")
    expected_sha = preview["expected_sha256"]
    require(bool(re.fullmatch(r"[0-9a-f]{64}", expected_sha)), "The normal reprepare review SHA is missing")
    request = request_reprepare(session, batch, expected_sha256=expected_sha, idempotency_key=REPREPARE_KEY)
    require(request.status == "queued" and batch.stage == "reprepare", "The normal repository did not queue the expected reprepare")
    require(all(entry.get("target_chunking_version") == "4" and entry.get("target_preparation_policy") == policy for entry in request.documents.values()), "The durable request differs from its reviewed v4 policy")
    verify_scope(session, batch)
    return {"operator_action": "normal_v4_reprepare_requested", "request_id": request.id, "idempotency_key": REPREPARE_KEY, "expected_review_sha256": expected_sha, "target_preparation_policy": policy, "document_ids": sorted(request.documents), "source_assets_unchanged": True, "publication_or_index_requested": False}, request.id


def validate_partial_checkpoint(session: Session, batch: RegulatoryDocumentIntake, request: RegulatoryDocumentIntakeReprepare) -> dict[str, Any]:
    from onyx.db.document_intake_normalization import (
        _frozen_source,
        validated_normalized_documents,
    )
    from onyx.db.document_intake_reprepare import reprepare_snapshot
    from onyx.db.models import RegulatoryDocumentIntakeFile, UserFile
    from onyx.db.regulatory_canonical_revisions import get_canonical_revisions
    from onyx.db.regulatory_writer_publication import canonical_scope_digest
    from onyx.document_index.publication_models import publication_digest
    from onyx.file_processing.original_ingestion import OriginalIngestionReceipt
    from onyx.regulatory.document_intake.models import IntakePreparationPolicy
    from onyx.regulatory.document_intake.preparation import (
        load_verified_intake_documents,
    )

    preview = reprepare_snapshot(session, batch)
    policy = IntakePreparationPolicy().model_dump(mode="json")
    current = {item["document_id"]: item for item in preview["documents"]}
    require(set(current) == {MAIN_DOCUMENT, PDF_DOCUMENT} and preview.get("target_preparation_policy") == policy, "The current guarded source scope or policy changed")
    for document_id in (MAIN_DOCUMENT, PDF_DOCUMENT):
        original_entry, live_entry = request.documents[document_id], current[document_id]
        require(all(original_entry.get(key) == live_entry.get(key) for key in ("user_file_id", "original_file_id", "source_asset_id", "source_sha256", "parent_document_id", "role", "relationships", "source_receipt")), "A retained original source or relationship changed")
    require(request.documents[MAIN_DOCUMENT].get("status") == "prepared" and request.documents[PDF_DOCUMENT].get("status") == "pending", "Only the committed main and pending PDF checkpoint may resume")
    require(current[PDF_DOCUMENT]["canonical"] == request.documents[PDF_DOCUMENT]["canonical"] and pdf_chunk_verification(session)["chunking_version"] == "3", "The cancelled PDF before-snapshot changed")
    main = current[MAIN_DOCUMENT]["canonical"]
    require(len(main) == 365 and all(row["metadata"].get("intake_chunking_version") == "4" and row["metadata"].get("intake_preparation_policy") == policy for row in main), "The committed main v4 profile changed")
    canonical = [row for row in main if row["chunk_type"] != "hierarchical_aggregate" and row["metadata"].get("chunk_variant") != "hierarchical_aggregate"]
    identity = [{key: row[key] for key in ("id", "chunk_type", "heading_path", "text")} for row in canonical]
    metadata_input = [{"chunk_id": row["id"], "text": row["text"], "heading_path": row["heading_path"], "chunk_type": row["chunk_type"]} for row in canonical]
    def digest(value: object) -> str:
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(len(canonical) == 342 and digest(sorted(identity, key=lambda value: value["id"])) == MAIN_CANONICAL_SHA and digest(metadata_input) == MAIN_METADATA_SHA, "The 342 main canonical identities or actual metadata input changed")
    entry = request.documents[MAIN_DOCUMENT]
    original = OriginalIngestionReceipt.model_validate(entry["source_receipt"])
    receipt = OriginalIngestionReceipt.model_validate(entry["after_receipt"])
    member = session.get_one(RegulatoryDocumentIntakeFile, UUID(MAIN_DOCUMENT))
    file = session.get_one(UserFile, UUID(MAIN_FILE))
    require(receipt.file_id == original.file_id and receipt.raw_sha256 == original.raw_sha256 and receipt.generation_hash == file.regulatory_chunk_generation_hash and receipt.canonical_sha256 == canonical_scope_digest(session, member.user_file_id), "The prepared main receipt changed")
    revision_ids = [UUID(value) for value in entry["after_revision_ids"]]
    revisions = get_canonical_revisions(session, revision_ids)
    snapshots = sorted((revision.snapshot for revision in revisions.values()), key=lambda row: (row.position, row.id))
    require(len(revision_ids) == len(revisions) == 365 and [row.model_dump(mode="json") for row in snapshots] == main and publication_digest(main) == receipt.canonical_sha256, "The main after-revisions differ from its committed canonical receipt")
    if entry.get("normalization_derivation") is not None:
        documents = validated_normalized_documents(session, batch, member, original, receipt, entry["normalization_derivation"], entry["after_revision_ids"])
    else:
        _context, arguments = _frozen_source(session, batch, member, original)
        arguments.update(chunking_version="4", formatter_version=4, preparation_policy=IntakePreparationPolicy.model_validate(policy))
        documents = load_verified_intake_documents(**arguments)
        require(documents.original_extraction.plaintext_sha256 == original.plaintext_sha256 == receipt.plaintext_sha256 and documents.original_extraction.documents_sha256 == receipt.documents_sha256, "The committed main plaintext/source context changed")
    require(documents.original_extraction.raw_sha256 == original.raw_sha256, "The immutable main raw source changed")
    return {"prepared_document_id": MAIN_DOCUMENT, "pending_document_id": PDF_DOCUMENT, "main_canonical_count": 342, "main_after_revision_count": len(revisions), "actual_main_metadata_input_sha256": MAIN_METADATA_SHA, "prepared_main_canonical_sha256": receipt.canonical_sha256, "current_review_sha256": preview["expected_sha256"], "original_sources_and_committed_checkpoint_verified": True}


def retry_frozen_reprepare(session: Session, batch: RegulatoryDocumentIntake) -> tuple[dict[str, Any], UUID | None]:
    from onyx.db.document_intake_reprepare import latest_reprepare
    from onyx.regulatory.document_intake.models import IntakePreparationPolicy

    request = latest_reprepare(session, batch.id)
    require(request is not None, "The frozen reprepare request is missing")
    assert request is not None
    require(str(request.id) == REPREPARE_REQUEST and request.intake_id == batch.id and request.idempotency_key == REPREPARE_KEY and request.expected_sha256 == REPREPARE_REVIEW_SHA, "The frozen reprepare request identity changed")
    if batch.status == "preparing" and batch.stage == "reprepare":
        return {"operator_action": "existing_reprepare_retry_inspected_only", "request_id": request.id, "status": request.status}, None
    require(batch.status == "failed" and batch.stage == "reprepare" and request.status == "failed", "Only the failed original reprepare may be retried")
    error = batch.error or ""
    require(error.startswith("(psycopg2.errors.QueryCanceled) canceling statement due to user request") and "DELETE FROM regulatory_chunk" in error and PDF_FILE in error and request.error == batch.error, "The guarded PDF cancellation signature changed")
    require(batch.review_sha256 is None and batch.labeling_run_id is None and batch.lease_token is None and batch.lease_expires_at is None, "The cancelled reprepare has a later review or live owner")
    policy = IntakePreparationPolicy().model_dump(mode="json")
    require(set(request.documents) == {MAIN_DOCUMENT, PDF_DOCUMENT} and all(entry.get("target_chunking_version") == "4" and entry.get("target_preparation_policy") == policy for entry in request.documents.values()), "The cancelled request scope or target policy changed")
    checkpoint = validate_partial_checkpoint(session, batch, request)
    audit = json.dumps(request.documents, sort_keys=True, default=str)
    deliveries = repository.retry_intake(session, batch)
    require(deliveries == [] and batch.status == "preparing" and batch.stage == "reprepare" and batch.error is None and batch.lease_token is None and batch.lease_expires_at is None, "The normal retry did not preserve the requested reprepare stage")
    require(json.dumps(request.documents, sort_keys=True, default=str) == audit and request.expected_sha256 == REPREPARE_REVIEW_SHA, "The normal retry changed the frozen reprepare audit")
    verify_scope(session, batch)
    return {"operator_action": "normal_frozen_reprepare_retry_requested", "request_id": request.id, "expected_review_sha256": request.expected_sha256, "target_preparation_policy": policy, "verified_partial_checkpoint": checkpoint, "publication_or_index_requested": False}, request.id


def require_completed_labeling(run: RegulatoryLabelingRun) -> None:
    require(run.id == LABELING_RUN and run.document_set_id == 15 and set(run.file_ids) == {MAIN_FILE, PDF_FILE}, "The completed labeling scope changed")
    require(run.status == "completed" and run.stage == "finished" and run.total_chunks == run.completed_chunks == 662 and run.failed_chunks == run.stale_chunks == run.unresolved_derived_chunks == 0 and run.derived_chunks == 35 and not run.cancel_requested and run.error is None, "All canonical and derived labels must finish without failures before review")


def unpublished_epoch_scope(session: Session) -> list[dict[str, Any]]:
    publications = rows(session, """
        SELECT user_file_id,epoch,gate_closed,writer_manifest,original_ingestion_receipt
        FROM regulatory_file_publication WHERE user_file_id IN (:main,:pdf) ORDER BY user_file_id
    """, main=MAIN_FILE, pdf=PDF_FILE)
    require({str(item["user_file_id"]) for item in publications} == {MAIN_FILE, PDF_FILE} and all(item["epoch"] == 0 and not item["gate_closed"] and item["writer_manifest"] is None for item in publications), "The unpublished epoch-zero scope changed")
    require(not rows(session, "SELECT id FROM regulatory_indexing_job WHERE user_file_id IN (:main,:pdf) LIMIT 1", main=MAIN_FILE, pdf=PDF_FILE), "An indexing request already exists")
    return publications


def review_receipt_proof(session: Session, batch: RegulatoryDocumentIntake, run: RegulatoryLabelingRun, *, verify_completed_labels: bool = True) -> dict[str, Any]:
    from collections import Counter

    from sqlalchemy import select

    from onyx.db.document_intake_metadata import (
        metadata_chunks,
        validate_intake_metadata_receipts,
    )
    from onyx.db.document_intake_reprepare import latest_reprepare
    from onyx.db.models import (
        RegulatoryChunk,
        RegulatoryDerivedLabelProjection,
        RegulatoryLabelingItem,
    )
    from onyx.db.regulatory_labeling import (
        current_derived_label_sources_for_search,
        current_labeling_items_for_search,
    )
    from onyx.file_processing.original_ingestion import OriginalIngestionReceipt
    from onyx.regulatory.document_intake.metadata import (
        METADATA_AUDIT_KEY,
        METADATA_MODEL,
        METADATA_VERSION,
        metadata_input_hash,
    )
    from onyx.regulatory.document_intake.models import IntakePreparationPolicy

    assets = verify_scope(session, batch)
    publications = unpublished_epoch_scope(session)
    request = latest_reprepare(session, batch.id)
    policy = IntakePreparationPolicy().model_dump(mode="json")
    require(request is not None and str(request.id) == REPREPARE_REQUEST and request.expected_sha256 == REPREPARE_REVIEW_SHA and request.status == "prepared" and request.error is None, "The completed frozen reprepare changed")
    assert request is not None
    require(set(request.documents) == {MAIN_DOCUMENT, PDF_DOCUMENT} and all(item.get("status") == "prepared" and item.get("target_preparation_policy") == policy for item in request.documents.values()), "A source reprepare checkpoint is incomplete")
    require(rows(session, "SELECT version_num FROM alembic_version") == [{"version_num": "576f0377f747"}], "The required FK-index migration head changed")
    indexes = rows(session, """
        SELECT idx.relname AS name,t.relname AS table_name,a.attname AS column_name,
               i.indisvalid AND i.indisready AND NOT i.indisunique AND i.indnkeyatts=1
               AND i.indnatts=1 AND i.indpred IS NULL AND i.indexprs IS NULL
               AND am.amname='btree' AS valid_ready
        FROM pg_index i JOIN pg_class idx ON idx.oid=i.indexrelid
        JOIN pg_class t ON t.oid=i.indrelid JOIN pg_namespace n ON n.oid=t.relnamespace
        JOIN pg_am am ON am.oid=idx.relam
        JOIN pg_attribute a ON a.attrelid=t.oid AND a.attnum=i.indkey[0]
        WHERE n.nspname=current_schema() AND idx.relname=ANY(:names) ORDER BY idx.relname
    """, names=list(FK_INDEXES))
    require({item["name"]: (item["table_name"], item["column_name"]) for item in indexes} == FK_INDEXES and all(item["valid_ready"] for item in indexes), "The six source-reference indexes are not valid and ready")
    all_rows = list(session.scalars(select(RegulatoryChunk).where(RegulatoryChunk.user_file_id.in_([UUID(MAIN_FILE), UUID(PDF_FILE)])).order_by(RegulatoryChunk.user_file_id, RegulatoryChunk.position, RegulatoryChunk.id)))
    require(len(all_rows) == 697 and all(item.source == "indexed" and item.status == "active" and item.chunk_metadata.get("intake_chunking_version") == "4" and item.chunk_metadata.get("intake_preparation_policy") == policy for item in all_rows), "The current v4 source scope changed")
    canonical_ids: set[str] = set()
    document_proofs: list[dict[str, Any]] = []
    retained_text: dict[str, str] = {}
    publication_by_file = {str(item["user_file_id"]): item for item in publications}
    for asset in assets:
        file_id = str(asset["user_file_id"])
        receipt = OriginalIngestionReceipt.model_validate(publication_by_file[file_id]["original_ingestion_receipt"])
        require(receipt.raw_sha256 == asset["sha256"], "An original raw source receipt changed")
        with retained_source_stream(receipt.file_id, session) as stream:
            raw = stream.read(10_000_001)
        require(len(raw) <= 10_000_000 and hashlib.sha256(raw).hexdigest() == asset["sha256"], "An original raw source stream changed")
        with retained_source_stream(asset["text_file_id"], session) as stream:
            prepared = stream.read(2_000_001)
        require(len(prepared) <= 2_000_000 and hashlib.sha256(prepared).hexdigest() == asset["text_sha256"], "A retained prepared source changed")
        retained_text[file_id] = prepared.decode("utf-8")
        file_rows = [item for item in all_rows if str(item.user_file_id) == file_id]
        chunks = metadata_chunks(file_rows)
        require((len(file_rows), len(chunks)) == ((365, 342) if file_id == MAIN_FILE else (332, 320)), "Canonical metadata coverage changed")
        input_hash = metadata_input_hash(chunks)
        if file_id == MAIN_FILE:
            require(input_hash == MAIN_METADATA_SHA, "The actual main metadata input changed")
        chunk_ids = {item.chunk_id for item in chunks}
        canonical_ids.update(chunk_ids)
        for item in file_rows:
            if item.id not in chunk_ids:
                continue
            audit = item.chunk_metadata.get(METADATA_AUDIT_KEY, {})
            require(audit.get("input_sha256") == input_hash and audit.get("input_stale") is False and audit.get("version") == METADATA_VERSION and audit.get("model") == METADATA_MODEL and isinstance(audit.get("request_hash"), str) and bool(re.fullmatch(r"[0-9a-f]{64}", audit["request_hash"])) and isinstance(audit.get("fields"), dict) and all(not field.get("stale") for field in audit["fields"].values()), "A canonical metadata receipt is incomplete or stale")
        document_proofs.append({"user_file_id": file_id, "canonical_chunks": len(chunks), "all_chunks": len(file_rows), "metadata_input_sha256": input_hash, "raw_source_sha256": asset["sha256"], "prepared_source_sha256": asset["text_sha256"], "headingless_canonical_chunks": sum(not item.heading_path for item in chunks), "heading_samples": [item.heading_path[:4] for item in chunks[:3]]})
    validate_intake_metadata_receipts(session, batch)
    require(len(canonical_ids) == 662, "The complete metadata target count changed")
    pdf_rows = [item for item in all_rows if str(item.user_file_id) == PDF_FILE and item.id in canonical_ids]
    source_lines = [line.strip() for line in retained_text[PDF_FILE].splitlines() if line.strip().startswith("|")]
    actual_lines = [line.strip() for item in pdf_rows for line in item.text.splitlines() if line.strip().startswith("|")]
    require(len(source_lines) == 465 and not (Counter(source_lines) - Counter(actual_lines)) and sum(item.chunk_type == "table" for item in pdf_rows) == 68, "A source table line or whole-table block changed")
    if not verify_completed_labels:
        return {"canonical_metadata_receipts": 662, "source_table_lines_preserved": 465, "whole_table_blocks": 68, "fk_indexes_valid_ready": 6, "migration_head": "576f0377f747", "documents": document_proofs}
    items = list(session.scalars(select(RegulatoryLabelingItem).where(RegulatoryLabelingItem.run_id == run.id)))
    require(len(items) == 662 and {item.regulatory_chunk_id for item in items} == canonical_ids and all(item.status == "completed" and item.error is None for item in items), "Completed labeling items do not cover the canonical source")
    require(current_labeling_items_for_search(session, run=run, items=items) == canonical_ids, "A completed labeling source or context receipt is stale")
    derived = list(session.scalars(select(RegulatoryDerivedLabelProjection).where(RegulatoryDerivedLabelProjection.run_id == run.id)))
    require(len(derived) == 35 and all(item.user_file_id in {UUID(MAIN_FILE), UUID(PDF_FILE)} and item.resolution == "lineage" and item.unresolved_reason is None and bool(current_derived_label_sources_for_search(session, run=run, target=item)) for item in derived), "A derived label lineage receipt is unresolved or stale")
    return {"canonical_metadata_receipts": 662, "current_canonical_label_receipts": 662, "resolved_derived_label_receipts": 35, "source_table_lines_preserved": 465, "whole_table_blocks": 68, "fk_indexes_valid_ready": 6, "migration_head": "576f0377f747", "documents": document_proofs}


def refresh_completed_review(session: Session, batch: RegulatoryDocumentIntake) -> dict[str, Any]:
    require(batch.labeling_run_id == LABELING_RUN and batch.error is None and (batch.status, batch.stage) in {("preparing", "labeling"), ("ready", "review")}, "Only this completed intake labeling stage may refresh its review")
    run = session.get_one(RegulatoryLabelingRun, LABELING_RUN, with_for_update=True)
    require_completed_labeling(run)
    proof = review_receipt_proof(session, batch, run)
    before_digest = repository.review_digest(session, batch)
    before_epochs = unpublished_epoch_scope(session)
    result = repository.intake_snapshot(session, batch)
    require(batch.status == "ready" and batch.stage == "review" and batch.error is None and batch.review_sha256 == before_digest == repository.review_digest(session, batch) and bool(re.fullmatch(r"[0-9a-f]{64}", batch.review_sha256 or "")), "The normal GET refresh did not produce the unchanged ready review")
    require(result["status"] == "ready" and result["stage"] == "review" and result["labeling_failed_count"] == 0 and result["active_document_count"] == 2 and {item["user_file_id"] for item in result["documents"]} == {MAIN_FILE, PDF_FILE}, "The ready review snapshot scope is incomplete")
    by_id = {item["id"]: item for item in result["documents"]}
    require(by_id[MAIN_DOCUMENT]["role"] == "root" and by_id[PDF_DOCUMENT]["role"] == "annex" and by_id[PDF_DOCUMENT]["parent_document_id"] == MAIN_DOCUMENT and any(link["target_document_id"] == PDF_DOCUMENT for link in by_id[MAIN_DOCUMENT]["document_links"]), "The ready attachment navigation changed")
    verify_scope(session, batch)
    require(unpublished_epoch_scope(session) == before_epochs, "Review refresh changed publication/source authority")
    diagnostics = [{key: item[key] for key in ("id", "user_file_id", "display_name", "source_display_name", "source_url", "role", "parent_document_id", "status", "chunk_count", "labeling_failed_count", "document_links")} for item in result["documents"]]
    return {"operator_action": "normal_get_review_refreshed", "status": batch.status, "stage": batch.stage, "review_sha256": batch.review_sha256, "edit_review_sha256": result["edit_review_sha256"], "publication_review_sha256": result["publication_review_sha256"], "labeling_run_id": str(run.id), "source_and_receipt_proof": proof, "documents": diagnostics, "approval_or_publication_or_index_requested": False}


def strict_v4_canary(input_path: Path, expected_sha: str) -> dict[str, Any]:
    """Exercise the real normal freezer and strict encoder using detached fixture rows."""
    import logging
    import math

    from sqlalchemy import select

    from onyx.db.models import RegulatoryChunk, RegulatorySourceAsset, UserFile
    from onyx.indexing.embedder import DefaultIndexingEmbedder
    from onyx.regulatory.amendments.annexes.analysis import resolve_review_context_llm
    from onyx.regulatory.document_intake.models import IntakePreparationPolicy
    from onyx.regulatory.projection import prepare_normal_context_view
    from onyx.regulatory.writer_projection import _intake_view_vectors
    from shared_configs.enums import EmbeddingProvider

    require(bool(re.fullmatch(r"[0-9a-f]{64}", expected_sha)), "Canary requires the exact fixture SHA")
    wire = input_path.read_bytes()
    require(0 < len(wire) <= 131072 and hashlib.sha256(wire).hexdigest() == expected_sha, "Fixture size or SHA changed")
    prefix = "title: none | text: "
    require(wire.decode().startswith(prefix), "Fixture is missing the established Google instruction wrapper")
    source = wire.decode()[len(prefix):]
    require(source.strip() == source and source.startswith("|"), "Fixture is not one exact source table")
    with get_session_with_current_tenant() as session:
        session.execute(text("SET TRANSACTION READ ONLY"))
        asset = session.scalar(select(RegulatorySourceAsset).where(RegulatorySourceAsset.id == UUID("7743688a-0cdf-4b8a-8315-b292b21724ba")))
        require(asset is not None and asset.text_sha256 == EXPECTED_ASSETS[str(asset.id)][1], "Retained source identity changed")
        assert asset is not None and asset.text_file_id is not None
        stream = retained_source_stream(asset.text_file_id, session)
        with stream:
            prepared = stream.read(2_000_001)
        require(len(prepared) <= 2_000_000 and hashlib.sha256(prepared).hexdigest() == asset.text_sha256 and source in prepared.decode("utf-8"), "Fixture is not an exact span of the retained source")
        settings = get_current_search_settings(session)
        require(settings.id == 11 and settings.provider_type == EmbeddingProvider.GOOGLE and settings.model_name == "gemini-embedding-2" and settings.enable_contextual_rag, "The configured strict canary model or contextual policy changed")
        embedder = DefaultIndexingEmbedder.from_db_search_settings(settings)
        dimension, model_dim = settings.final_embedding_dim, settings.model_dim
        session.expunge(settings)
        session.rollback()
    llm = resolve_review_context_llm(settings, None)
    file = UserFile(id=uuid5(BATCH, "detached-whole-table-canary"), name="Resmî Gazete whole-table canary")
    row = RegulatoryChunk(id="detached-whole-table-canary", user_file_id=file.id, text=source, position=0, projection_ordinal=0, heading_path=["EK"], chunk_type="table", chunk_metadata={"chunk_variant": "atomic", "intake_chunking_version": "4", "intake_preparation_policy": IntakePreparationPolicy().model_dump(mode="json")}, source="indexed", status="active")
    view = prepare_normal_context_view(rows=[row], user_file=file, search_settings=settings, embedder=embedder, llm=llm)
    require(len(view.projections) == 1 and not view.calls, "A self-contained one-table canary must not call a contextual model")
    projection = view.projections[0]
    require(projection.embedding_texts[0] == source and projection.embedding_config.get("strict_full_input") is True and projection.embedding_config.get("max_sequence_length") == 2048, "The normal freezer changed the complete table or strict budget")
    tokenizer = embedder.embedding_model.tokenizer
    require(512 < len(tokenizer.encode(source)) <= 1848, "The fixture does not exercise the larger-table path")
    native_counts: list[float] = []
    class NativeCounts(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.msg == "Whole-table embedding native_tokens=%s auto_truncate=false" and isinstance(record.args, tuple):
                count = record.args[0]
                if isinstance(count, (int, float)):
                    native_counts.append(float(count))
    from onyx.natural_language_processing.search_nlp_models import (
        logger as encoder_logger,
    )
    handler = NativeCounts()
    native_logger = encoder_logger.logger
    old_level = native_logger.level
    native_logger.addHandler(handler)
    native_logger.setLevel(logging.INFO)
    try:
        vectors = _intake_view_vectors(view.projections, [], embedder, model_dim=model_dim, vector_dimension=dimension, tenant_id="public", check_owner=lambda: None)
    finally:
        native_logger.removeHandler(handler)
        native_logger.setLevel(old_level)
    result = next(iter(vectors.values()))
    require(len(result) == len(projection.embedding_texts) and len(native_counts) == len(result), "Strict provider acknowledgements lack their native token evidence")
    require(all(len(vector) == dimension and all(math.isfinite(value) for value in vector) for vector in result), "Strict provider returned invalid vectors")
    return {"success": True, "fixture_wire_sha256": expected_sha, "source_table_sha256": hashlib.sha256(source.encode()).hexdigest(), "exact_source_span_verified": True, "normal_frozen_inputs_sha256": projection.embedding_input_sha256, "frozen_encoder_configuration": projection.embedding_config, "input_approximate_tokens": [len(tokenizer.encode(value)) for value in projection.embedding_texts], "native_processed_tokens_unordered": sorted(native_counts), "vector_dimensions": [len(vector) for vector in result], "complete_table_preserved": True, "requested_auto_truncate": False, "raw_truncated_field": "not inferred from SDK; native no-truncation request contract enforced", "contextual_provider_calls": 0, "document_or_index_writes": False}


def worker_failures() -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    for path in sorted(Path("/var/log/onyx").glob("*regulatory*.log")):
        # Bound reads to recent log bytes; this does not alter the worker logs.
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 2_000_000))
            tail = stream.read().decode(errors="replace").splitlines()[-3000:]
        selected: set[int] = set()
        for index, line in enumerate(tail):
            if str(BATCH) in line or ORIGINAL_ERROR in line:
                selected.update(range(max(0, index - 2), min(len(tail), index + 60)))
        for index in sorted(selected):
            line = tail[index]
            if SECRET_LINE.search(line):
                continue
            if str(BATCH) in line or "embedding content token limit" in line or re.match(r"\s*File \"|^[\w.]+(?:Error|Exception):", line):
                failures.append({"log": path.name, "line": line[:600]})
    return failures[-50:]


def exact_worker_activity() -> dict[str, Any]:
    """Query active/reserved tasks without consuming or resubmitting deliveries."""
    from onyx.background.celery.versioned_apps.client import app

    inspector = app.control.inspect(timeout=3.0)
    matches: list[dict[str, Any]] = []
    responders: set[str] = set()
    for state in ("active", "reserved"):
        response = getattr(inspector, state)() or {}
        for worker, tasks in response.items():
            if not worker.startswith("regulatory_annex@"):
                continue
            responders.add(worker)
            for task in tasks:
                kwargs = task.get("kwargs", {})
                if isinstance(kwargs, str):
                    continue
                if kwargs.get("batch_id") != str(BATCH):
                    continue
                matches.append({"worker": worker, "state": state, "task_id": task.get("id"), "task_name": task.get("name"), "started_at_unix": task.get("time_start"), "acknowledged": task.get("acknowledged"), "worker_pid": task.get("worker_pid"), "kwargs": {key: kwargs.get(key) for key in ("batch_id", "tenant_id", "environment", "database_identity", "reprepare_request_id")}})
    markers = [str(BATCH), REPREPARE_REQUEST, *[str(task["task_id"]) for task in matches if task["task_id"]]]
    path = Path("/var/log/onyx/celery_worker_regulatory_annex.log")
    log_matches: list[str] = []
    if path.exists():
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 2_000_000))
            lines = stream.read().decode(errors="replace").splitlines()
        for line in lines:
            if any(marker in line for marker in markers) and not SECRET_LINE.search(line):
                log_matches.append(line[:1200])
    return {"regulatory_worker_responders": sorted(responders), "exact_intake_tasks": matches, "request_id": REPREPARE_REQUEST, "bounded_exact_log_matches": log_matches[-80:], "operation": "read_only_inspect_active_reserved_and_log_tail"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-image-sha", required=True)
    parser.add_argument("--observed-image-sha", required=True)
    parser.add_argument("--retry", action="store_true")
    parser.add_argument("--embedding-canary", action="store_true")
    parser.add_argument("--review-canary", action="store_true")
    parser.add_argument("--reprepare-v4", action="store_true")
    parser.add_argument("--retry-reprepare-v4", action="store_true")
    parser.add_argument("--refresh-review", action="store_true")
    parser.add_argument("--strict-v4-canary", action="store_true")
    parser.add_argument("--input-path", type=Path)
    parser.add_argument("--expected-input-sha")
    args = parser.parse_args()
    require(sum((args.retry, args.embedding_canary, args.review_canary, args.reprepare_v4, args.retry_reprepare_v4, args.refresh_review, args.strict_v4_canary)) <= 1, "Mutations and canaries must be separate operations")
    require(not args.strict_v4_canary or (args.input_path is not None and args.expected_input_sha is not None), "Strict canary requires its exact fixture path and SHA")
    require(bool(re.fullmatch(r"[0-9a-f]{40}", args.expected_image_sha)), "Expected exact deployed commit SHA")
    require(args.observed_image_sha == args.expected_image_sha, "Selected DEV container image differs from expected SHA")
    require(not args.refresh_review or args.expected_image_sha == REFRESH_IMAGE_SHA, "Review refresh requires its exact approved deployed revision")
    require(POSTGRES_DB == "customs-regulations-dev", "Expected explicit DEV database configuration")
    require(config.REGULATORY_ANNEX_ENVIRONMENT == "dev", "Expected explicit DEV intake environment")
    tenant = get_current_tenant_id()
    require(tenant == "public", "Expected the original public tenant")
    set_is_ee_based_on_env_variable()
    SqlEngine.init_engine(pool_size=2, max_overflow=0)
    enqueue = False
    reprepare_id: UUID | None = None
    with get_session_with_current_tenant() as session:
        if not (args.retry or args.reprepare_v4 or args.retry_reprepare_v4 or args.refresh_review):
            session.execute(text("SET TRANSACTION READ ONLY"))
        require(session.execute(text("SELECT current_database()")).scalar_one() == "customs-regulations-dev", "Connected database is not DEV")
        batch = session.get(RegulatoryDocumentIntake, BATCH, with_for_update=args.retry or args.reprepare_v4 or args.retry_reprepare_v4 or args.refresh_review, populate_existing=True)
        require(batch is not None, "Exact frozen intake is missing")
        assert batch is not None
        report = snapshot(session, batch)
        if args.retry:
            if batch.status == "preparing":
                report["operator_action"] = "already_preparing_inspected_only"
                session.rollback()
            else:
                require(batch.status == "failed" and batch.stage == "chunking", "Retry is allowed only for the original failed chunking state")
                require(batch.error == ORIGINAL_ERROR, "Original failure signature changed")
                require(batch.review_sha256 is None and batch.labeling_run_id is None, "A later intake stage already exists")
                require(not rows(session, "SELECT id FROM regulatory_chunk WHERE user_file_id=:id LIMIT 1", id=PDF_FILE), "PDF chunks already exist; inspect instead of retry")
                deliveries = repository.retry_intake(session, batch)
                require(deliveries == [] and batch.status == "preparing" and batch.stage == "chunking", "Standard retry produced an unexpected delivery stage")
                verify_scope(session, batch)
                report["retry_result"] = {"status": batch.status, "stage": batch.stage, "error": batch.error}
                session.commit()
                enqueue = True
                report["operator_action"] = "standard_retry_committed"
        elif args.reprepare_v4:
            action, reprepare_id = request_v4_reprepare(session, batch)
            report.update(action)
            if reprepare_id is None:
                session.rollback()
            else:
                session.commit()
        elif args.retry_reprepare_v4:
            action, reprepare_id = retry_frozen_reprepare(session, batch)
            report.update(action)
            if reprepare_id is None:
                session.rollback()
            else:
                session.commit()
        elif args.refresh_review:
            report.update(refresh_completed_review(session, batch))
            session.commit()
        else:
            report["operator_action"] = "inspected_read_only"
            session.rollback()
    if enqueue:
        from onyx.background.celery.tasks.regulatory_amendments.document_intake import (
            enqueue_intake_batch,
        )
        enqueue_intake_batch(batch_id=BATCH, tenant_id=tenant)
        report["delivery"] = "standard_enqueue_intake_batch_sent"
    if reprepare_id is not None:
        from onyx.background.celery.tasks.regulatory_amendments.document_intake import (
            enqueue_intake_batch,
        )
        enqueue_intake_batch(batch_id=BATCH, tenant_id=tenant, reprepare_request_id=reprepare_id)
        report["delivery"] = "standard_enqueue_intake_batch_with_frozen_reprepare_sent"
    if args.embedding_canary:
        report["embedding_canary"] = embedding_canary()
    if args.review_canary:
        report["review_budget_canary"] = review_budget_canary()
    if args.strict_v4_canary:
        report["strict_v4_canary"] = strict_v4_canary(args.input_path, args.expected_input_sha)
    report["expected_image_sha"] = args.expected_image_sha
    report["worker_failures"] = worker_failures()
    if not (args.retry or args.reprepare_v4 or args.retry_reprepare_v4 or args.refresh_review or args.embedding_canary or args.review_canary or args.strict_v4_canary):
        report["exact_worker_activity"] = exact_worker_activity()
    # Provider error strings can occasionally contain credentials; redact the entire value.
    def sanitized(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: sanitized(item) for key, item in value.items()}
        if isinstance(value, list):
            return [sanitized(item) for item in value]
        return "[redacted sensitive diagnostic]" if isinstance(value, str) and SECRET_LINE.search(value) else value
    print(json.dumps(sanitized(report), ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
