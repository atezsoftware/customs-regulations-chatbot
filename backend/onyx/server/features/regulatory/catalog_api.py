"""API-key and session accessible admin browsing without generative search."""

import math
from collections import Counter
from collections.abc import Sequence
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from onyx.auth.permissions import require_permission
from onyx.configs.constants import PUBLIC_API_TAGS
from onyx.context.search.enums import QueryType
from onyx.context.search.models import IndexFilters, InferenceChunk
from onyx.context.search.utils import get_query_embedding
from onyx.db.catalog_search_scope import catalog_heading_paths, catalog_source_titles
from onyx.db.document_intake import pending_intake_file_ids
from onyx.db.document_intake_existing_relations import load_reviewed_file_relations
from onyx.db.document_intake_source_lineage import load_intake_source_lineages
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import Permission, UserFileStatus
from onyx.db.models import User, UserFile
from onyx.db.regulatory_catalog import (
    catalog_chunk_counts,
    catalog_chunk_page_offset,
    catalog_folder_path,
    catalog_index_requests,
    catalog_search_rows,
    get_catalog_file,
    list_catalog_files,
    list_catalog_folders,
    normalize_catalog_path,
)
from onyx.db.regulatory_indexing_jobs import (
    fetch_latest_regulatory_indexing_progress_for_user_files,
)
from onyx.db.reranking import get_reranker_configuration
from onyx.db.search_settings import get_current_search_settings
from onyx.document_index.factory import get_default_document_index
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.natural_language_processing.search_nlp_models import EmbeddingModel
from onyx.regulatory.catalog_scope import (
    CatalogSearchScope,
    resolve_catalog_source_titles,
    select_catalog_heading_prefixes,
)
from onyx.regulatory.heading_path import extract_regulatory_instrument_source_hint
from onyx.regulatory.publication_reads import (
    filter_publication_read,
    observe_publication_read,
    require_publication_files,
)
from onyx.reranking.normalized_selection import select_normalized_rerank_candidates
from onyx.reranking.service import rerank_chunks
from onyx.reranking.staged import rerank_regulatory_candidates_in_batches
from onyx.server.features.document_set.file_models import (
    RegulatoryIndexingProgressSnapshot,
)
from onyx.server.features.regulatory.catalog_models import (
    CATALOG_SEARCH_CANDIDATES,
    CATALOG_SEARCH_PER_FILE,
    CATALOG_SEARCH_RESULTS,
    CatalogChunkLocation,
    CatalogDocumentSet,
    CatalogFile,
    CatalogFilePage,
    CatalogFolder,
    CatalogFolderPage,
    CatalogIndexRequest,
    CatalogSearchHit,
    CatalogSearchPage,
    CatalogSearchRequest,
)
from onyx.server.features.regulatory.models import RegulatoryChunkSnapshot
from onyx.server.utils_vector_db import require_vector_db
from shared_configs.configs import MODEL_SERVER_HOST, MODEL_SERVER_PORT
from shared_configs.contextvars import get_current_tenant_id

router = APIRouter(prefix="/regulatory/catalog", tags=PUBLIC_API_TAGS)


def _file_snapshots(
    db_session: Session, files: Sequence[UserFile]
) -> dict[UUID, CatalogFile]:
    file_ids = [file.id for file in files]
    counts = catalog_chunk_counts(db_session, file_ids)
    requests = catalog_index_requests(db_session, file_ids)
    progress = fetch_latest_regulatory_indexing_progress_for_user_files(
        db_session, user_file_ids=file_ids
    )
    pending = pending_intake_file_ids(db_session, file_ids, include_approving=True)
    lineages = load_intake_source_lineages(db_session, file_ids)
    reviewed = load_reviewed_file_relations(db_session, file_ids)
    return {
        file.id: CatalogFile(
            id=file.id,
            name=file.name,
            folder_path=catalog_folder_path(file.name),
            user_id=file.user_id,
            file_id=file.file_id,
            status=file.status.value,
            file_type=file.file_type,
            created_at=file.created_at,
            chunk_count=counts.get(file.id, 0),
            document_sets=[
                CatalogDocumentSet(id=item.id, name=item.name)
                for item in sorted(file.document_sets, key=lambda item: item.id)
            ],
            pending_review=file.id in pending,
            source_lineage=lineages.get(file.id),
            reviewed_relations=reviewed.get(file.id),
            regulatory_indexing_progress=(
                RegulatoryIndexingProgressSnapshot.from_progress(progress[file.id])
                if file.id in progress
                else None
            ),
            index_request=(
                CatalogIndexRequest(
                    attempt_id=requests[file.id].attempt_id,
                    status=requests[file.id].status.value,
                    updated_at=requests[file.id].updated_at,
                    failure_code=requests[file.id].failure_code,
                )
                if file.id in requests
                else None
            ),
        )
        for file in files
    }


@router.get("/files")
def get_files(
    q: str = Query(default="", max_length=500),
    folder: str | None = Query(default=None, max_length=2000),
    recursive: bool = True,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=25, ge=1, le=100),
    _: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> CatalogFilePage:
    files, total = list_catalog_files(
        db_session, q=q, folder=folder, recursive=recursive, offset=offset, limit=limit
    )
    snapshots = _file_snapshots(db_session, files)
    return CatalogFilePage(
        items=list(snapshots.values()), total=total, offset=offset, limit=limit
    )


@router.get("/files/{file_id}")
def get_file(
    file_id: UUID,
    _: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> CatalogFile:
    file = get_catalog_file(db_session, file_id)
    if file is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "File not found")
    return _file_snapshots(db_session, [file])[file.id]


@router.post("/files/{file_id}/index", dependencies=[Depends(require_vector_db)])
def index_file(
    file_id: UUID,
    _: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> CatalogFile:
    """Queue only this reviewed, chunked file through the durable index writer."""
    # Reuse the document-set action's claim, expiry and recovery semantics.
    from onyx.server.features.document_set.api import _request_user_file_indexing

    file = get_catalog_file(db_session, file_id)
    if file is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "File not found")
    if file.status != UserFileStatus.CHUNKED:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT, "Only chunked files can be indexed"
        )
    if pending_intake_file_ids(db_session, [file_id], include_approving=True):
        raise OnyxError(
            OnyxErrorCode.CONFLICT,
            "Review and approve this new document in Updates before indexing",
        )
    if not catalog_chunk_counts(db_session, [file_id]).get(file_id, 0):
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT, "This file has no regulatory chunks"
        )
    _request_user_file_indexing([file], db_session, get_current_tenant_id())
    return _file_snapshots(db_session, [file])[file.id]


@router.get("/files/{file_id}/chunks/{chunk_id}/location")
def locate_chunk(
    file_id: UUID,
    chunk_id: str,
    limit: int = Query(default=25, ge=1, le=100),
    _: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> CatalogChunkLocation:
    observation = observe_publication_read()
    offset = catalog_chunk_page_offset(db_session, file_id, chunk_id, limit=limit)
    if offset is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Chunk not found in this file")
    require_publication_files(observation, (file_id,))
    return CatalogChunkLocation(offset=offset)


@router.get("/folders")
def get_folders(
    parent: str = Query(default="", max_length=2000),
    q: str = Query(default="", max_length=500),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=250),
    _: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> CatalogFolderPage:
    children, total = list_catalog_folders(
        db_session, parent=parent, q=q, offset=offset, limit=limit
    )
    prefix = normalize_catalog_path(parent)
    return CatalogFolderPage(
        items=[
            CatalogFolder(
                name=name, path=f"{prefix}/{name}" if prefix else name, file_count=count
            )
            for name, count in children
        ],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.post("/search", dependencies=[Depends(require_vector_db)])
def search_chunks(
    request: CatalogSearchRequest,
    _: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> CatalogSearchPage:
    return search_catalog_chunks(request, db_session)


def search_catalog_chunks(
    request: CatalogSearchRequest,
    db_session: Session,
    *,
    adaptive: bool = False,
    scope_query: str | None = None,
) -> CatalogSearchPage:
    """Share the same bounded, reranked evidence with both admin search screens."""
    observation = observe_publication_read()
    settings = get_current_search_settings(db_session)
    reranker = get_reranker_configuration(db_session)
    if not reranker.enabled:
        raise OnyxError(
            OnyxErrorCode.SERVICE_UNAVAILABLE,
            "Arama için indeks ayarlarından yeniden sıralayıcıyı etkinleştirin.",
        )
    index = get_default_document_index(settings, None, db_session)
    # This permission-gated catalog deliberately spans every owner's files.
    # Tenant and publication constraints still apply inside the index adapter.
    filters = IndexFilters(
        access_control_list=None,
        tenant_id=get_current_tenant_id(),
        regulatory_chunks_only=True,
    )
    scope: CatalogSearchScope | None = None
    if adaptive:
        original_query = scope_query or request.query
        hint = extract_regulatory_instrument_source_hint(original_query)
        if hint:
            titles = resolve_catalog_source_titles(
                original_query, catalog_source_titles(db_session, hint)
            )
            file_ids = [title.file_id for title in titles]
            scope = CatalogSearchScope(source_hint=hint, file_ids=file_ids)
            if not file_ids:
                return CatalogSearchPage(
                    items=[],
                    total=0,
                    offset=0,
                    limit=0,
                    has_more=False,
                    result_limit=None,
                    per_file_limit=None,
                    normalized_threshold=0.8,
                    scope=scope,
                )
            require_publication_files(observation, tuple(file_ids))
            paths = catalog_heading_paths(db_session, file_ids)
            prefixes = select_catalog_heading_prefixes(
                original_query, hint, paths, source_title=titles[0].name
            )
            scope.heading_prefixes = [list(prefix) for prefix in prefixes]
            scope.heading_paths = [
                path
                for path in paths
                if not prefixes
                or any(tuple(path[: len(prefix)]) == prefix for prefix in prefixes)
            ]
            filters.attached_document_ids = [str(file_id) for file_id in file_ids]
            filters.regulatory_heading_scope = [
                prefix[-1] for prefix in prefixes
            ] or None
    model = EmbeddingModel.from_db_model(
        search_settings=settings,
        server_host=MODEL_SERVER_HOST,
        server_port=MODEL_SERVER_PORT,
    )
    embedding = get_query_embedding(request.query, embedding_model=model)
    retrieved = index.hybrid_retrieval(
        query=request.query,
        query_embedding=embedding,
        final_keywords=None,
        query_type=QueryType.SEMANTIC,
        filters=filters,
        num_to_retrieve=CATALOG_SEARCH_CANDIDATES,
    )[:CATALOG_SEARCH_CANDIDATES]

    rows = catalog_search_rows(
        db_session,
        [chunk.regulatory_chunk_id for chunk in retrieved if chunk.regulatory_chunk_id],
    )
    files = {file.id: file for _, file in rows.values()}
    snapshots = _file_snapshots(db_session, list(files.values()))
    candidates: list[InferenceChunk] = []
    hits: dict[str, CatalogSearchHit] = {}
    seen: set[str] = set()
    for result in retrieved:
        canonical_id = result.regulatory_chunk_id
        if canonical_id is None or canonical_id in seen or canonical_id not in rows:
            continue
        chunk, file = rows[canonical_id]
        if str(file.id) != result.document_id or (
            scope and file.id not in scope.file_ids
        ):
            continue
        seen.add(canonical_id)
        snapshot = RegulatoryChunkSnapshot.from_model(chunk)
        if (
            scope
            and scope.heading_prefixes
            and not any(
                snapshot.heading_path[: len(prefix)] == prefix
                for prefix in scope.heading_prefixes
            )
        ):
            continue
        hits[canonical_id] = CatalogSearchHit(
            file=snapshots[file.id],
            chunk=snapshot,
            score=None,
            embedding_score=result.embedding_score
            if result.embedding_score is not None
            and math.isfinite(result.embedding_score)
            else None,
        )
        # Score the canonical text that is actually delivered to the administrator.
        candidates.append(
            result.model_copy(
                update={
                    "chunk_id": len(candidates),
                    "content": snapshot.text,
                    "blurb": snapshot.text,
                    "heading_path": snapshot.heading_path,
                    "title": file.name,
                    "semantic_identifier": file.name,
                    "doc_summary": "",
                    "chunk_context": "",
                }
            )
        )
    candidates = filter_publication_read(
        observation, candidates, lambda candidate: candidate.document_id
    )
    ranked_hits: list[CatalogSearchHit] = []
    if candidates:
        reranked = rerank_regulatory_candidates_in_batches(
            query=request.query,
            chunks=candidates,
            config=reranker,
            rerank=rerank_chunks,
        )
        if not reranked.used_external or len(reranked.scores_by_chunk) != len(
            candidates
        ):
            raise OnyxError(
                OnyxErrorCode.SERVICE_UNAVAILABLE,
                "Yeniden sıralama tamamlanamadı. Lütfen tekrar deneyin.",
            )
        normalized = select_normalized_rerank_candidates(
            chunks=reranked.ordered_chunks,
            scores=reranked.scores_by_chunk,
            baseline_limit=CATALOG_SEARCH_RESULTS,
            threshold=0.8,
            anchor_zero=adaptive,
        )
        file_counts: Counter[str] = Counter()
        selected = normalized.ordered_chunks if adaptive else reranked.ordered_chunks
        for candidate in selected:
            if (
                not adaptive
                and file_counts[candidate.document_id] >= CATALOG_SEARCH_PER_FILE
            ):
                continue
            canonical_id = candidate.regulatory_chunk_id
            if canonical_id is None:
                continue
            hit = hits[canonical_id]
            ranked_hits.append(
                hit.model_copy(
                    update={
                        "score": reranked.scores_by_chunk[
                            (candidate.document_id, candidate.chunk_id)
                        ],
                        "reranker_score": reranked.scores_by_chunk[
                            (candidate.document_id, candidate.chunk_id)
                        ],
                        "normalized_score": normalized.normalized_scores_by_chunk[
                            (candidate.document_id, candidate.chunk_id)
                        ],
                    }
                )
            )
            file_counts[candidate.document_id] += 1
            if not adaptive and len(ranked_hits) == CATALOG_SEARCH_RESULTS:
                break
    page = (
        ranked_hits
        if adaptive
        else ranked_hits[request.offset : request.offset + request.limit]
    )
    require_publication_files(observation, tuple(hit.file.id for hit in page))
    return CatalogSearchPage(
        items=page,
        total=len(ranked_hits),
        offset=request.offset,
        limit=len(page) if adaptive else request.limit,
        has_more=False if adaptive else request.offset + len(page) < len(ranked_hits),
        result_limit=None if adaptive else CATALOG_SEARCH_RESULTS,
        per_file_limit=None if adaptive else CATALOG_SEARCH_PER_FILE,
        normalized_threshold=0.8 if adaptive else None,
        scope=scope,
    )
