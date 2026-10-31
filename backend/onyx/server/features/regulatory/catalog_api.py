"""API-key and session accessible admin browsing without generative search."""

from collections.abc import Sequence
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from onyx.auth.permissions import require_permission
from onyx.configs.constants import PUBLIC_API_TAGS
from onyx.context.search.enums import QueryType
from onyx.context.search.models import IndexFilters
from onyx.context.search.utils import get_query_embedding
from onyx.db.document_intake import pending_intake_file_ids
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import Permission, UserFileStatus
from onyx.db.models import User, UserFile
from onyx.db.regulatory_catalog import (
    catalog_chunk_counts,
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
from onyx.db.search_settings import get_current_search_settings
from onyx.document_index.factory import get_default_document_index
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.natural_language_processing.search_nlp_models import EmbeddingModel
from onyx.regulatory.publication_reads import (
    filter_publication_read,
    observe_publication_read,
    require_publication_files,
)
from onyx.server.features.document_set.file_models import (
    RegulatoryIndexingProgressSnapshot,
)
from onyx.server.features.regulatory.catalog_models import (
    CATALOG_SEARCH_CANDIDATES,
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
    observation = observe_publication_read()
    settings = get_current_search_settings(db_session)
    index = get_default_document_index(settings, None, db_session)
    # This permission-gated catalog deliberately spans every owner's files.
    # Tenant and publication constraints still apply inside the index adapter.
    filters = IndexFilters(
        access_control_list=None,
        tenant_id=get_current_tenant_id(),
        regulatory_chunks_only=True,
    )
    if request.mode == "bm25":
        retrieved = index.keyword_retrieval(
            query=request.query,
            filters=filters,
            num_to_retrieve=CATALOG_SEARCH_CANDIDATES,
            include_hidden=False,
        )
    else:
        model = EmbeddingModel.from_db_model(
            search_settings=settings,
            server_host=MODEL_SERVER_HOST,
            server_port=MODEL_SERVER_PORT,
        )
        embedding = get_query_embedding(request.query, embedding_model=model)
        if request.mode == "semantic":
            retrieved = index.semantic_retrieval(
                query_embedding=embedding,
                filters=filters,
                num_to_retrieve=CATALOG_SEARCH_CANDIDATES,
            )
        else:
            retrieved = index.hybrid_retrieval(
                query=request.query,
                query_embedding=embedding,
                final_keywords=None,
                query_type=QueryType.SEMANTIC,
                filters=filters,
                num_to_retrieve=CATALOG_SEARCH_CANDIDATES,
            )

    rows = catalog_search_rows(
        db_session,
        [chunk.regulatory_chunk_id for chunk in retrieved if chunk.regulatory_chunk_id],
    )
    files = {file.id: file for _, file in rows.values()}
    snapshots = _file_snapshots(db_session, list(files.values()))
    hits: list[CatalogSearchHit] = []
    seen: set[str] = set()
    for result in retrieved:
        canonical_id = result.regulatory_chunk_id
        if canonical_id is None or canonical_id in seen or canonical_id not in rows:
            continue
        chunk, file = rows[canonical_id]
        if str(file.id) != result.document_id:
            continue
        seen.add(canonical_id)
        hits.append(
            CatalogSearchHit(
                file=snapshots[file.id],
                chunk=RegulatoryChunkSnapshot.from_model(chunk),
                score=result.score,
            )
        )
    hits = filter_publication_read(observation, hits, lambda hit: str(hit.file.id))
    page = hits[request.offset : request.offset + request.limit]
    require_publication_files(observation, tuple(hit.file.id for hit in page))
    return CatalogSearchPage(
        items=page,
        total=len(hits),
        offset=request.offset,
        limit=request.limit,
        has_more=request.offset + len(page) < len(hits),
    )
