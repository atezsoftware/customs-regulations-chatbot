"""Authenticated, read-only access to the same PC corpus as Mevzuat AI."""

import hashlib
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from onyx.auth.users import current_user
from onyx.configs.constants import PUBLIC_API_TAGS
from onyx.db.engine.sql_engine import get_session
from onyx.db.models import User
from onyx.db.regulatory_catalog import (
    catalog_display_names,
    catalog_folder_path,
    get_catalog_file,
    list_catalog_files,
    list_catalog_folders,
    normalize_catalog_path,
)
from onyx.db.regulatory_library import library_scope, scoped_library_file_ids
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.regulatory.publication_reads import (
    observe_publication_read,
    require_publication_files,
)
from onyx.regulatory.read_document import read_document_markdown
from onyx.regulatory.reading_navigation import remove_reading_navigation
from onyx.server.features.regulatory.catalog_models import (
    CatalogFolder,
    CatalogFolderPage,
)
from onyx.server.features.regulatory.library_models import (
    LibraryChunk,
    LibraryContent,
    LibraryFile,
    LibraryFilePage,
    LibrarySearchHit,
    LibrarySearchRequest,
    LibrarySearchResponse,
    LibrarySearchResults,
)
from onyx.server.features.regulatory.library_repeals import verified_repeals
from onyx.server.features.regulatory.library_rerank import rerank_library_candidates
from onyx.server.features.regulatory.library_search_memory import (
    answer_library_small_talk,
    contextual_library_query,
    recall_previous_search,
)
from onyx.server.features.regulatory.simple_search_api import run_simple_search
from onyx.server.features.regulatory.simple_search_models import SimpleSearchRequest
from onyx.server.utils_vector_db import require_vector_db

router = APIRouter(prefix="/regulatory/library", tags=PUBLIC_API_TAGS)


@router.get("/files")
def get_files(
    q: str = Query(default="", max_length=500),
    folder: str | None = Query(default=None, max_length=2000),
    recursive: bool = True,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> LibraryFilePage:
    scope = library_scope(db_session, user)
    observation = observe_publication_read()
    files, total = list_catalog_files(
        db_session,
        q=q.strip(),
        folder=folder,
        recursive=recursive,
        offset=offset,
        limit=limit,
        document_set_id=scope.asv3_document_set_id,
    )
    names = catalog_display_names(db_session, files)
    require_publication_files(observation, tuple(file.id for file in files))
    return LibraryFilePage(
        items=[
            LibraryFile(
                id=file.id,
                name=names[file.id],
                folder_path=catalog_folder_path(file.name),
                chunk_count=file.chunk_count or 0,
            )
            for file in files
        ],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get("/folders")
def get_folders(
    parent: str = Query(default="", max_length=2000),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=250),
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> CatalogFolderPage:
    scope = library_scope(db_session, user)
    children, total = list_catalog_folders(
        db_session,
        parent=parent,
        q="",
        offset=offset,
        limit=limit,
        document_set_id=scope.asv3_document_set_id,
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


@router.get("/files/{file_id}/content")
def get_content(
    file_id: UUID,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> LibraryContent:
    scope = library_scope(db_session, user)
    if scope.asv3_document_set_id is None or file_id not in scoped_library_file_ids(
        db_session, scope.asv3_document_set_id, [file_id]
    ):
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Document not found")
    observation = observe_publication_read()
    require_publication_files(observation, (file_id,))
    file = get_catalog_file(db_session, file_id)
    if file is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Document not found")
    name = catalog_display_names(db_session, [file])[file_id]
    markdown = read_document_markdown(db_session, file)
    require_publication_files(observation, (file_id,))
    source_sha256 = hashlib.sha256(markdown.encode()).hexdigest()
    repeals = verified_repeals(file_id, source_sha256)
    markdown = remove_reading_navigation(markdown)
    content_sha256 = hashlib.sha256(markdown.encode()).hexdigest()
    return LibraryContent(
        file_id=file_id,
        name=name,
        markdown=markdown,
        content_sha256=content_sha256,
        verified_repeals=repeals,
    )


@router.post("/simple-search")
def search(
    request: LibrarySearchRequest,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> LibrarySearchResponse:
    scope = library_scope(db_session, user)
    reply = answer_library_small_talk(request) or recall_previous_search(request)
    if reply is not None:
        return LibrarySearchResponse(
            kind="memory",
            search=LibrarySearchResults(items=[]),
            summary=reply,
            summary_error=None,
        )
    require_vector_db()
    result = run_simple_search(
        SimpleSearchRequest(query=contextual_library_query(request)),
        db_session,
        scope_filters=scope,
        overview_language=request.language,
        rerank_candidates=rerank_library_candidates,
    )
    return LibrarySearchResponse(
        search=LibrarySearchResults(
            items=[
                LibrarySearchHit(
                    file=LibraryFile(
                        id=hit.file.id,
                        name=hit.file.display_name or hit.file.name,
                        folder_path=hit.file.folder_path,
                        chunk_count=hit.file.chunk_count,
                    ),
                    chunk=LibraryChunk(
                        id=hit.chunk.id,
                        text=hit.chunk.text,
                        chunk_type=hit.chunk.chunk_type,
                        heading_path=hit.chunk.heading_path,
                        chunk_metadata={
                            key: value
                            for key in ("article_no", "article_title")
                            if isinstance(
                                value := hit.chunk.chunk_metadata.get(key), str
                            )
                        },
                    ),
                )
                for hit in result.search.items
            ],
        ),
        summary=result.summary,
        summary_error="unavailable" if result.summary_error else None,
    )
