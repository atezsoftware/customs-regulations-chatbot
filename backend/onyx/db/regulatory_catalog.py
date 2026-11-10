"""Bounded catalog reads across every owner and document set in this tenant."""

import re
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import Select, exists, func, select
from sqlalchemy.orm import Session, lazyload, selectinload
from sqlalchemy.sql.elements import ColumnElement

from onyx.db.enums import RegulatoryChunkStatus, UserFileStatus
from onyx.db.models import (
    DocumentSet,
    DocumentSet__UserFile,
    FileRecord,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryFilePublication,
    UserFile,
    UserFileProjectionRepair,
)


def catalog_display_name_expression() -> ColumnElement[str]:
    """Search the same original leaf title that is displayed, never URLs or body text."""
    metadata = FileRecord.file_metadata
    original = (
        select(
            func.coalesce(
                func.nullif(
                    func.btrim(
                        metadata["legistation_metadata"]["official_title"].astext
                    ),
                    "",
                ),
                func.nullif(func.btrim(metadata["original_filename"].astext), ""),
                func.nullif(func.btrim(metadata["original_name"].astext), ""),
                func.nullif(
                    func.btrim(
                        func.replace(
                            func.regexp_replace(
                                func.regexp_replace(
                                    func.replace(FileRecord.display_name, "\\", "/"),
                                    r"^.*/",
                                    "",
                                ),
                                r"\.(md|txt)$",
                                "",
                                "i",
                            ),
                            "_",
                            " ",
                        )
                    ),
                    "",
                ),
            )
        )
        .where(FileRecord.file_id == UserFile.file_id)
        .correlate(UserFile)
        .scalar_subquery()
    )
    leaf = func.regexp_replace(func.replace(UserFile.name, "\\", "/"), r"^.*/", "")
    fallback = func.replace(
        func.regexp_replace(leaf, r"\.(md|txt)$", "", "i"), "_", " "
    )
    return func.coalesce(original, fallback)


def catalog_display_path(storage_name: str, display_name: str) -> str:
    folder = catalog_folder_path(storage_name)
    return f"{folder}/{display_name}" if folder else display_name


def catalog_display_names(
    db_session: Session, files: Sequence[UserFile]
) -> dict[UUID, str]:
    if not files:
        return {}
    rows = db_session.execute(
        select(UserFile.id, catalog_display_name_expression()).where(
            UserFile.id.in_([file.id for file in files])
        )
    )
    return {file_id: display_name for file_id, display_name in rows}


def _name_matches(q: str) -> ColumnElement[bool]:
    return catalog_display_name_expression().icontains(q, autoescape=True)


def normalize_catalog_path(value: str) -> str:
    """Archive imports preserve relative paths in UserFile.name."""
    return re.sub(r"/+", "/", value.replace("\\", "/")).strip("/")


def catalog_folder_path(name: str) -> str:
    normalized = normalize_catalog_path(name)
    return normalized.rsplit("/", 1)[0] if "/" in normalized else ""


def _normalized_name() -> ColumnElement[str]:
    return func.btrim(
        func.regexp_replace(func.replace(UserFile.name, "\\", "/"), r"/+", "/", "g"),
        "/",
    )


def catalog_file_query(
    *, q: str, folder: str | None, recursive: bool, document_set_id: int | None = None
) -> Select[tuple[UserFile]]:
    query = select(UserFile)
    if document_set_id is not None:
        # The Tariff proxy authenticates via a shared service key. A public set
        # grants the same read ACL to each authenticated caller.
        query = query.where(
            UserFile.status == UserFileStatus.COMPLETED,
            exists().where(
                DocumentSet__UserFile.user_file_id == UserFile.id,
                DocumentSet__UserFile.document_set_id == document_set_id,
                DocumentSet.id == DocumentSet__UserFile.document_set_id,
                DocumentSet.is_public.is_(True),
                DocumentSet.is_deleting.is_(False),
            ),
            exists().where(
                RegulatoryChunk.user_file_id == UserFile.id,
                RegulatoryChunk.status == RegulatoryChunkStatus.ACTIVE.value,
            ),
            ~exists().where(
                RegulatoryFilePublication.user_file_id == UserFile.id,
                RegulatoryFilePublication.gate_closed.is_(True),
            ),
        )
        pending = (
            select(RegulatoryDocumentIntakeFile.user_file_id)
            .join(
                RegulatoryDocumentIntake,
                RegulatoryDocumentIntake.id == RegulatoryDocumentIntakeFile.intake_id,
            )
            .where(
                RegulatoryDocumentIntake.status != "completed",
                RegulatoryDocumentIntakeFile.user_file_id.is_not(None),
            )
        )
        query = query.where(~UserFile.id.in_(pending))
    if q:
        query = query.where(_name_matches(q))
    if folder is not None:
        parent = normalize_catalog_path(folder)
        prefix = f"{parent}/" if parent else ""
        path = _normalized_name()
        if prefix:
            query = query.where(path.startswith(prefix, autoescape=True))
        if not parent or not recursive:
            query = query.where(
                func.strpos(func.substr(path, len(prefix) + 1), "/") == 0
            )
    return query


def list_catalog_files(
    db_session: Session,
    *,
    q: str,
    folder: str | None,
    recursive: bool,
    offset: int,
    limit: int,
    document_set_id: int | None = None,
) -> tuple[list[UserFile], int]:
    query = catalog_file_query(
        q=q, folder=folder, recursive=recursive, document_set_id=document_set_id
    )
    total = db_session.scalar(select(func.count()).select_from(query.subquery())) or 0
    files = list(
        db_session.scalars(
            query.options(selectinload(UserFile.document_sets))
            .order_by(func.lower(catalog_display_name_expression()), UserFile.id)
            .offset(offset)
            .limit(limit)
        ).all()
    )
    return files, total


def get_catalog_file(db_session: Session, file_id: UUID) -> UserFile | None:
    return db_session.scalar(
        select(UserFile)
        .where(UserFile.id == file_id)
        .options(selectinload(UserFile.document_sets))
    )


def catalog_chunk_counts(
    db_session: Session, file_ids: Sequence[UUID]
) -> dict[UUID, int]:
    if not file_ids:
        return {}
    rows = db_session.execute(
        select(RegulatoryChunk.user_file_id, func.count(RegulatoryChunk.id))
        .where(RegulatoryChunk.user_file_id.in_(file_ids))
        .group_by(RegulatoryChunk.user_file_id)
    ).all()
    return {file_id: count for file_id, count in rows}


def catalog_index_requests(
    db_session: Session, file_ids: Sequence[UUID]
) -> dict[UUID, UserFileProjectionRepair]:
    if not file_ids:
        return {}
    rows = db_session.scalars(
        select(UserFileProjectionRepair).where(
            UserFileProjectionRepair.user_file_id.in_(file_ids)
        )
    ).all()
    return {row.user_file_id: row for row in rows}


def list_catalog_folders(
    db_session: Session,
    *,
    parent: str,
    q: str,
    offset: int,
    limit: int,
    document_set_id: int | None = None,
) -> tuple[list[tuple[str, int]], int]:
    """Group immediate children in SQL across the complete filtered catalog."""
    parent = normalize_catalog_path(parent)
    prefix = f"{parent}/" if parent else ""
    path = _normalized_name()
    remainder = func.substr(path, len(prefix) + 1)
    child = func.split_part(remainder, "/", 1).label("child")
    query = (
        select(child, func.count(UserFile.id).label("file_count"))
        .where(func.strpos(remainder, "/") > 0)
        .group_by(child)
    )
    if document_set_id is not None:
        visible = catalog_file_query(
            q="", folder=None, recursive=True, document_set_id=document_set_id
        ).with_only_columns(UserFile.id)
        query = query.where(UserFile.id.in_(visible))
    if prefix:
        query = query.where(path.startswith(prefix, autoescape=True))
    if q:
        query = query.where(_name_matches(q))
    total = db_session.scalar(select(func.count()).select_from(query.subquery())) or 0
    rows = db_session.execute(
        query.order_by(func.lower(child), child).offset(offset).limit(limit)
    ).all()
    return [(name, count) for name, count in rows], total


def catalog_search_rows(
    db_session: Session,
    chunk_ids: Sequence[str],
    *,
    include_document_sets: bool = True,
) -> dict[str, tuple[RegulatoryChunk, UserFile]]:
    if not chunk_ids:
        return {}
    rows = db_session.execute(
        select(RegulatoryChunk, UserFile)
        .join(UserFile, UserFile.id == RegulatoryChunk.user_file_id)
        .where(RegulatoryChunk.id.in_(chunk_ids))
        .options(
            lazyload(UserFile.projects),
            selectinload(UserFile.document_sets)
            if include_document_sets
            else lazyload(UserFile.document_sets),
        )
        .execution_options(populate_existing=True)
    ).all()
    return {chunk.id: (chunk, file) for chunk, file in rows}


def catalog_chunk_page_offset(
    db_session: Session, file_id: UUID, chunk_id: str, *, limit: int
) -> int | None:
    """Locate a canonical chunk using the full file viewer's stable row order."""
    if not 1 <= limit <= 100:
        raise ValueError("Chunk page limit must be between 1 and 100")
    ordered = (
        select(
            RegulatoryChunk.id,
            (
                func.row_number().over(
                    order_by=(RegulatoryChunk.position, RegulatoryChunk.id)
                )
                - 1
            ).label("row_offset"),
        )
        .where(RegulatoryChunk.user_file_id == file_id)
        .subquery()
    )
    offset = db_session.scalar(
        select(ordered.c.row_offset).where(ordered.c.id == chunk_id)
    )
    return None if offset is None else (int(offset) // limit) * limit
