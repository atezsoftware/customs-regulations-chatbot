"""Bounded catalog reads across every owner and document set in this tenant."""

import re
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import Select, func, or_, select
from sqlalchemy.orm import Session, selectinload
from sqlalchemy.sql.elements import ColumnElement

from onyx.db.models import (
    AmendmentSourcePackage,
    DocumentSet__UserFile,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryRetainedPdfEvidence,
    RegulatorySourceAsset,
    UserFile,
    UserFileProjectionRepair,
)


def _name_matches(q: str) -> ColumnElement[bool]:
    source = (
        select(RegulatoryDocumentIntakeFile.id)
        .join(
            RegulatoryDocumentIntake,
            RegulatoryDocumentIntake.id == RegulatoryDocumentIntakeFile.intake_id,
        )
        .join(
            RegulatorySourceAsset,
            RegulatorySourceAsset.id == RegulatoryDocumentIntakeFile.source_asset_id,
        )
        .join(
            AmendmentSourcePackage,
            AmendmentSourcePackage.id == RegulatoryDocumentIntake.source_package_id,
        )
        .join(
            DocumentSet__UserFile,
            (
                DocumentSet__UserFile.user_file_id
                == RegulatoryDocumentIntakeFile.user_file_id
            )
            & (
                DocumentSet__UserFile.document_set_id
                == RegulatoryDocumentIntake.document_set_id
            ),
        )
        .where(
            RegulatoryDocumentIntakeFile.user_file_id == UserFile.id,
            RegulatorySourceAsset.package_id
            == RegulatoryDocumentIntake.source_package_id,
            AmendmentSourcePackage.document_set_id
            == RegulatoryDocumentIntake.document_set_id,
            AmendmentSourcePackage.created_by == RegulatoryDocumentIntake.created_by,
            RegulatoryDocumentIntake.created_by == UserFile.user_id,
            or_(
                RegulatorySourceAsset.display_name.icontains(q, autoescape=True),
                RegulatorySourceAsset.original_url.icontains(q, autoescape=True),
                RegulatorySourceAsset.final_url.icontains(q, autoescape=True),
            ),
        )
        .correlate(UserFile)
        .exists()
    )
    retained = (
        select(RegulatoryRetainedPdfEvidence.id)
        .join(
            DocumentSet__UserFile,
            (
                DocumentSet__UserFile.user_file_id
                == RegulatoryRetainedPdfEvidence.user_file_id
            )
            & (
                DocumentSet__UserFile.document_set_id
                == RegulatoryRetainedPdfEvidence.document_set_id
            ),
        )
        .where(
            RegulatoryRetainedPdfEvidence.user_file_id == UserFile.id,
            RegulatoryRetainedPdfEvidence.original_filename.icontains(
                q, autoescape=True
            ),
        )
        .correlate(UserFile)
        .exists()
    )
    return or_(UserFile.name.icontains(q, autoescape=True), source, retained)


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


def _file_query(
    *, q: str, folder: str | None, recursive: bool
) -> Select[tuple[UserFile]]:
    query = select(UserFile)
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
) -> tuple[list[UserFile], int]:
    query = _file_query(q=q, folder=folder, recursive=recursive)
    total = db_session.scalar(select(func.count()).select_from(query.subquery())) or 0
    files = list(
        db_session.scalars(
            query.options(selectinload(UserFile.document_sets))
            .order_by(func.lower(UserFile.name), UserFile.id)
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
    db_session: Session, *, parent: str, q: str, offset: int, limit: int
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
    db_session: Session, chunk_ids: Sequence[str]
) -> dict[str, tuple[RegulatoryChunk, UserFile]]:
    if not chunk_ids:
        return {}
    rows = db_session.execute(
        select(RegulatoryChunk, UserFile)
        .join(UserFile, UserFile.id == RegulatoryChunk.user_file_id)
        .where(RegulatoryChunk.id.in_(chunk_ids))
        .options(selectinload(UserFile.document_sets))
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
