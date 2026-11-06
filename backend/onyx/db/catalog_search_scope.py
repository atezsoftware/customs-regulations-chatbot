"""Read file-owned titles and full heading paths for a scoped catalog search."""

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from onyx.db.models import RegulatoryChunk, UserFile
from onyx.regulatory.catalog_scope import CatalogSourceTitle, catalog_fold


def catalog_source_titles(
    session: Session, source_hint: str
) -> list[CatalogSourceTitle]:
    # Filter by legal form first; spelling folding covers ASCII archive filenames.
    words = catalog_fold(source_hint).split()
    form = next(
        (
            stem
            for stem in (
                "kanun",
                "yonetmel",
                "teblig",
                "genelge",
                "karar",
                "sozles",
                "anlasma",
                "tuzuk",
                "yonerge",
                "protokol",
                "convention",
                "treaty",
                "regulation",
                "directive",
                "decision",
                "decree",
                "agreement",
                "protocol",
                "rules",
                "law",
                "act",
                "code",
            )
            if any(w.startswith(stem) for w in words)
        ),
        None,
    )
    if form is None:
        return []
    root = (
        select(RegulatoryChunk.heading_path[0].astext)
        .where(RegulatoryChunk.user_file_id == UserFile.id)
        .order_by(RegulatoryChunk.position, RegulatoryChunk.id)
        .limit(1)
        .correlate(UserFile)
        .scalar_subquery()
    )
    folded_name = func.translate(func.lower(UserFile.name), "ıİşğüöç", "iIsguoc")
    rows = session.execute(
        select(UserFile.id, UserFile.name, root).where(
            folded_name.contains(form, autoescape=True)
        )
    ).all()
    return [
        CatalogSourceTitle(file_id=file_id, name=name, root_heading=heading or "")
        for file_id, name, heading in rows
    ]


def catalog_heading_paths(
    session: Session, file_ids: Sequence[UUID]
) -> list[list[str]]:
    if not file_ids:
        return []
    paths = session.scalars(
        select(RegulatoryChunk.heading_path)
        .where(
            RegulatoryChunk.user_file_id.in_(file_ids),
            RegulatoryChunk.status != "superseded",
        )
        .distinct()
    ).all()
    return sorted([list(path) for path in paths if path])
