"""PostgreSQL coverage for document-set labeling setup counts."""

from collections.abc import Sequence
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from onyx.db import regulatory_labeling as repository
from onyx.db.models import (
    DocumentSet,
    DocumentSet__UserFile,
    RegulatoryChunk,
    UserFile,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import LabelingData
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_data as labeling_data,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)

ChunkSpec = tuple[str | None, dict[str, object], str]


def _add_file(
    session: Session,
    *,
    data: LabelingData,
    document_set_id: int,
    chunks: Sequence[ChunkSpec],
) -> UUID:
    file_id = uuid4()
    session.add(
        UserFile(
            id=file_id,
            user_id=data.user_id,
            file_id=str(uuid4()),
            name=f"Counting fixture {file_id}",
            file_type="text/markdown",
        )
    )
    session.flush()
    session.add(
        DocumentSet__UserFile(
            document_set_id=document_set_id,
            user_file_id=file_id,
        )
    )
    for position, (chunk_type, metadata, status) in enumerate(chunks):
        session.add(
            RegulatoryChunk(
                id=str(uuid4()),
                user_file_id=file_id,
                text=f"Chunk {position}",
                position=position,
                projection_ordinal=position,
                chunk_type=chunk_type,
                heading_path=[],
                chunk_metadata=metadata,
                source="indexed",
                status=status,
            )
        )
    session.flush()
    return file_id


def _add_set(session: Session, data: LabelingData) -> int:
    document_set = DocumentSet(
        name=f"Counting set {uuid4()}",
        user_id=data.user_id,
        is_public=False,
    )
    session.add(document_set)
    session.flush()
    return document_set.id


def _ungrouped_chunk_counts(session: Session, document_set_id: int) -> tuple[int, int]:
    atomic_expression = repository._atomic_sql_expression()
    canonical, derived = session.execute(
        select(
            func.count().filter(atomic_expression),
            func.count().filter(~atomic_expression),
        )
        .join(
            DocumentSet__UserFile,
            DocumentSet__UserFile.user_file_id == RegulatoryChunk.user_file_id,
        )
        .where(
            DocumentSet__UserFile.document_set_id == document_set_id,
            RegulatoryChunk.status == "active",
        )
    ).one()
    return canonical, derived


def test_counts_preserve_canonical_and_derived_metadata_semantics(
    labeling_data: LabelingData,
) -> None:
    with Session(labeling_data.database.engine) as session:
        document_set_id = _add_set(session, labeling_data)
        file_id = _add_file(
            session,
            data=labeling_data,
            document_set_id=document_set_id,
            chunks=[
                ("article", {"chunk_variant": "atomic"}, "active"),
                ("hierarchical_aggregate", {}, "active"),
                (
                    "article",
                    {"chunk_variant": "hierarchical_aggregate"},
                    "active",
                ),
                (
                    "article",
                    {"bound_to_regulatory_chunk_id": "canonical-id"},
                    "active",
                ),
                (
                    "article",
                    {"source_regulatory_chunk_ids": ["canonical-id"]},
                    "active",
                ),
                ("article", {}, "active"),
                (
                    "article",
                    {
                        "chunk_variant": None,
                        "bound_to_regulatory_chunk_id": None,
                        "source_regulatory_chunk_ids": None,
                    },
                    "active",
                ),
                ("article", {"source_regulatory_chunk_ids": []}, "active"),
                (
                    "article",
                    {"source_regulatory_chunk_ids": "malformed-source"},
                    "active",
                ),
                ("article", {"chunk_variant": "unknown-variant"}, "active"),
            ],
        )

        rows = list(
            session.scalars(
                select(RegulatoryChunk).where(RegulatoryChunk.user_file_id == file_id)
            )
        )
        known_variant_rows = [
            row
            for row in rows
            if row.chunk_metadata.get("chunk_variant") != "unknown-variant"
        ]
        python_counts_for_known_variants = (
            sum(repository._is_atomic(row) for row in known_variant_rows),
            sum(not repository._is_atomic(row) for row in known_variant_rows),
        )
        ungrouped_counts = _ungrouped_chunk_counts(session, document_set_id)
        counts, warnings = repository.get_labeling_counts(session, document_set_id)

        assert (counts.files, counts.canonical_chunks, counts.derived_chunks) == (
            1,
            3,
            7,
        )
        assert python_counts_for_known_variants == (3, 6)
        assert (counts.canonical_chunks, counts.derived_chunks) == ungrouped_counts
        assert counts.canonical_chunks + counts.derived_chunks == len(rows)
        assert warnings == []


def test_counts_are_active_only_set_scoped_and_report_files_without_canonical_chunks(
    labeling_data: LabelingData,
) -> None:
    with Session(labeling_data.database.engine) as session:
        document_set_id = _add_set(session, labeling_data)
        other_set_id = _add_set(session, labeling_data)
        _add_file(
            session,
            data=labeling_data,
            document_set_id=document_set_id,
            chunks=[
                ("article", {"chunk_variant": "atomic"}, "active"),
                ("article", {"chunk_variant": "atomic"}, "superseded"),
                (
                    "article",
                    {"chunk_variant": "hierarchical_aggregate"},
                    "active",
                ),
            ],
        )
        _add_file(
            session,
            data=labeling_data,
            document_set_id=document_set_id,
            chunks=[
                (
                    "article",
                    {"source_regulatory_chunk_ids": ["canonical-id"]},
                    "active",
                )
            ],
        )
        _add_file(
            session,
            data=labeling_data,
            document_set_id=document_set_id,
            chunks=[("article", {"chunk_variant": "atomic"}, "superseded")],
        )
        _add_file(
            session,
            data=labeling_data,
            document_set_id=document_set_id,
            chunks=[],
        )
        _add_file(
            session,
            data=labeling_data,
            document_set_id=other_set_id,
            chunks=[("article", {"chunk_variant": "atomic"}, "active")],
        )

        ungrouped_counts = _ungrouped_chunk_counts(session, document_set_id)
        counts, warnings = repository.get_labeling_counts(session, document_set_id)
        other_counts, other_warnings = repository.get_labeling_counts(
            session, other_set_id
        )

        assert (counts.files, counts.canonical_chunks, counts.derived_chunks) == (
            4,
            1,
            2,
        )
        assert (counts.canonical_chunks, counts.derived_chunks) == ungrouped_counts
        assert warnings == ["3 file(s) have no active canonical regulatory chunks."]
        assert (
            other_counts.files,
            other_counts.canonical_chunks,
            other_counts.derived_chunks,
        ) == (1, 1, 0)
        assert other_warnings == []
