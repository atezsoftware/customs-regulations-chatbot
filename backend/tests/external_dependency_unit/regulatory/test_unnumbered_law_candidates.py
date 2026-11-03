"""Natural law candidates use real scoped PostgreSQL metadata and read fences."""

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.context.search.models import IndexFilters
from onyx.db import asv3_corpus
from onyx.db.models import (
    DocumentSet,
    RegulatoryChunk,
    RegulatoryFilePublication,
    User,
    UserFile,
    UserRole,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import LabelingData
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_data as labeling_data,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)


def test_actual_natural_law_metadata_candidate_preserves_all_existing_fences(
    labeling_data: LabelingData,
) -> None:
    data = labeling_data
    captures = json.loads(
        (
            Path(__file__).parents[2]
            / "unit/onyx/asv3/fixtures/numbered_intake_display_names.json"
        ).read_text()
    )["laws"]
    captured = next(row for row in captures if row["number"] == "5393")
    with Session(data.database.engine) as session:
        owner = session.get_one(User, data.user_id)
        corpus = session.get_one(DocumentSet, data.document_set_id)
        corpus.name = asv3_corpus.PC_CORPUS_NAME
        source = session.get_one(UserFile, data.file_id)
        source.name = "Belediye Kanunu"
        for identifier, row in zip(
            (data.first_id, data.second_id), captured["candidate_heads"], strict=True
        ):
            chunk = session.get_one(RegulatoryChunk, identifier)
            chunk.text, chunk.heading_path = row["text"], row["heading_path"]
            chunk.chunk_metadata = row["chunk_metadata"]
        stranger = User(
            id=uuid4(),
            email=f"natural-law-{uuid4().hex}@test.local",
            hashed_password="unused",
            is_active=True,
            is_verified=True,
            role=UserRole.BASIC,
        )
        session.add(stranger)
        session.commit()
        filters = asv3_corpus.resolve_pc_corpus_scope(
            session,
            user=owner,
            filters=IndexFilters(access_control_list=[], as_of_date=date(2026, 5, 18)),
        )

        def lookup(
            query: str = "5393 sayılı Kanun",
            *,
            allow_canonical_filename_fallback: bool = False,
            source_ids: tuple[UUID, ...] | None = None,
        ) -> tuple[list[asv3_corpus.CorpusSource], bool]:
            return asv3_corpus.find_sources(
                session,
                user=owner,
                filters=filters,
                query=query,
                allow_canonical_filename_fallback=allow_canonical_filename_fallback,
                source_ids=source_ids,
            )

        try:
            assert lookup() == ([], False)
            assert lookup(
                allow_canonical_filename_fallback=True, source_ids=(source.id,)
            ) == ([], False)
            direct, more = lookup("", source_ids=(source.id,))
            assert not more and [row.id for row in direct] == [source.id]
            for query in (
                "5393 sayılı Kanun",
                "05393 sayılı Kanun",
                "5393 sayılı Belediye Kanunu",
            ):
                found, more = lookup(query, allow_canonical_filename_fallback=True)
                assert not more and [row.id for row in found] == [source.id]
                assert [row.text for row in found[0].identity_originals] == [
                    captured["candidate_heads"][0]["text"]
                ]
            with pytest.raises(PermissionError):
                asv3_corpus.find_sources(
                    session,
                    user=stranger,
                    filters=filters,
                    query="5393 sayılı Kanun",
                    allow_canonical_filename_fallback=True,
                )
            publication = RegulatoryFilePublication(
                user_file_id=source.id,
                scope_key="1" * 64,
                owner_id=uuid4(),
                fencing_token=1,
                lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
                gate_closed=True,
                epoch=0,
                next_ordinal=4,
            )
            session.add(publication)
            session.commit()
            assert lookup(allow_canonical_filename_fallback=True) == ([], False)
            session.delete(publication)
            first = session.get_one(RegulatoryChunk, data.first_id)
            first.chunk_metadata = {**first.chunk_metadata, "document_number": "3213"}
            session.commit()
            # The second row still locates the candidate; its contradictory own head denies it.
            assert lookup(allow_canonical_filename_fallback=True) == ([], False)
        finally:
            session.rollback()
            session.delete(stranger)
            session.commit()
