"""Predecessor suggestions distinguish own legal identity from download filenames."""

from copy import deepcopy
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.document_lineage import intake_lineage_selections, predecessor_candidates
from onyx.db.models import (
    DocumentSet,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryDocumentLineageReview,
    User,
    UserFile,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    _admitted,
    _existing_rows,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_owner as intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_session as intake_session,
)
from tests.external_dependency_unit.regulatory.test_document_lineage import _previous

_DOWNLOAD = "resmigazete.gov.tr_eskiler_2025_01_20250105-{}.htm.pdf"
_TITLE = "Ev Tipi Çamaşır Kurutma Makinelerinin Enerji Etiketlemesine Dair Tebliğ"


def _chunk(session: Session, file: UserFile) -> RegulatoryChunk:
    return session.scalars(
        select(RegulatoryChunk).where(
            RegulatoryChunk.user_file_id == file.id, RegulatoryChunk.position == 0
        )
    ).one()


def _identity(
    session: Session, file: UserFile, *, title: str = _TITLE, number: str = "SGM-2013/6"
) -> RegulatoryChunk:
    chunk = _chunk(session, file)
    chunk.chunk_metadata = {"document_title": title, "document_number": number}
    return chunk


def _new(
    session: Session, owner: tuple[User, DocumentSet, ModelConfiguration]
) -> tuple[RegulatoryDocumentIntake, RegulatoryDocumentIntakeFile, UserFile]:
    user, target, model = owner
    batch, members = _admitted(session, user, target, model)
    member = next(member for member in members if member.role == "root")
    file = session.get_one(UserFile, member.user_file_id)
    file.name = _DOWNLOAD.format("2")
    return batch, member, file


def _repeal(session: Session, file: UserFile) -> None:
    chunk = _identity(session, file, number="SGM-2024/4")
    chunk.heading_path = [_TITLE, "MADDE 9"]
    chunk.chunk_metadata = {**chunk.chunk_metadata, "article_no": "9"}
    chunk.text = "15/5/2013 tarihli ve 28648 sayılı Resmî Gazete’de yayımlanan Tebliğ (SGM-2013/6)"
    session.add(
        RegulatoryChunk(
            id=f"rc_repeal_{uuid4().hex}",
            user_file_id=file.id,
            text="yürürlükten kaldırılmıştır.",
            position=1,
            projection_ordinal=1,
            heading_path=[_TITLE, "MADDE 9"],
            chunk_type="article",
            chunk_metadata={"article_no": "9"},
            source="indexed",
            status="active",
        )
    )


def test_download_filenames_do_not_become_identical_legal_titles(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
) -> None:
    batch, member, _file = _new(intake_session, intake_owner)
    user, target, _model = intake_owner
    old = _previous(intake_session, user, target, _DOWNLOAD.format("3"))
    before = _existing_rows(intake_session, [old.id])
    assert predecessor_candidates(intake_session, batch, member.id) == []
    assert _existing_rows(intake_session, [old.id]) == before
    assert intake_lineage_selections(intake_session, batch) == {}


def test_own_legal_metadata_precedes_technical_name_and_shows_legal_title(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
) -> None:
    batch, member, file = _new(intake_session, intake_owner)
    _identity(intake_session, file, number="SGM-2024/4")
    user, target, _model = intake_owner
    unrelated = _previous(intake_session, user, target, _DOWNLOAD.format("3"))
    _identity(
        intake_session,
        unrelated,
        title="Ev ve Büro Tipi Elektronik Cihazların Hazırda Bekleme Tüketimi Tebliği",
        number="SGM-2024/5",
    )
    previous = _previous(intake_session, user, target, "20130515-7.pdf")
    _identity(intake_session, previous)
    result = predecessor_candidates(intake_session, batch, member.id)
    assert result[0]["user_file_id"] == str(previous.id)
    assert result[0]["title"] == _TITLE and result[0]["score"] == 100
    assert "kendi hukuki başlığı" in result[0]["reason"]
    assert all(row["user_file_id"] != str(unrelated.id) for row in result)
    assert intake_lineage_selections(intake_session, batch) == {}


def test_explicit_repeal_target_matches_own_number_not_cited_body_text(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
) -> None:
    batch, member, file = _new(intake_session, intake_owner)
    _repeal(intake_session, file)
    user, target, _model = intake_owner
    wrong = _previous(intake_session, user, target, _DOWNLOAD.format("3"))
    wrong_chunk = _identity(intake_session, wrong, number="SGM-2024/5")
    wrong_chunk.text = "SGM-2013/6 sayılı tebliğe yapılan bir atıftır."
    right = _previous(intake_session, user, target, "old-download.pdf")
    _identity(intake_session, right, number="SGM: 2013/6")
    before = _existing_rows(intake_session, [wrong.id, right.id])
    result = predecessor_candidates(intake_session, batch, member.id)
    assert [row["user_file_id"] for row in result] == [str(right.id)]
    assert (
        result[0]["score"] == 100
        and "kaldırılan mevzuatın kendi numarası" in result[0]["reason"]
    )
    assert _existing_rows(intake_session, [wrong.id, right.id]) == before
    assert not list(intake_session.scalars(select(RegulatoryDocumentLineageReview)))


def test_absent_explicit_target_does_not_fall_back_to_unrelated_title(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
) -> None:
    batch, member, file = _new(intake_session, intake_owner)
    _repeal(intake_session, file)
    user, target, _model = intake_owner
    wrong = _previous(intake_session, user, target, _DOWNLOAD.format("3"))
    _identity(intake_session, wrong, number="SGM-2024/5")
    assert predecessor_candidates(intake_session, batch, member.id) == []
    result = predecessor_candidates(
        intake_session, batch, member.id, query="SGM-2024/5"
    )
    assert [row["user_file_id"] for row in result] == [str(wrong.id)]
    assert "kendi mevzuat kimliği" in result[0]["reason"]
    assert intake_lineage_selections(intake_session, batch) == {}


@pytest.mark.parametrize(
    "invalid", ["stale_field", "stale_input", "conflicting", "aggregate_only"]
)
def test_stale_conflicting_or_derived_number_is_not_an_own_identity(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    invalid: str,
) -> None:
    batch, member, file = _new(intake_session, intake_owner)
    _repeal(intake_session, file)
    user, target, _model = intake_owner
    old = _previous(intake_session, user, target, "20130515-7.pdf")
    chunk = _identity(intake_session, old)
    if invalid in {"stale_field", "stale_input"}:
        chunk.chunk_metadata = {
            **chunk.chunk_metadata,
            "intake_legal_metadata": {
                "input_stale": invalid == "stale_input",
                "fields": {
                    "document_number": {
                        "source": "llm",
                        "stale": invalid == "stale_field",
                    }
                },
            },
        }
    elif invalid == "conflicting":
        intake_session.add(
            RegulatoryChunk(
                id=f"rc_conflict_{uuid4().hex}",
                user_file_id=old.id,
                text="Başka bir numara",
                position=1,
                projection_ordinal=1,
                heading_path=["MADDE 2"],
                chunk_type="article",
                chunk_metadata={"document_number": "SGM-2020/1"},
                source="indexed",
                status="active",
            )
        )
    else:
        chunk.chunk_type = "hierarchical_aggregate"
    before = deepcopy(chunk.chunk_metadata)
    assert predecessor_candidates(intake_session, batch, member.id) == []
    assert chunk.chunk_metadata == before


def test_different_numbered_laws_keep_meaningful_title_fallback_and_manual_number_search(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
) -> None:
    batch, member, file = _new(intake_session, intake_owner)
    file.name = "5000 Sayılı Gümrük Kanunu.md"
    user, target, _model = intake_owner
    old = _previous(intake_session, user, target, "4000 sayılı Gümrük Kanunu.pdf")
    assert predecessor_candidates(intake_session, batch, member.id)[0][
        "user_file_id"
    ] == str(old.id)
    assert predecessor_candidates(intake_session, batch, member.id, query="4000")[0][
        "user_file_id"
    ] == str(old.id)
    assert intake_lineage_selections(intake_session, batch) == {}


@pytest.mark.parametrize("source", ["indexed", "amendment"])
def test_exact_identity_reads_current_legacy_or_update_metadata_with_nullable_type(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    source: str,
) -> None:
    batch, member, file = _new(intake_session, intake_owner)
    _repeal(intake_session, file)
    user, target, _model = intake_owner
    previous = _previous(intake_session, user, target, "20130515-7.pdf")
    chunk = _identity(intake_session, previous)
    chunk.source, chunk.chunk_type = source, None
    result = predecessor_candidates(intake_session, batch, member.id)
    assert [row["user_file_id"] for row in result] == [str(previous.id)]
