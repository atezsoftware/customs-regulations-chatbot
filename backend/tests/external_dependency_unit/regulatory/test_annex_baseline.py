from datetime import date, datetime, timezone
from io import BytesIO
from uuid import uuid4

import pytest
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.models import (
    DocumentSet,
    DocumentSet__UserFile,
    RegulatoryChunk,
    User,
    UserFile,
)
from tests.external_dependency_unit.regulatory.test_amendment_sources import (
    source_session as source_session,
)


def _file(session: Session, document_set: DocumentSet) -> UserFile:
    user = User(
        id=uuid4(),
        email=f"{uuid4()}@test.local",
        hashed_password="unused",
        role="basic",
        is_active=True,
        is_superuser=False,
        is_verified=True,
    )
    session.add(user)
    session.flush()
    file = UserFile(
        id=uuid4(),
        user_id=user.id,
        name="Regulation",
        file_id=str(uuid4()),
        file_type="text/markdown",
    )
    session.add(file)
    session.flush()
    session.add(
        DocumentSet__UserFile(document_set_id=document_set.id, user_file_id=file.id)
    )
    session.flush()
    return file


def _chunk(
    session: Session, file: UserFile, position: int, text: str, **metadata: object
) -> RegulatoryChunk:
    row = RegulatoryChunk(
        id=str(uuid4()),
        user_file_id=file.id,
        text=text,
        position=position,
        projection_ordinal=position,
        heading_path=["EK-1"],
        chunk_type="appendix",
        chunk_metadata={"appendix_label": "EK-1", **metadata},
        source="indexed",
        status="active",
    )
    session.add(row)
    session.flush()
    return row


def test_scoped_complete_legacy_baseline_uses_real_image_companions(
    source_session: Session,
) -> None:
    from onyx.configs.constants import FileOrigin
    from onyx.file_store.postgres_file_store import PostgresBackedFileStore
    from onyx.regulatory.amendments.annexes.baseline import prepare_legacy_baseline

    document_set = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    source_session.add(document_set)
    source_session.flush()
    file = _file(source_session, document_set)
    other = _file(source_session, document_set)
    store = PostgresBackedFileStore()
    atomic = [_chunk(source_session, file, i, f"Approved {i}") for i in range(40)]
    _chunk(source_session, other, 0, "WRONG REGULATION")
    _chunk(
        source_session,
        file,
        41,
        "aggregate duplicate",
        chunk_variant="hierarchical_aggregate",
        source_regulatory_chunk_ids=[row.id for row in atomic],
    )
    buffer = BytesIO()
    Image.new("RGB", (10, 20)).save(buffer, format="PNG")
    image_id = store.save_file(
        BytesIO(buffer.getvalue()),
        display_name="old.png",
        file_origin=FileOrigin.OTHER,
        file_type="image/png",
    )
    companion = _chunk(
        source_session,
        file,
        42,
        "Approved image transcript",
        chunk_variant="image_companion",
        image_file_id=image_id,
        bound_to_regulatory_chunk_id=atomic[0].id,
    )
    companion.chunk_metadata = {
        key: value
        for key, value in companion.chunk_metadata.items()
        if key != "appendix_label"
    }
    source_session.flush()
    baseline = prepare_legacy_baseline(
        source_session,
        store,
        document_set_id=document_set.id,
        user_file_id=file.id,
        annex_label="EK 1",
        as_of_date=date(2026, 1, 1),
    )
    assert len(baseline.elements) == 41
    assert "WRONG REGULATION" not in [item.text for item in baseline.elements]
    assert len(baseline.originals) == 1
    assert baseline.originals[0].available and baseline.originals[0].sha256
    assert baseline.visual_evidence_available
    assert baseline.canonical_text.startswith("Approved 0")
    assert (
        baseline.revision_id
        == prepare_legacy_baseline(
            source_session,
            store,
            document_set_id=document_set.id,
            user_file_id=file.id,
            annex_label="EK-1",
            as_of_date=date(2026, 1, 1),
        ).revision_id
    )
    with pytest.raises(ValueError, match="scope"):
        prepare_legacy_baseline(
            source_session,
            store,
            document_set_id=document_set.id + 100,
            user_file_id=file.id,
            annex_label="EK-1",
            as_of_date=date(2026, 1, 1),
        )
    store.delete_file(image_id)
    missing = prepare_legacy_baseline(
        source_session,
        store,
        document_set_id=document_set.id,
        user_file_id=file.id,
        annex_label="EK-1",
        as_of_date=date(2026, 1, 1),
    )
    assert not missing.visual_evidence_available
    assert missing.baseline_sha256 != baseline.baseline_sha256
    assert missing.canonical_text == baseline.canonical_text
    assert missing.originals[0].issue == "original_unavailable"


def test_revision_dates_and_row_identity_do_not_follow_latest_approval(
    source_session: Session,
) -> None:
    from onyx.db.regulatory_annexes import (
        get_effective_annex_revision,
        get_or_create_annex,
        get_revision_elements,
        persist_annex_revision,
    )
    from onyx.regulatory.amendments.annexes.extraction import extract_annex_structure

    document_set = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    source_session.add(document_set)
    source_session.flush()
    file = _file(source_session, document_set)
    annex = get_or_create_annex(
        source_session,
        document_set_id=document_set.id,
        user_file_id=file.id,
        annex_label="EK-1",
    )
    old = extract_annex_structure(
        b"<table><tr><th>Code</th><th>Rate</th></tr><tr><td>A</td><td>5</td></tr><tr><td>B</td><td>8</td></tr></table>",
        "text/html",
    )
    first = persist_annex_revision(
        source_session,
        annex_id=annex.id,
        extraction=old,
        baseline_sha256="a" * 64,
        effective_start=None,
        effective_end=date(2027, 1, 1),
        approved_at=datetime.now(timezone.utc),
    )
    new = extract_annex_structure(
        b"<table><tr><th>Code</th><th>Rate</th></tr><tr><td>A</td><td>7</td></tr><tr><td>X</td><td>6</td></tr><tr><td>B</td><td>8</td></tr></table>",
        "text/html",
    )
    second = persist_annex_revision(
        source_session,
        annex_id=annex.id,
        extraction=new,
        baseline_sha256="b" * 64,
        effective_start=date(2027, 1, 1),
        effective_end=None,
        approved_at=datetime.now(timezone.utc),
        predecessor_revision_id=first.id,
    )
    assert annex.latest_approved_revision_id == second.id
    current = get_effective_annex_revision(source_session, annex.id, date(2026, 1, 1))
    future = get_effective_annex_revision(source_session, annex.id, date(2027, 1, 1))
    assert current is not None and current.id == first.id
    assert future is not None and future.id == second.id
    before = {
        row.payload["semantic_key"]: row
        for row in get_revision_elements(source_session, first.id)
    }
    after = {
        row.payload["semantic_key"]: row
        for row in get_revision_elements(source_session, second.id)
    }
    for key in before:
        assert before[key].element_id == after[key].element_id
    assert (
        next(row for row in after.values() if row.payload["text"] == "8").position
        != next(row for row in before.values() if row.payload["text"] == "8").position
    )
    assert (
        len(
            source_session.scalars(select(UserFile).where(UserFile.id == file.id)).all()
        )
        == 1
    )


def test_initial_index_preserves_section_image_and_reindex_keeps_links(
    source_session: Session,
) -> None:
    from typing import cast

    from onyx.configs.constants import DocumentSource
    from onyx.connectors.models import Document, TextSection
    from onyx.natural_language_processing.utils import BaseTokenizer
    from onyx.regulatory.indexing import documents_to_regulatory_chunks

    class Tokenizer:
        def encode(self, text: str) -> list[int]:
            return list(range(len(text.split())))

    document_set = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    source_session.add(document_set)
    source_session.flush()
    file = _file(source_session, document_set)
    document = Document(
        id=str(file.id),
        source=DocumentSource.USER_FILE,
        metadata={},
        semantic_identifier="Regulation",
        sections=[
            TextSection(
                text="EK-1\n\nApproved image transcript",
                image_file_id="original-image",
                link="https://example.gov/annex",
            )
        ],
    )
    for _ in range(2):
        chunks = documents_to_regulatory_chunks(
            [document],
            source_session,
            cast(BaseTokenizer, Tokenizer()),
            enable_contextual_rag=False,
        )
        assert chunks[0].image_file_id == "original-image"
        assert chunks[0].source_links == {0: "https://example.gov/annex"}

    section = document.sections[0]
    assert isinstance(section, TextSection)
    section.image_file_id = None
    section.link = None
    preserved = documents_to_regulatory_chunks(
        [document],
        source_session,
        cast(BaseTokenizer, Tokenizer()),
        enable_contextual_rag=False,
    )
    assert preserved[0].image_file_id == "original-image"
    assert preserved[0].source_links == {0: "https://example.gov/annex"}


def test_annex_linked_and_historical_rows_cannot_be_destroyed_by_raw_rechunk(
    source_session: Session,
) -> None:
    from onyx.db.regulatory_chunks import replace_indexed_chunks_for_file
    from onyx.regulatory.chunker import RegulatoryChunker

    document_set = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    source_session.add(document_set)
    source_session.flush()
    file = _file(source_session, document_set)
    row = _chunk(source_session, file, 0, "Historical approved original")
    row.status = "superseded"
    row.validity_end_date = date(2026, 1, 1)
    source_session.flush()
    with pytest.raises(ValueError, match="canonical"):
        replace_indexed_chunks_for_file(
            source_session,
            file.id,
            RegulatoryChunker().chunk_text("New source text").chunks,
        )
    preserved = source_session.get(RegulatoryChunk, row.id)
    assert preserved is not None and preserved.text == "Historical approved original"


def test_annex_element_links_follow_approved_chunk_lineage(
    source_session: Session,
) -> None:
    from onyx.db.models import RegulatoryAnnexElementChunk
    from onyx.db.regulatory_annexes import (
        copy_annex_chunk_links,
        get_or_create_annex,
        persist_annex_revision,
    )
    from onyx.regulatory.amendments.annexes.models import (
        AnnexExtraction,
        ExtractedAnnexElement,
    )

    document_set = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    source_session.add(document_set)
    source_session.flush()
    file = _file(source_session, document_set)
    old = _chunk(source_session, file, 0, "old")
    new = _chunk(source_session, file, 1, "approved correction")
    annex = get_or_create_annex(
        source_session,
        document_set_id=document_set.id,
        user_file_id=file.id,
        annex_label="EK-1",
    )
    revision = persist_annex_revision(
        source_session,
        annex_id=annex.id,
        extraction=AnnexExtraction(
            source_sha256="a" * 64,
            mime_type="text/markdown",
            elements=[ExtractedAnnexElement(kind="text", text=old.text)],
        ),
        baseline_sha256="c" * 64,
        effective_start=None,
        effective_end=None,
        approved_at=datetime.now(timezone.utc),
        chunk_ids_by_position={0: [old.id]},
    )
    copy_annex_chunk_links(source_session, old_chunk_id=old.id, new_chunk_id=new.id)
    links = source_session.scalars(
        select(RegulatoryAnnexElementChunk).where(
            RegulatoryAnnexElementChunk.revision_id == revision.id
        )
    ).all()
    assert {link.chunk_id for link in links} == {old.id, new.id}
    assert len({link.element_id for link in links}) == 1
    other = _file(source_session, document_set)
    wrong = _chunk(source_session, other, 0, "other regulation")
    with pytest.raises(ValueError, match="scope"):
        copy_annex_chunk_links(
            source_session, old_chunk_id=old.id, new_chunk_id=wrong.id
        )


def test_extract_ready_asset_consumes_verified_filestore_bytes(
    source_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from collections.abc import Generator
    from contextlib import contextmanager

    from onyx.db.amendment_sources import create_source_package, list_source_assets
    from onyx.file_store.file_store import get_default_file_store
    from onyx.regulatory.amendments.annexes import job
    from onyx.regulatory.amendments.annexes.extraction import extract_source_asset

    @contextmanager
    def session_context() -> Generator[Session, None, None]:
        yield source_session

    monkeypatch.setattr(job, "get_session_with_current_tenant", session_context)
    document_set = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    source_session.add(document_set)
    source_session.flush()
    from onyx.configs.constants import FileOrigin

    input_id = get_default_file_store().save_file(
        BytesIO(b"EK-1 approved source"),
        display_name="source.txt",
        file_origin=FileOrigin.OTHER,
        file_type="text/plain",
    )
    package, _ = create_source_package(
        source_session,
        document_set_id=document_set.id,
        environment="local-test",
        idempotency_key=str(uuid4()),
        request_hash="a" * 64,
        input_spec={"mime_type": "text/plain"},
        input_file_id=input_id,
        created_by=None,
    )
    job.run_source_package(package_id=package.id, environment="local-test")
    source_session.expire_all()
    asset = list_source_assets(source_session, package.id)[0]
    extraction = extract_source_asset(
        source_session,
        get_default_file_store(),
        package_id=package.id,
        asset_id=asset.id,
        document_set_id=document_set.id,
        environment="local-test",
    )
    assert extraction.source_sha256 == asset.sha256
    assert extraction.elements[0].text == "EK-1 approved source"
    assert extraction.elements[0].source_asset_id == str(asset.id)
    with pytest.raises(ValueError):
        extract_source_asset(
            source_session,
            get_default_file_store(),
            package_id=package.id,
            asset_id=asset.id,
            document_set_id=document_set.id + 100,
            environment="local-test",
        )


def test_baseline_preserves_approved_correction_without_relabeling_old_pixels(
    source_session: Session,
) -> None:
    from onyx.file_store.file_store import get_default_file_store
    from onyx.regulatory.amendments.annexes.baseline import prepare_legacy_baseline

    document_set = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    source_session.add(document_set)
    source_session.flush()
    file = _file(source_session, document_set)
    old = _chunk(source_session, file, 0, "Raw 5%", image_file_id="missing-old-image")
    old.status = "superseded"
    old.validity_end_date = date(2026, 1, 1)
    corrected = _chunk(
        source_session, file, 1, "Approved 7%", image_file_id="missing-old-image"
    )
    corrected.source = "amendment"
    corrected.supersedes_chunk_id = old.id
    corrected.validity_start_date = date(2026, 1, 1)
    companion = _chunk(
        source_session,
        file,
        2,
        "Supporting original 5%",
        image_file_id="missing-old-image",
        chunk_variant="image_companion",
        bound_to_regulatory_chunk_id=old.id,
    )
    companion.chunk_metadata.pop("appendix_label")
    companion.heading_path = ["Image evidence"]
    source_session.flush()
    current = prepare_legacy_baseline(
        source_session,
        get_default_file_store(),
        document_set_id=document_set.id,
        user_file_id=file.id,
        annex_label="EK-1",
        as_of_date=date(2026, 2, 1),
    )
    historical = prepare_legacy_baseline(
        source_session,
        get_default_file_store(),
        document_set_id=document_set.id,
        user_file_id=file.id,
        annex_label="EK-1",
        as_of_date=date(2025, 12, 1),
    )
    assert current.canonical_text == "Approved 7%"
    assert historical.canonical_text == "Raw 5%"
    assert current.canonical_amendment_chunk_ids == [corrected.id]
    assert current.originals[0].canonical_chunk_ids == [corrected.id, companion.id]
    supporting = next(
        element
        for element in current.elements
        if element.canonical_chunk_id == companion.id
    )
    assert supporting.canonical_role == "supporting"
    assert supporting.bound_to_regulatory_chunk_id == old.id
    assert supporting.text == "Supporting original 5%"
    assert "raw_original_may_precede_approved_correction" in current.issues
    assert not current.visual_evidence_available


def test_revision_content_is_immutable(source_session: Session) -> None:
    from sqlalchemy import update
    from sqlalchemy.exc import DBAPIError

    from onyx.db.models import RegulatoryAnnexRevision
    from onyx.db.regulatory_annexes import get_or_create_annex, persist_annex_revision
    from onyx.regulatory.amendments.annexes.models import AnnexExtraction

    document_set = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    source_session.add(document_set)
    source_session.flush()
    file = _file(source_session, document_set)
    annex = get_or_create_annex(
        source_session,
        document_set_id=document_set.id,
        user_file_id=file.id,
        annex_label="EK-1",
    )
    revision = persist_annex_revision(
        source_session,
        annex_id=annex.id,
        extraction=AnnexExtraction(source_sha256="a" * 64, mime_type="text/plain"),
        baseline_sha256="a" * 64,
        effective_start=None,
        effective_end=None,
        approved_at=datetime.now(timezone.utc),
    )
    with pytest.raises(DBAPIError, match="immutable"):
        with source_session.begin_nested():
            source_session.execute(
                update(RegulatoryAnnexRevision)
                .where(RegulatoryAnnexRevision.id == revision.id)
                .values(snapshot={"fabricated": "replacement"})
            )


def test_annex_labels_preserve_suffix_scope_and_reject_noisy_legacy_label() -> None:
    from onyx.db.regulatory_annexes import normalize_annex_label

    assert normalize_annex_label("EK 1") == normalize_annex_label("EK-1")
    assert normalize_annex_label("EK iii") != normalize_annex_label("EK ii")
    assert normalize_annex_label("EK 1/A") != normalize_annex_label("EK 1A")
    with pytest.raises(ValueError, match="ambiguous"):
        normalize_annex_label("EK EK")


def test_validated_correspondence_preserves_changed_vision_element_identity(
    source_session: Session,
) -> None:
    from onyx.db.regulatory_annexes import (
        get_or_create_annex,
        get_revision_elements,
        persist_annex_revision,
    )
    from onyx.regulatory.amendments.annexes.models import (
        AnnexExtraction,
        ExtractedAnnexElement,
    )

    document_set = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    source_session.add(document_set)
    source_session.flush()
    file = _file(source_session, document_set)
    annex = get_or_create_annex(
        source_session,
        document_set_id=document_set.id,
        user_file_id=file.id,
        annex_label="EK-1",
    )
    old = persist_annex_revision(
        source_session,
        annex_id=annex.id,
        extraction=AnnexExtraction(
            source_sha256="a" * 64,
            mime_type="image/png",
            elements=[ExtractedAnnexElement(kind="table_cell", text="5%")],
        ),
        baseline_sha256="a" * 64,
        effective_start=None,
        effective_end=None,
        approved_at=None,
    )
    identity = get_revision_elements(source_session, old.id)[0].element_id
    new = AnnexExtraction(
        source_sha256="b" * 64,
        mime_type="image/png",
        elements=[ExtractedAnnexElement(kind="table_cell", text="7%")],
    )
    with pytest.raises(ValueError, match="correspondence"):
        persist_annex_revision(
            source_session,
            annex_id=annex.id,
            extraction=new,
            baseline_sha256="x" * 64,
            effective_start=None,
            effective_end=None,
            approved_at=None,
            predecessor_revision_id=old.id,
            element_id_by_position={0: uuid4()},
        )
    revision = persist_annex_revision(
        source_session,
        annex_id=annex.id,
        extraction=new,
        baseline_sha256="b" * 64,
        effective_start=None,
        effective_end=None,
        approved_at=None,
        predecessor_revision_id=old.id,
        element_id_by_position={0: identity},
    )
    assert get_revision_elements(source_session, revision.id)[0].element_id == identity
    assert get_revision_elements(source_session, old.id)[0].payload["text"] == "5%"
