"""Published family scope and actual annex structure are checked in PostgreSQL."""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from types import SimpleNamespace
from typing import Literal
from unittest.mock import MagicMock, Mock
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import document_intake_review
from onyx.db.enums import UserFileStatus
from onyx.db.models import (
    AmendmentSourcePackage,
    DocumentSet__UserFile,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryDocumentIntakeReviewDecision,
    RegulatoryFilePublication,
    RegulatorySourceAsset,
    UserFile,
)
from onyx.db.regulatory_amendment_source_families import resolve_amendment_source_family
from onyx.db.regulatory_writer_publication import canonical_scope_digest
from onyx.regulatory.amendments import search_retriever
from onyx.regulatory.amendments.annexes.models import SourceLink
from onyx.regulatory.amendments.models import AmendmentInstruction
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    LabelingData,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_data as labeling_data,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)


def family(
    session: Session,
    data: LabelingData,
    *,
    link_kind: Literal["url", "embedded", "internal"] = "url",
) -> tuple[
    UserFile,
    UserFile,
    AmendmentSourcePackage,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryDocumentIntakeFile,
    list[RegulatoryFilePublication],
]:
    primary = session.get_one(UserFile, data.file_id)
    primary.status = UserFileStatus.COMPLETED
    child = UserFile(
        id=uuid4(),
        user_id=data.user_id,
        file_id=str(uuid4()),
        name="Verified annex PDF",
        file_type="application/pdf",
        status=UserFileStatus.COMPLETED,
    )
    session.add(child)
    session.flush()
    session.add(
        DocumentSet__UserFile(
            document_set_id=data.document_set_id, user_file_id=child.id
        )
    )
    package = AmendmentSourcePackage(
        id=uuid4(),
        document_set_id=data.document_set_id,
        environment="family-local-test",
        idempotency_key=str(uuid4()),
        request_hash="a" * 64,
        input_spec={},
        status="ready",
        asset_count=2,
        total_bytes=2,
        issues=[],
        manifest_file_id="frozen-family-manifest",
        manifest_sha256="b" * 64,
        created_by=data.user_id,
    )
    session.add(package)
    session.flush()
    assets = [
        RegulatorySourceAsset(
            id=uuid4(),
            package_id=package.id,
            sha256=sha256(str(file.id).encode()).hexdigest(),
            file_id=file.file_id,
            text_file_id=file.file_id + "-prepared",
            text_sha256="c" * 64,
            mime_type=file.file_type,
            display_name=file.name,
            byte_count=1,
        )
        for file in (primary, child)
    ]
    session.add_all(assets)
    batch = RegulatoryDocumentIntake(
        id=uuid4(),
        source_package_id=package.id,
        document_set_id=data.document_set_id,
        created_by=data.user_id,
        idempotency_key=uuid4(),
        labeling_model_configuration_id=data.model_id,
        status="completed",
        stage="completed",
        review_sha256="d" * 64,
    )
    session.add(batch)
    session.flush()
    link = SourceLink(
        parent_asset_hash=assets[0].sha256,
        target_asset_hash=assets[1].sha256,
        source_field="pdf:annotation:0",
        source_page=3,
        label="Ekleri için tıklayınız",
        kind=link_kind,
        original_url="annex.pdf",
        requested_url="https://example.gov/annex.pdf",
    )
    parent_member = RegulatoryDocumentIntakeFile(
        id=uuid4(),
        intake_id=batch.id,
        source_asset_id=assets[0].id,
        user_file_id=primary.id,
        role="root",
        relationships=[link.model_dump(mode="json")],
        status="completed",
        approval_attempt_id=uuid4(),
    )
    child_member = RegulatoryDocumentIntakeFile(
        id=uuid4(),
        intake_id=batch.id,
        source_asset_id=assets[1].id,
        user_file_id=child.id,
        parent_document_id=parent_member.id,
        role="annex",
        relationships=[link.model_dump(mode="json")],
        status="completed",
        approval_attempt_id=uuid4(),
    )
    session.add(parent_member)
    session.flush()
    session.add(child_member)
    publications = []
    for file in (primary, child):
        publication = RegulatoryFilePublication(
            user_file_id=file.id,
            scope_key="owned-local-family",
            owner_id=uuid4(),
            fencing_token=1,
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
            gate_closed=False,
            epoch=1,
            next_ordinal=4,
        )
        session.add(publication)
        publications.append(publication)
    for position, label in enumerate(["EK I", "EK II", "EK IV"]):
        session.add(
            RegulatoryChunk(
                id="rc_family_" + uuid4().hex,
                user_file_id=child.id,
                position=position,
                projection_ordinal=position,
                text=f"Actual {label} body",
                heading_path=["Own annex", label],
                chunk_type="paragraph",
                chunk_metadata={"appendix_label": label},
                source="indexed",
                status="active",
            )
        )
    session.flush()
    return primary, child, package, batch, parent_member, child_member, publications


@pytest.mark.parametrize("label", [None, "EK I", "EK II", "EK IV", "EK VIII"])
def test_instruction_scope_uses_primary_or_actual_annex(
    labeling_data: LabelingData, label: str | None
) -> None:
    with Session(labeling_data.database.engine) as session:
        primary, child, _package, batch, _parent, _child, _publications = family(
            session, labeling_data
        )
        for matched in ([primary.id], [primary.id, child.id]):
            result = resolve_amendment_source_family(
                session,
                document_set_id=labeling_data.document_set_id,
                user_file_ids=[primary.id, child.id],
                matched_file_ids=matched,
                appendix_label=label,
            )
            assert (
                result is not None
                and result.primary_file_id == primary.id
                and result.intake_id == batch.id
            )
            assert set(result.member_file_ids) == {primary.id, child.id}
            assert result.target_file_ids == (
                ()
                if label == "EK VIII"
                else (primary.id,)
                if label is None
                else (child.id,)
            )


@pytest.mark.parametrize(
    "condition", ["valid", "stale_snapshot", "membership", "unreviewed"]
)
def test_reviewed_source_evidence_does_not_hide_published_annex_family(
    labeling_data: LabelingData,
    condition: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(labeling_data.database.engine) as session:
        primary, child, package, batch, parent, _child, _pub = family(
            session, labeling_data
        )
        evidence = UserFile(
            id=uuid4(),
            user_id=labeling_data.user_id,
            file_id=str(uuid4()),
            name="Official HTML rendition",
            file_type="text/html",
            status=UserFileStatus.CHUNKED,
        )
        session.add(evidence)
        session.flush()
        asset = RegulatorySourceAsset(
            id=uuid4(),
            package_id=package.id,
            sha256="f" * 64,
            file_id=evidence.file_id,
            text_file_id="prepared-evidence",
            text_sha256="e" * 64,
            display_name="source.htm",
            mime_type="text/html",
            byte_count=1,
        )
        session.add(asset)
        session.flush()
        member = RegulatoryDocumentIntakeFile(
            id=uuid4(),
            intake_id=batch.id,
            source_asset_id=asset.id,
            user_file_id=evidence.id,
            parent_document_id=parent.id,
            role="annex",
            relationships=[],
            status="source_evidence",
        )
        session.add(member)
        package.asset_count = 3
        session.flush()
        source_assets = list(
            session.scalars(
                select(RegulatorySourceAsset).where(
                    RegulatorySourceAsset.package_id == package.id
                )
            )
        )
        hashes = {package.manifest_file_id: package.manifest_sha256}
        for item in source_assets:
            hashes[item.file_id] = item.sha256
            hashes[item.text_file_id] = item.text_sha256
        monkeypatch.setattr(
            document_intake_review, "_blob_sha256", lambda pointer: hashes[pointer]
        )
        decision = RegulatoryDocumentIntakeReviewDecision(
            id=uuid4(),
            intake_id=batch.id,
            document_id=member.id,
            sequence=1,
            disposition="source_evidence",
            duplicate_of_document_id=parent.id,
            source_snapshot={
                "source": document_intake_review._source_snapshot(
                    session, batch, member
                ),
                "duplicate_source": document_intake_review._source_snapshot(
                    session, batch, parent
                ),
                "canonical_sha256": canonical_scope_digest(session, evidence.id),
            },
            reason="Reviewed equivalent source evidence",
            reviewed_by=labeling_data.user_id,
        )
        if condition == "stale_snapshot":
            decision.source_snapshot["source"]["asset_sha256"] = "0" * 64
        if condition != "unreviewed":
            session.add(decision)
        if condition == "membership":
            session.add(
                DocumentSet__UserFile(
                    document_set_id=labeling_data.document_set_id,
                    user_file_id=evidence.id,
                )
            )
        session.flush()
        resolved = resolve_amendment_source_family(
            session,
            document_set_id=labeling_data.document_set_id,
            user_file_ids=[primary.id, child.id],
            matched_file_ids=[primary.id],
            appendix_label="EK I",
        )
        if condition != "valid":
            assert resolved is None
            return
        assert resolved is not None
        assert resolved.member_file_ids == tuple(sorted([primary.id, child.id]))
        assert resolved.target_file_ids == (child.id,)


@pytest.mark.parametrize(
    "change",
    [
        "unpublished",
        "gate",
        "manifest",
        "manifest_sha",
        "processing",
        "wrong_owner",
        "member_partial",
        "no_approval",
        "batch_partial",
        "wrong_DS",
        "missing_membership",
        "outside_scope",
        "unmatched_child_outside_scope",
        "wrong_link",
        "asymmetric_link",
        "body_link_only",
        "cycle",
        "second_root",
        "cross_batch",
        "foreign_asset",
        "source_evidence",
        "duplicate_version",
    ],
)
def test_family_never_waives_ambiguous_or_unapproved_sources(
    labeling_data: LabelingData, change: str
) -> None:
    with Session(labeling_data.database.engine) as session:
        primary, child, package, batch, parent, annex, publications = family(
            session, labeling_data
        )
        scope = [primary.id, child.id]
        matched = [primary.id, child.id]
        ds = labeling_data.document_set_id
        if change == "unpublished":
            publications[1].epoch = 0
        elif change == "gate":
            publications[1].gate_closed = True
        elif change == "manifest":
            publications[1].writer_manifest = {"pending": True}
        elif change == "manifest_sha":
            publications[1].writer_manifest_sha256 = "e" * 64
        elif change == "processing":
            child.status = UserFileStatus.CHUNKED
        elif change == "wrong_owner":
            child.user_id = None
        elif change == "member_partial":
            annex.status = "indexing"
        elif change == "no_approval":
            annex.approval_attempt_id = None
        elif change == "batch_partial":
            batch.status = "ready"
        elif change == "wrong_DS":
            ds = labeling_data.other_document_set_id
        elif change == "missing_membership":
            session.delete(
                session.get_one(
                    DocumentSet__UserFile, (labeling_data.document_set_id, child.id)
                )
            )
        elif change == "outside_scope":
            scope = [primary.id]
        elif change == "unmatched_child_outside_scope":
            scope = matched = [primary.id]
        elif change == "wrong_link":
            parent.relationships = [
                {**parent.relationships[0], "target_asset_hash": "f" * 64}
            ]
        elif change == "asymmetric_link":
            annex.relationships = []
        elif change == "body_link_only":
            parent.relationships = annex.relationships = []
        elif change == "cycle":
            parent.parent_document_id = annex.id
        elif change == "second_root":
            annex.parent_document_id = None
            annex.role = "root"
        elif change == "cross_batch":
            other_package = AmendmentSourcePackage(
                document_set_id=ds,
                created_by=labeling_data.user_id,
                environment="family-local-test",
                idempotency_key=str(uuid4()),
                request_hash="e" * 64,
                input_spec={},
                status="ready",
            )
            session.add(other_package)
            session.flush()
            other_batch = RegulatoryDocumentIntake(
                source_package_id=other_package.id,
                document_set_id=ds,
                created_by=labeling_data.user_id,
                idempotency_key=uuid4(),
                labeling_model_configuration_id=labeling_data.model_id,
                status="completed",
                stage="completed",
                review_sha256="e" * 64,
            )
            session.add(other_batch)
            session.flush()
            annex.intake_id = other_batch.id
        elif change == "foreign_asset":
            package.asset_count = 1
        elif change == "source_evidence":
            session.add(
                RegulatoryDocumentIntakeReviewDecision(
                    id=uuid4(),
                    intake_id=batch.id,
                    document_id=annex.id,
                    sequence=1,
                    disposition="source_evidence",
                    duplicate_of_document_id=parent.id,
                    source_snapshot={},
                    reason="Retained duplicate",
                    reviewed_by=labeling_data.user_id,
                )
            )
        elif change == "duplicate_version":
            duplicate = UserFile(
                id=uuid4(),
                user_id=labeling_data.user_id,
                file_id=str(uuid4()),
                name=primary.name,
                file_type=primary.file_type,
                status=UserFileStatus.COMPLETED,
            )
            session.add(duplicate)
            session.flush()
            session.add(
                DocumentSet__UserFile(document_set_id=ds, user_file_id=duplicate.id)
            )
            session.add(
                RegulatoryFilePublication(
                    user_file_id=duplicate.id,
                    scope_key="owned-local-family",
                    owner_id=uuid4(),
                    fencing_token=1,
                    lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
                    gate_closed=False,
                    epoch=1,
                    next_ordinal=1,
                )
            )
            matched.append(duplicate.id)
            scope = list(matched)
        session.flush()
        assert (
            resolve_amendment_source_family(
                session,
                document_set_id=ds,
                user_file_ids=scope,
                matched_file_ids=matched,
                appendix_label="EK I",
            )
            is None
        )


def test_two_real_appendix_files_remain_ambiguous(labeling_data: LabelingData) -> None:
    with Session(labeling_data.database.engine) as session:
        primary, child, *_ = family(session, labeling_data)
        chunk = session.get_one(RegulatoryChunk, labeling_data.first_id)
        chunk.heading_path = ["Own primary", "EK I"]
        chunk.chunk_metadata = {"appendix_label": "EK I"}
        session.flush()
        result = resolve_amendment_source_family(
            session,
            document_set_id=labeling_data.document_set_id,
            user_file_ids=[primary.id, child.id],
            matched_file_ids=[primary.id, child.id],
            appendix_label="EK I",
        )
        assert result is not None and set(result.target_file_ids) == {
            primary.id,
            child.id,
        }


@pytest.mark.parametrize("kind", ["url", "embedded", "internal"])
def test_production_retriever_cache_keeps_article_and_each_roman_appendix_separate(
    labeling_data: LabelingData,
    monkeypatch: pytest.MonkeyPatch,
    kind: Literal["url", "embedded", "internal"],
) -> None:
    with Session(labeling_data.database.engine) as session:
        primary, child, *_ = family(session, labeling_data, link_kind=kind)
        title = "Enerji Tüketimi Çevreye Duyarlı Tasarım Gereklilikleri Hakkında Tebliğ"
        primary.name = child.name = title
        main = session.get_one(RegulatoryChunk, labeling_data.first_id)
        main.heading_path = [title, "MADDE 9", "(1)"]
        main.chunk_metadata = {"article_no": "9", "paragraph_no": "1"}
        for row in session.scalars(
            select(RegulatoryChunk).where(RegulatoryChunk.user_file_id == child.id)
        ):
            row.heading_path = [title, *row.heading_path[1:]]
        session.flush()

        @contextmanager
        def read_session() -> Iterator[Session]:
            yield session

        monkeypatch.setattr(
            search_retriever, "get_session_with_current_tenant", read_session
        )
        monkeypatch.setattr(
            search_retriever,
            "get_current_search_settings",
            lambda _session: MagicMock(),
        )
        monkeypatch.setattr(
            search_retriever, "get_default_document_index", lambda *_args: MagicMock()
        )
        monkeypatch.setattr(
            search_retriever,
            "get_tools",
            lambda _session: [
                SimpleNamespace(id=1, in_code_tool_id=search_retriever.SEARCH_TOOL_ID)
            ],
        )
        search = Mock(
            side_effect=AssertionError(
                "Exact source targets must not call SearchTool/provider"
            )
        )
        monkeypatch.setattr(search_retriever, "SearchTool", search)
        family_spy = Mock(wraps=resolve_amendment_source_family)
        monkeypatch.setattr(
            search_retriever, "resolve_amendment_source_family", family_spy
        )
        retriever = search_retriever.build_amendment_search_retriever(
            session,
            document_set_id=labeling_data.document_set_id,
            created_by=labeling_data.user_id,
            user_file_ids=[primary.id, child.id],
            llm=MagicMock(),
        )
        for appendix in (None, "I", "II", "IV", None):
            operation = (
                "MADDE 1- Aynı Tebliğin 9 uncu maddesinin birinci fıkrasında yer alan “A” ibaresi “B” şeklinde değiştirilmiştir."
                if appendix is None
                else f"MADDE 2- Aynı Tebliğin Ek {appendix}'inde yer alan “A” ibaresi “B” şeklinde değiştirilmiştir."
            )
            instruction = AmendmentInstruction(
                instruction_text=operation, target_source=title
            )
            assert retriever._source_file_loader is not None
            assert retriever._structural_candidate_loader is not None
            assert retriever._source_file_loader(instruction) == [
                str(primary.id if appendix is None else child.id)
            ]
            candidates = retriever._structural_candidate_loader(instruction)
            assert candidates and {
                candidate.user_file_id for candidate in candidates
            } == {str(primary.id if appendix is None else child.id)}
            if appendix is None:
                assert candidates[0].chunk_id == main.id
            else:
                assert candidates[0].metadata["appendix_label"] == f"EK {appendix}"
        assert family_spy.call_count == 4
        assert [
            call.kwargs["appendix_label"] for call in family_spy.call_args_list
        ] == [None, "EK-I", "EK-II", "EK-IV"]
        search.assert_not_called()
