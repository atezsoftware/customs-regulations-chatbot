"""Approved standalone annexes retain their explicitly selected existing parent."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
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
from onyx.server.features.regulatory.catalog_api import _file_snapshots
from tests.external_dependency_unit.regulatory.test_amendment_source_family_resolution import (
    family,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    LabelingData,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_data as labeling_data,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)


def reviewed_pair(
    session: Session,
    data: LabelingData,
    monkeypatch: pytest.MonkeyPatch,
    *,
    appendix_label: str = "EK-1",
) -> tuple[
    UserFile,
    UserFile,
    RegulatoryDocumentIntakeFile,
    RegulatoryDocumentIntakeReviewDecision,
]:
    parent = session.get_one(UserFile, data.file_id)
    parent.status = UserFileStatus.COMPLETED
    child = UserFile(
        id=uuid4(),
        user_id=data.user_id,
        file_id=str(uuid4()),
        name="Başlık belirlenemedi",
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
    for file in (parent, child):
        existing = session.get(RegulatoryFilePublication, file.id)
        if existing is not None:
            existing.lease_expires_at = datetime.now(timezone.utc) - timedelta(
                minutes=1
            )
            continue
        session.add(
            RegulatoryFilePublication(
                user_file_id=file.id,
                scope_key="test-existing-parent",
                owner_id=uuid4(),
                fencing_token=1,
                lease_expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
                gate_closed=False,
                epoch=1,
                next_ordinal=4,
            )
        )
    package = AmendmentSourcePackage(
        id=uuid4(),
        document_set_id=data.document_set_id,
        created_by=data.user_id,
        environment="local-test",
        idempotency_key=str(uuid4()),
        request_hash="a" * 64,
        input_spec={},
        status="ready",
        asset_count=1,
        total_bytes=1,
        issues=[],
        manifest_file_id="manifest",
        manifest_sha256="b" * 64,
    )
    session.add(package)
    session.flush()
    asset = RegulatorySourceAsset(
        id=uuid4(),
        package_id=package.id,
        sha256="c" * 64,
        file_id=child.file_id,
        text_file_id="prepared",
        text_sha256="d" * 64,
        mime_type="application/pdf",
        display_name="standalone-annex.pdf",
        byte_count=1,
    )
    batch = RegulatoryDocumentIntake(
        id=uuid4(),
        source_package_id=package.id,
        document_set_id=data.document_set_id,
        created_by=data.user_id,
        idempotency_key=uuid4(),
        labeling_model_configuration_id=data.model_id,
        status="completed",
        stage="completed",
        review_sha256="e" * 64,
    )
    session.add_all([asset, batch])
    session.flush()
    member = RegulatoryDocumentIntakeFile(
        id=uuid4(),
        intake_id=batch.id,
        source_asset_id=asset.id,
        user_file_id=child.id,
        parent_document_id=None,
        role="root",
        relationships=[],
        status="completed",
        approval_attempt_id=uuid4(),
    )
    session.add(member)
    session.add(
        RegulatoryChunk(
            id="rc_existing_annex_" + uuid4().hex,
            user_file_id=child.id,
            position=0,
            projection_ordinal=0,
            text=f"Actual {appendix_label} annex body",
            heading_path=["Own annex", appendix_label],
            chunk_type="paragraph",
            chunk_metadata={"appendix_label": appendix_label},
            source="indexed",
            status="active",
        )
    )
    session.flush()
    hashes = {"manifest": "b" * 64, child.file_id: "c" * 64, "prepared": "d" * 64}
    monkeypatch.setattr(
        document_intake_review, "_blob_sha256", lambda pointer: hashes[pointer]
    )
    decision = RegulatoryDocumentIntakeReviewDecision(
        id=uuid4(),
        intake_id=batch.id,
        document_id=member.id,
        sequence=1,
        disposition="legislation",
        existing_parent_file_id=parent.id,
        existing_parent_snapshot=document_intake_review._parent_snapshot(
            session, batch, member, parent.id
        ),
        source_snapshot={
            "source": document_intake_review._source_snapshot(session, batch, member),
            "duplicate_source": None,
            "canonical_sha256": canonical_scope_digest(session, child.id),
        },
        reason="User reviewed the existing parent",
        reviewed_by=data.user_id,
    )
    session.add(decision)
    session.flush()
    return parent, child, member, decision


def test_catalog_and_updates_follow_reviewed_existing_parent(
    labeling_data: LabelingData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(labeling_data.database.engine) as session:
        parent, child, _member, _decision = reviewed_pair(
            session, labeling_data, monkeypatch
        )
        views = _file_snapshots(session, [parent, child])
        parents = views[parent.id].reviewed_relations
        children = views[child.id].reviewed_relations
        assert parents is not None and children is not None
        assert parents.attachments[0].user_file_id == child.id
        assert children.parents[0].user_file_id == parent.id
        resolved = resolve_amendment_source_family(
            session,
            document_set_id=labeling_data.document_set_id,
            user_file_ids=[parent.id, child.id],
            matched_file_ids=[parent.id],
            appendix_label="EK-1",
        )
        assert resolved is not None and resolved.target_file_ids == (child.id,)


@pytest.mark.parametrize(
    "change",
    [
        "source_pointer",
        "parent_pointer",
        "parent_membership",
        "child_owner",
        "outside_scope",
        "unpublished_child",
        "closed_parent_gate",
    ],
)
def test_reviewed_relation_never_overrides_current_source_or_scope(
    labeling_data: LabelingData,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    with Session(labeling_data.database.engine) as session:
        parent, child, member, _decision = reviewed_pair(
            session, labeling_data, monkeypatch
        )
        scope = [parent.id, child.id]
        if change == "source_pointer":
            child.file_id = str(uuid4())
        elif change == "parent_pointer":
            parent.file_id = str(uuid4())
        elif change == "parent_membership":
            session.delete(
                session.get_one(
                    DocumentSet__UserFile, (labeling_data.document_set_id, parent.id)
                )
            )
        elif change == "child_owner":
            child.user_id = None
        elif change == "outside_scope":
            scope = [parent.id]
        elif change == "unpublished_child":
            session.get_one(RegulatoryFilePublication, child.id).epoch = 0
        elif change == "closed_parent_gate":
            session.get_one(RegulatoryFilePublication, parent.id).gate_closed = True
        session.flush()
        assert (
            resolve_amendment_source_family(
                session,
                document_set_id=labeling_data.document_set_id,
                user_file_ids=scope,
                matched_file_ids=[parent.id],
                appendix_label="EK-1",
            )
            is None
        )


def test_legal_updates_and_title_edits_preserve_existing_parent_identity(
    labeling_data: LabelingData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(labeling_data.database.engine) as session:
        parent, child, *_ = reviewed_pair(session, labeling_data, monkeypatch)
        parent.name = "New legal title"
        session.get_one(
            RegulatoryChunk, labeling_data.first_id
        ).text = "Lawfully updated provision"
        session.flush()
        resolved = resolve_amendment_source_family(
            session,
            document_set_id=labeling_data.document_set_id,
            user_file_ids=[parent.id, child.id],
            matched_file_ids=[parent.id],
            appendix_label="EK-1",
        )
        assert resolved is not None and resolved.target_file_ids == (child.id,)


@pytest.mark.parametrize("include_external", [True, False])
def test_later_annex_never_hides_an_existing_intake_annex(
    labeling_data: LabelingData,
    monkeypatch: pytest.MonkeyPatch,
    include_external: bool,
) -> None:
    with Session(labeling_data.database.engine) as session:
        primary, original, *_ = family(session, labeling_data)
        parent, later, *_ = reviewed_pair(
            session, labeling_data, monkeypatch, appendix_label="EK-5"
        )
        assert primary.id == parent.id
        scope = [parent.id, original.id] + ([later.id] if include_external else [])
        resolved = resolve_amendment_source_family(
            session,
            document_set_id=labeling_data.document_set_id,
            user_file_ids=scope,
            matched_file_ids=[parent.id],
            appendix_label="EK-I",
        )
        assert resolved is not None and resolved.target_file_ids == (original.id,)
        if include_external:
            resolved = resolve_amendment_source_family(
                session,
                document_set_id=labeling_data.document_set_id,
                user_file_ids=scope,
                matched_file_ids=[parent.id],
                appendix_label="EK-5",
            )
            assert resolved is not None and resolved.target_file_ids == (later.id,)


def test_reviewed_attachment_keeps_its_own_source_linked_annexes(
    labeling_data: LabelingData,
) -> None:
    from onyx.db.document_intake_source_reference import intake_source_reference

    with Session(labeling_data.database.engine) as session:
        attached, nested, package, batch, member, _, publications = family(
            session, labeling_data
        )
        parent = UserFile(
            id=uuid4(),
            user_id=labeling_data.user_id,
            file_id=str(uuid4()),
            name="Existing legacy parent",
            file_type="application/pdf",
            status=UserFileStatus.COMPLETED,
        )
        session.add(parent)
        session.flush()
        session.add(
            DocumentSet__UserFile(
                document_set_id=labeling_data.document_set_id, user_file_id=parent.id
            )
        )
        session.add(
            RegulatoryFilePublication(
                user_file_id=parent.id,
                scope_key="legacy-parent",
                owner_id=uuid4(),
                fencing_token=1,
                lease_expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
                gate_closed=False,
                epoch=1,
                next_ordinal=0,
            )
        )
        for publication in publications:
            publication.lease_expires_at = datetime.now(timezone.utc) - timedelta(
                minutes=1
            )
        asset = session.get_one(RegulatorySourceAsset, member.source_asset_id)
        session.add(
            RegulatoryDocumentIntakeReviewDecision(
                id=uuid4(),
                intake_id=batch.id,
                document_id=member.id,
                sequence=1,
                disposition="legislation",
                existing_parent_file_id=parent.id,
                existing_parent_snapshot={
                    "user_file_id": str(parent.id),
                    "file_id": parent.file_id,
                    "document_set_id": labeling_data.document_set_id,
                    "source_url": parent.link_url,
                },
                source_snapshot={
                    "source": intake_source_reference(package, member, asset, attached)
                },
                reason="Reviewed existing parent",
                reviewed_by=labeling_data.user_id,
            )
        )
        session.flush()
        result = resolve_amendment_source_family(
            session,
            document_set_id=labeling_data.document_set_id,
            user_file_ids=[parent.id, attached.id, nested.id],
            matched_file_ids=[parent.id],
            appendix_label="EK-I",
        )
        assert result is not None
        assert set(result.member_file_ids) == {parent.id, attached.id, nested.id}
        assert result.target_file_ids == (nested.id,)


def test_existing_intake_annex_can_receive_a_later_reviewed_attachment(
    labeling_data: LabelingData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(labeling_data.database.engine) as session:
        parent, original, *_ = family(session, labeling_data)
        _, later, *_ = reviewed_pair(
            session,
            labeling_data._replace(file_id=original.id),
            monkeypatch,
            appendix_label="EK-5",
        )
        result = resolve_amendment_source_family(
            session,
            document_set_id=labeling_data.document_set_id,
            user_file_ids=[parent.id, original.id, later.id],
            matched_file_ids=[parent.id],
            appendix_label="EK-5",
        )
        assert result is not None and result.target_file_ids == (later.id,)
