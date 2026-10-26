"""New-document admission uses real PostgreSQL while preserving existing corpus rows."""

import json
from collections.abc import Generator
from hashlib import sha256
from io import BytesIO
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.auth.schemas import UserRole
from onyx.configs.constants import FileOrigin
from onyx.db import document_intake as intake
from onyx.db.engine.sql_engine import get_sqlalchemy_engine
from onyx.db.enums import UserFileProjectionRepairStatus, UserFileStatus
from onyx.db.models import (
    AmendmentBatch,
    AmendmentSourcePackage,
    DocumentSet,
    DocumentSet__UserFile,
    LLMProvider,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryLabelTaxonomy,
    RegulatorySourceAsset,
    User,
    UserFile,
    UserFileProjectionRepair,
)
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from onyx.regulatory.amendments.annexes.models import SourceLink
from tests.external_dependency_unit.conftest import create_test_user

_ENVIRONMENT = "document-intake-regression"


def _existing_rows(session: Session, file_ids: list[UUID]) -> str:
    rows: dict[str, Any] = {}
    for model, key in (
        (UserFile, UserFile.id),
        (RegulatoryChunk, RegulatoryChunk.user_file_id),
        (DocumentSet__UserFile, DocumentSet__UserFile.user_file_id),
        (UserFileProjectionRepair, UserFileProjectionRepair.user_file_id),
        (RegulatoryLabelingItem, RegulatoryLabelingItem.user_file_id),
    ):
        records = session.execute(select(model.__table__).where(key.in_(file_ids)))
        rows[model.__tablename__] = sorted(
            (dict(row) for row in records.mappings()), key=lambda row: str(row)
        )
    return json.dumps(rows, sort_keys=True, default=str)


def _repair_file_ids(session: Session) -> set[UUID]:
    return set(session.scalars(select(UserFileProjectionRepair.user_file_id)))


@pytest.fixture
def intake_session(
    db_session: Session,  # noqa: ARG001
    tenant_context: None,  # noqa: ARG001
) -> Generator[Session, None, None]:
    with get_sqlalchemy_engine().connect() as connection:
        transaction = connection.begin()
        try:
            with Session(
                connection, join_transaction_mode="create_savepoint"
            ) as session:
                existing = list(session.scalars(select(UserFile.id)))
                before = _existing_rows(session, existing)
                yield session
                session.flush()
                assert _existing_rows(session, existing) == before
        finally:
            transaction.rollback()


@pytest.fixture
def intake_owner(
    intake_session: Session, monkeypatch: pytest.MonkeyPatch
) -> tuple[User, DocumentSet, ModelConfiguration]:
    owner = create_test_user(intake_session, "intake-owner", UserRole.ADMIN)
    target = intake_session.scalar(
        select(DocumentSet).where(DocumentSet.name == "PC Külliyatı")
    )
    if target is None:
        target = DocumentSet(
            name="PC Külliyatı", user_id=owner.id, is_up_to_date=True, is_public=True
        )
        intake_session.add(target)
    provider = LLMProvider(name=f"intake-test-{uuid4()}", provider="vertex_ai")
    model = ModelConfiguration(
        llm_provider=provider, name="labeling-fixture", is_visible=True
    )
    intake_session.add(model)
    intake_session.flush()
    monkeypatch.setattr(
        intake,
        "resolve_intake_labeling_provider_binding",
        lambda *_args, **_kwargs: None,
    )
    return owner, target, model


def _package(
    session: Session, owner: User, target: DocumentSet
) -> tuple[AmendmentSourcePackage, list[RegulatorySourceAsset], list[SourceLink]]:
    package = AmendmentSourcePackage(
        document_set_id=target.id,
        created_by=owner.id,
        environment=_ENVIRONMENT,
        idempotency_key=str(uuid4()),
        request_hash=uuid4().hex * 2,
        input_spec={"url": "https://example.gov/new-law.html"},
        status="processing",
    )
    session.add(package)
    session.flush()
    store = PostgresBackedFileStore()
    assets = []
    for name, body in (
        ("new-law.html", "MADDE 1 — Ekler için tıklayınız."),
        ("EK-1.html", "EK-1 — İthalatta ibraz edilecek belgeler."),
    ):
        encoded = body.encode()
        digest = sha256(encoded).hexdigest()
        file_id = store.save_file(
            BytesIO(encoded),
            name,
            FileOrigin.USER_FILE,
            "text/html",
            db_session=session,
        )
        assets.append(
            RegulatorySourceAsset(
                package_id=package.id,
                sha256=digest,
                file_id=file_id,
                text_file_id=file_id,
                text_sha256=digest,
                mime_type="text/html",
                display_name=name,
                byte_count=len(encoded),
                original_url=f"https://example.gov/{name}",
                final_url=f"https://example.gov/{name}",
            )
        )
    session.add_all(assets)
    session.flush()
    links = [
        SourceLink(
            parent_asset_hash=assets[0].sha256,
            target_asset_hash=assets[1].sha256,
            parent_url=assets[0].original_url,
            requested_url=assets[1].original_url,
            source_field="href",
            label="Ekler için tıklayınız",
            kind="url",
        )
    ]
    manifest = json.dumps(
        {
            "status": "ready",
            "assets": [{"sha256": a.sha256} for a in assets],
            "links": [link.model_dump(mode="json") for link in links],
            "issues": [],
        }
    ).encode()
    package.manifest_file_id = store.save_file(
        BytesIO(manifest),
        "source-manifest.json",
        FileOrigin.OTHER,
        "application/json",
        db_session=session,
    )
    package.manifest_sha256 = sha256(manifest).hexdigest()
    package.asset_count = len(assets)
    package.total_bytes = sum(asset.byte_count for asset in assets)
    package.status = "ready"
    session.flush()
    return package, assets, links


def _admitted(
    session: Session, owner: User, target: DocumentSet, model: ModelConfiguration
) -> tuple[RegulatoryDocumentIntake, list[RegulatoryDocumentIntakeFile]]:
    package, assets, links = _package(session, owner, target)
    batch, created = intake.create_intake(
        session,
        source_package_id=package.id,
        idempotency_key=uuid4(),
        labeling_model_configuration_id=model.id,
        user=owner,
        environment=_ENVIRONMENT,
    )
    assert created
    members = intake.admit_intake_assets(
        session, batch, assets, links, root_hash=assets[0].sha256
    )
    for member in members:
        session.add(
            RegulatoryChunk(
                id=f"rc_intake_{uuid4().hex}",
                user_file_id=member.user_file_id,
                text=f"New source {member.role}",
                position=0,
                projection_ordinal=0,
                heading_path=[member.role],
                chunk_metadata={"source_asset_id": str(member.source_asset_id)},
                chunk_type="article",
                status="active",
                source="indexed",
            )
        )
    session.flush()
    return batch, members


def _completed_labels(
    session: Session,
    batch: RegulatoryDocumentIntake,
    members: list[RegulatoryDocumentIntakeFile],
) -> RegulatoryLabelingRun:
    taxonomy = RegulatoryLabelTaxonomy(
        name=f"intake-taxonomy-{uuid4()}",
        version_hash=uuid4().hex * 2,
        definition={"labels": ["Gümrük"]},
        label_count=1,
    )
    session.add(taxonomy)
    session.flush()
    run = RegulatoryLabelingRun(
        document_set_id=batch.document_set_id,
        taxonomy_id=taxonomy.id,
        model_configuration_id=batch.labeling_model_configuration_id,
        idempotency_key=uuid4(),
        model="labeling-fixture",
        provider_binding={},
        file_ids=[str(member.user_file_id) for member in members],
        snapshot_hash=uuid4().hex * 2,
        status="completed",
        stage="finished",
        total_chunks=len(members),
        completed_chunks=len(members),
    )
    session.add(run)
    session.flush()
    for chunk in session.scalars(
        select(RegulatoryChunk).where(
            RegulatoryChunk.user_file_id.in_(
                [member.user_file_id for member in members]
            )
        )
    ):
        session.add(
            RegulatoryLabelingItem(
                run_id=run.id,
                regulatory_chunk_id=chunk.id,
                user_file_id=chunk.user_file_id,
                text_snapshot=chunk.text,
                source_snapshot={},
                status="completed",
                labels=["Gümrük"],
            )
        )
    batch.labeling_run_id, batch.stage = run.id, "labeling"
    session.flush()
    return run


def test_admission_is_idempotent_separate_files_preserve_parent_graph_and_existing_corpus(
    intake_session: Session, intake_owner: tuple[User, DocumentSet, ModelConfiguration]
) -> None:
    owner, target, model = intake_owner
    package, assets, links = _package(intake_session, owner, target)
    key = uuid4()
    amendment_ids = list(intake_session.scalars(select(AmendmentBatch.id)))
    first, created = intake.create_intake(
        intake_session,
        source_package_id=package.id,
        idempotency_key=key,
        labeling_model_configuration_id=model.id,
        user=owner,
        environment=_ENVIRONMENT,
    )
    again, created_again = intake.create_intake(
        intake_session,
        source_package_id=package.id,
        idempotency_key=key,
        labeling_model_configuration_id=model.id,
        user=owner,
        environment=_ENVIRONMENT,
    )
    assert created and not created_again and first.id == again.id
    members = intake.admit_intake_assets(
        intake_session, first, assets, links, root_hash=assets[0].sha256
    )
    repeated = intake.admit_intake_assets(
        intake_session, first, assets, links, root_hash=assets[0].sha256
    )
    assert {m.id for m in members} == {m.id for m in repeated}
    root = next(member for member in members if member.role == "root")
    annex = next(member for member in members if member.role == "annex")
    assert root.user_file_id != annex.user_file_id
    assert annex.parent_document_id == root.id
    assert annex.relationships[0]["label"] == "Ekler için tıklayınız"
    ids = {member.user_file_id for member in members}
    assert (
        set(
            intake_session.scalars(
                select(DocumentSet__UserFile.user_file_id).where(
                    DocumentSet__UserFile.document_set_id == target.id,
                    DocumentSet__UserFile.user_file_id.in_(ids),
                )
            )
        )
        == ids
    )
    assert list(intake_session.scalars(select(AmendmentBatch.id))) == amendment_ids


@pytest.mark.parametrize("mismatch", ["target", "owner", "environment", "incomplete"])
def test_intake_rejects_wrong_or_incomplete_source_scope(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    mismatch: str,
) -> None:
    owner, target, model = intake_owner
    package, _, _ = _package(intake_session, owner, target)
    if mismatch == "target":
        other = DocumentSet(name=f"other-{uuid4()}", is_up_to_date=True)
        intake_session.add(other)
        intake_session.flush()
        package.document_set_id = other.id
    elif mismatch == "owner":
        package.created_by = create_test_user(intake_session, "other-intake-owner").id
    elif mismatch == "environment":
        package.environment = "another-environment"
    else:
        package.status, package.issues = "partial", [{"code": "download_failed"}]
    intake_session.flush()
    with pytest.raises(ValueError):
        intake.create_intake(
            intake_session,
            source_package_id=package.id,
            idempotency_key=uuid4(),
            labeling_model_configuration_id=model.id,
            user=owner,
            environment=_ENVIRONMENT,
        )


@pytest.mark.parametrize("mismatch", ["foreign_asset", "missing_root", "dangling_link"])
def test_first_admission_rejects_source_graph_outside_the_frozen_package(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    mismatch: str,
) -> None:
    owner, target, model = intake_owner
    package, assets, links = _package(intake_session, owner, target)
    batch, _ = intake.create_intake(
        intake_session,
        source_package_id=package.id,
        idempotency_key=uuid4(),
        labeling_model_configuration_id=model.id,
        user=owner,
        environment=_ENVIRONMENT,
    )
    root_hash = assets[0].sha256
    if mismatch == "foreign_asset":
        _, foreign, _ = _package(intake_session, owner, target)
        assets = [assets[0], foreign[1]]
    elif mismatch == "missing_root":
        root_hash = "0" * 64
    else:
        links = [links[0].model_copy(update={"parent_asset_hash": "0" * 64})]
    with pytest.raises((ValueError, intake.IntakeConflictError)):
        intake.admit_intake_assets(
            intake_session, batch, assets, links, root_hash=root_hash
        )
    assert intake.intake_files(intake_session, batch) == []


def test_approve_all_indexes_only_new_labeled_files_and_is_idempotent(
    intake_session: Session, intake_owner: tuple[User, DocumentSet, ModelConfiguration]
) -> None:
    owner, target, model = intake_owner
    batch, members = _admitted(intake_session, owner, target, model)
    run = _completed_labels(intake_session, batch, members)
    intake.refresh_intake(intake_session, batch)
    assert batch.status == "ready" and batch.review_sha256
    expected = {member.user_file_id for member in members}
    assert set(run.file_ids) == {str(identifier) for identifier in expected}
    snapshot = intake.intake_snapshot(intake_session, batch)
    assert len(snapshot["documents"]) == 2
    assert all(
        document["labels"]["labels"] == ["Gümrük"] for document in snapshot["documents"]
    )
    previous_repairs = _repair_file_ids(intake_session)
    assert intake.pending_intake_file_ids(intake_session, list(expected)) == expected
    deliveries = intake.approve_intake(intake_session, batch, batch.review_sha256)
    assert {identifier for identifier, _ in deliveries} == expected
    assert intake.approve_intake(intake_session, batch, batch.review_sha256) == []
    assert _repair_file_ids(intake_session) == previous_repairs | expected
    repairs = list(
        intake_session.scalars(
            select(UserFileProjectionRepair).where(
                UserFileProjectionRepair.user_file_id.in_(expected)
            )
        )
    )
    assert {repair.user_file_id for repair in repairs} == expected
    assert all(repair.initial_index for repair in repairs)
    assert intake.pending_intake_file_ids(intake_session, list(expected)) == set()
    assert (
        intake.pending_intake_file_ids(
            intake_session, list(expected), include_approving=True
        )
        == expected
    )


@pytest.mark.parametrize(
    "change", ["text", "relationship", "label", "membership", "role", "parent"]
)
def test_review_changes_block_approval_before_any_index_request(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    change: str,
) -> None:
    owner, target, model = intake_owner
    batch, members = _admitted(intake_session, owner, target, model)
    _completed_labels(intake_session, batch, members)
    intake.refresh_intake(intake_session, batch)
    reviewed = batch.review_sha256
    assert reviewed
    previous_repairs = _repair_file_ids(intake_session)
    if change == "text":
        chunk = intake_session.scalar(
            select(RegulatoryChunk).where(
                RegulatoryChunk.user_file_id == members[0].user_file_id
            )
        )
        assert chunk
        chunk.text += " changed after review"
    elif change == "relationship":
        members[0].relationships = [{**members[0].relationships[0], "label": "changed"}]
    elif change == "label":
        item = intake_session.scalar(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == batch.labeling_run_id
            )
        )
        assert item
        item.labels = ["Changed label"]
    elif change == "role":
        members[0].role = "changed-role"
    elif change == "parent":
        annex = next(member for member in members if member.role == "annex")
        annex.parent_document_id = None
    else:
        membership = intake_session.get_one(
            DocumentSet__UserFile, (target.id, members[0].user_file_id)
        )
        intake_session.delete(membership)
    intake_session.flush()
    with pytest.raises(intake.IntakeConflictError):
        intake.approve_intake(intake_session, batch, reviewed)
    assert _repair_file_ids(intake_session) == previous_repairs


def test_incomplete_label_scope_stays_partial_and_other_owner_cannot_read_intake(
    intake_session: Session, intake_owner: tuple[User, DocumentSet, ModelConfiguration]
) -> None:
    owner, target, model = intake_owner
    batch, members = _admitted(intake_session, owner, target, model)
    run = _completed_labels(intake_session, batch, members)
    run.file_ids = [str(members[0].user_file_id)]
    intake_session.flush()
    intake.refresh_intake(intake_session, batch)
    assert batch.status == "partial" and batch.review_sha256 is None
    other = create_test_user(intake_session, "different-review-owner", UserRole.ADMIN)
    with pytest.raises(ValueError, match="not found"):
        intake.get_intake(intake_session, batch.id, other)


@pytest.mark.parametrize("pending_lost_delivery", [False, True])
def test_index_retry_preserves_completed_file_and_reuses_pending_delivery_token(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    pending_lost_delivery: bool,
) -> None:
    owner, target, model = intake_owner
    batch, members = _admitted(intake_session, owner, target, model)
    _completed_labels(intake_session, batch, members)
    intake.refresh_intake(intake_session, batch)
    reviewed = batch.review_sha256
    assert reviewed
    previous_repairs = _repair_file_ids(intake_session)
    original = dict(intake.approve_intake(intake_session, batch, reviewed))
    completed, retrying = members
    intake_session.get_one(
        UserFile, completed.user_file_id
    ).status = UserFileStatus.COMPLETED
    completed.status = "completed"
    completed_repair = intake_session.get_one(
        UserFileProjectionRepair, completed.user_file_id
    )
    completed_repair.status = UserFileProjectionRepairStatus.SUCCEEDED
    retry_repair = intake_session.get_one(
        UserFileProjectionRepair, retrying.user_file_id
    )
    if not pending_lost_delivery:
        retry_repair.status = UserFileProjectionRepairStatus.FAILED
    batch.status, batch.stage = "failed", "indexing"
    retrying.status, retrying.error = "failed", "Lost or failed initial delivery"
    intake_session.flush()
    completed_before = _existing_rows(intake_session, [completed.user_file_id])
    deliveries = intake.retry_intake(intake_session, batch)
    assert len(deliveries) == 1 and deliveries[0][0] == retrying.user_file_id
    if pending_lost_delivery:
        assert deliveries[0][1] == original[retrying.user_file_id]
    else:
        assert deliveries[0][1] != original[retrying.user_file_id]
    assert _existing_rows(intake_session, [completed.user_file_id]) == completed_before
    assert completed_repair.attempt_id == original[completed.user_file_id]
    assert _repair_file_ids(intake_session) == previous_repairs | {
        member.user_file_id for member in members
    }
    assert batch.status == "approving" and retrying.status == "indexing"
    assert batch.review_sha256 == reviewed
