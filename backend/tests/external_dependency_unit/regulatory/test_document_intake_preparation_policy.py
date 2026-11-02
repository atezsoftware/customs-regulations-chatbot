"""Frozen preparation policies against PostgreSQL, without provider dispatch."""

import importlib.util
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from uuid import uuid4

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import inspect, select, text
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import document_intake_edit_budget as edit_budget
from onyx.db import document_intake_reprepare as reprepare
from onyx.db.enums import IndexModelStatus
from onyx.db.models import (
    CloudEmbeddingProvider,
    DocumentSet,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryFilePublication,
    RegulatorySourceAsset,
    SearchSettings,
    User,
    UserFile,
)
from onyx.db.regulatory_writer_publication import canonical_scope_digest
from onyx.file_processing.original_ingestion import OriginalIngestionReceipt
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from onyx.regulatory.amendments.annexes.models import SourceLink
from onyx.regulatory.document_intake.models import (
    IntakePreparationPolicy,
    IntakeSourceAsset,
)
from onyx.regulatory.document_intake.preparation import load_verified_intake_documents
from onyx.regulatory.document_intake.versions import reprepare_target_versions
from onyx.regulatory.indexing_jobs.configuration import (
    compute_regulatory_chunk_generation_hash,
)
from shared_configs.enums import EmbeddingProvider
from tests.external_dependency_unit.regulatory.test_document_intake import (
    _ENVIRONMENT,
    _admitted,
    _package,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_owner as intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_session as intake_session,
)


@pytest.fixture
def google_settings(intake_session: Session) -> SearchSettings:
    provider = intake_session.get(CloudEmbeddingProvider, EmbeddingProvider.GOOGLE)
    if provider is None:
        provider = CloudEmbeddingProvider(provider_type=EmbeddingProvider.GOOGLE)
        intake_session.add(provider)
    settings = SearchSettings(
        model_name="gemini-embedding-2",
        model_dim=1024,
        normalize=False,
        status=IndexModelStatus.PRESENT,
        index_name=f"policy-test-{uuid4().hex}",
        cloud_provider=provider,
        enable_contextual_rag=True,
        multipass_indexing=False,
    )
    intake_session.add(settings)
    intake_session.flush()
    return settings


def _create(
    session: Session, owner: tuple[User, DocumentSet, ModelConfiguration]
) -> RegulatoryDocumentIntake:
    user, target, model = owner
    package, _, _ = _package(session, user, target)
    batch, created = intake.create_intake(
        session,
        source_package_id=package.id,
        idempotency_key=uuid4(),
        labeling_model_configuration_id=model.id,
        user=user,
        environment=_ENVIRONMENT,
    )
    assert created
    return batch


def test_new_policy_is_persisted_and_idempotent_retry_does_not_reselect(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    google_settings: SearchSettings,
) -> None:
    batch = _create(intake_session, intake_owner)
    policy = deepcopy(batch.preparation_policy)
    assert policy == IntakePreparationPolicy().model_dump(mode="json")
    identity = batch.id, batch.source_package_id, batch.idempotency_key
    google_settings.model_name = "gemini-embedding-001"
    intake_session.flush()
    again, created = intake.create_intake(
        intake_session,
        source_package_id=batch.source_package_id,
        idempotency_key=batch.idempotency_key,
        labeling_model_configuration_id=batch.labeling_model_configuration_id,
        user=intake_owner[0],
        environment=_ENVIRONMENT,
    )
    intake_session.expire(again)
    assert not created
    assert (again.id, again.source_package_id, again.idempotency_key) == identity
    assert again.preparation_policy == policy


def test_policy_null_admission_stays_legacy_after_model_change_and_retry(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    google_settings: SearchSettings,
) -> None:
    google_settings.model_name = "gemini-embedding-001"
    intake_session.flush()
    batch = _create(intake_session, intake_owner)
    assert batch.preparation_policy is None
    identity = batch.id, batch.source_package_id
    batch.status, batch.stage = "failed", "metadata"
    google_settings.model_name = "gemini-embedding-2"
    intake_session.flush()
    assert intake.retry_intake(intake_session, batch) == []
    intake_session.expire(batch)
    assert batch.preparation_policy is None
    assert (batch.id, batch.source_package_id) == identity


def _failed_legacy(
    session: Session, owner: tuple[User, DocumentSet, ModelConfiguration]
) -> RegulatoryDocumentIntake:
    batch, members = _admitted(session, *owner)
    batch.preparation_policy = None
    batch.status, batch.stage = "failed", "metadata"
    for member in members:
        file = session.get_one(UserFile, member.user_file_id)
        generation = compute_regulatory_chunk_generation_hash(
            embedding_provider=EmbeddingProvider.GOOGLE,
            embedding_model_name="gemini-embedding-2",
            enable_contextual_rag=True,
            intake_chunking_version="3",
        )
        file.regulatory_chunk_generation_hash = generation
        row = session.scalar(
            select(RegulatoryChunk).where(RegulatoryChunk.user_file_id == file.id)
        )
        assert row is not None
        row.chunk_metadata = {
            **row.chunk_metadata,
            "intake_chunking_version": "3",
            "chunk_variant": "atomic",
        }
        session.flush()
        asset = session.get_one(RegulatorySourceAsset, member.source_asset_id)
        session.add(
            RegulatoryFilePublication(
                user_file_id=file.id,
                scope_key="isolated-policy-test",
                owner_id=uuid4(),
                fencing_token=1,
                lease_expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
                original_ingestion_receipt=OriginalIngestionReceipt(
                    file_id=file.file_id,
                    raw_sha256=asset.sha256,
                    documents_sha256="a" * 64,
                    plaintext_sha256="b" * 64,
                    canonical_sha256=canonical_scope_digest(session, file.id),
                    generation_hash=generation,
                ).model_dump(mode="json"),
            )
        )
    session.flush()
    return batch


def test_explicit_reprepare_can_upgrade_without_changing_legacy_batch_or_source(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    google_settings: SearchSettings,
) -> None:
    session = intake_session
    batch = _failed_legacy(session, intake_owner)
    members = intake.intake_files(session, batch)
    identities = [(m.id, m.user_file_id, m.source_asset_id) for m in members]
    raw_hashes = {
        m.source_asset_id: session.get_one(
            RegulatorySourceAsset, m.source_asset_id
        ).sha256
        for m in members
    }
    canonical = {
        m.user_file_id: canonical_scope_digest(session, m.user_file_id) for m in members
    }
    preview = reprepare.reprepare_snapshot(session, batch)
    assert preview["target_preparation_policy"] == IntakePreparationPolicy().model_dump(
        mode="json"
    )
    google_settings.model_name = "gemini-embedding-001"
    session.flush()
    with pytest.raises(intake.IntakeConflictError, match="snapshot changed"):
        reprepare.request_reprepare(
            session,
            batch,
            expected_sha256=preview["expected_sha256"],
            idempotency_key=uuid4(),
        )
    google_settings.model_name = "gemini-embedding-2"
    session.flush()
    request = reprepare.request_reprepare(
        session,
        batch,
        expected_sha256=preview["expected_sha256"],
        idempotency_key=uuid4(),
    )
    assert all(
        reprepare_target_versions(entry) == ("4", 4)
        for entry in request.documents.values()
    )
    assert batch.preparation_policy is None
    assert [
        (m.id, m.user_file_id, m.source_asset_id)
        for m in intake.intake_files(session, batch)
    ] == identities
    assert {
        m.source_asset_id: session.get_one(
            RegulatorySourceAsset, m.source_asset_id
        ).sha256
        for m in members
    } == raw_hashes
    assert {
        m.user_file_id: canonical_scope_digest(session, m.user_file_id) for m in members
    } == canonical


def test_already_queued_reprepare_keeps_its_old_target_after_model_change(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    google_settings: SearchSettings,
) -> None:
    session = intake_session
    batch = _failed_legacy(session, intake_owner)
    google_settings.model_name = "gemini-embedding-001"
    session.flush()
    preview = reprepare.reprepare_snapshot(session, batch)
    request = reprepare.request_reprepare(
        session,
        batch,
        expected_sha256=preview["expected_sha256"],
        idempotency_key=uuid4(),
    )
    frozen = deepcopy(request.documents)
    google_settings.model_name = "gemini-embedding-2"
    batch.status = "failed"
    session.flush()
    assert intake.retry_intake(session, batch) == []
    session.expire(request)
    assert request.documents == frozen
    assert all(
        reprepare_target_versions(entry) == ("3", 4)
        for entry in request.documents.values()
    )


def test_verified_source_context_and_prospective_edit_use_frozen_policy(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    google_settings: SearchSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = intake_session
    batch, members = _admitted(session, *intake_owner)
    policy = IntakePreparationPolicy.model_validate(batch.preparation_policy)
    member = next(m for m in members if m.role == "root")
    asset = session.get_one(RegulatorySourceAsset, member.source_asset_id)
    store = PostgresBackedFileStore()
    monkeypatch.setattr(
        store, "read_file", partial(store.read_file, db_session=session)
    )
    documents = load_verified_intake_documents(
        package_id=batch.source_package_id,
        asset=IntakeSourceAsset.model_validate(asset),
        user_file_id=member.user_file_id,
        links=[SourceLink.model_validate(x) for x in member.relationships],
        store=store,
        chunking_version="4",
        preparation_policy=policy,
    )
    assert documents.original_extraction.raw_sha256 == asset.sha256
    assert documents[0].doc_metadata is not None
    assert documents[0].doc_metadata["regulatory_document_intake"][
        "preparation_policy"
    ] == policy.model_dump(mode="json")
    row = session.scalar(
        select(RegulatoryChunk).where(
            RegulatoryChunk.user_file_id == member.user_file_id
        )
    )
    assert row is not None
    row.text = "| 0101 | " + "menşeli " * 300 + "| Ortak %50 sınırı |"
    row.chunk_type = "table"
    row.chunk_metadata = {
        "intake_chunking_version": "4",
        "intake_preparation_policy": policy.model_dump(mode="json"),
        "chunk_variant": "atomic",
    }
    file = session.get_one(UserFile, member.user_file_id)
    file.regulatory_chunk_generation_hash = compute_regulatory_chunk_generation_hash(
        embedding_provider=EmbeddingProvider.GOOGLE,
        embedding_model_name=google_settings.model_name,
        enable_contextual_rag=True,
        intake_chunking_version="4",
        intake_preparation_policy=policy,
    )
    session.flush()
    before = row.id, row.text, deepcopy(row.chunk_metadata)
    edit_budget.validate_intake_edit_budgets(session, {file.id: file}, {file.id: [row]})
    prospective = edit_budget.prospective_intake_edit_rows(
        [row],
        {row.id: {"text": "| 0101 | " + "menşeli " * 1000 + "| Ortak %50 sınırı |"}},
    )
    with pytest.raises(ValueError, match="content limit"):
        edit_budget.validate_intake_edit_budgets(
            session, {file.id: file}, {file.id: prospective}
        )
    session.refresh(row)
    assert (row.id, row.text, row.chunk_metadata) == before


def test_nullable_policy_migration_preserves_preexisting_row_during_upgrade_and_downgrade(
    intake_session: Session,
) -> None:
    connection = intake_session.connection()
    schema = "rg_policy_migration_" + uuid4().hex
    connection.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
    connection.exec_driver_sql(f'SET LOCAL search_path TO "{schema}"')
    try:
        connection.exec_driver_sql(
            "CREATE TABLE regulatory_document_intake (id uuid PRIMARY KEY, marker text NOT NULL)"
        )
        identifier = uuid4()
        connection.execute(
            text("INSERT INTO regulatory_document_intake VALUES (:id,'legacy')"),
            {"id": identifier},
        )
        path = (
            Path(__file__).resolve().parents[3]
            / "alembic/versions/e363f58e0eaf_freeze_new_document_table_preparation_.py"
        )
        spec = importlib.util.spec_from_file_location("intake_policy_migration", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with Operations.context(MigrationContext.configure(connection)):
            module.upgrade()
            column = next(
                c
                for c in inspect(connection).get_columns(
                    "regulatory_document_intake", schema=schema
                )
                if c["name"] == "preparation_policy"
            )
            assert column["nullable"] and column["default"] is None
            assert connection.execute(
                text(
                    "SELECT id,marker,preparation_policy FROM regulatory_document_intake"
                )
            ).one() == (identifier, "legacy", None)
            connection.execute(
                text(
                    "UPDATE regulatory_document_intake SET preparation_policy=CAST(:policy AS jsonb) WHERE id=:id"
                ),
                {
                    "id": identifier,
                    "policy": json.dumps(
                        IntakePreparationPolicy().model_dump(mode="json")
                    ),
                },
            )
            module.downgrade()
        assert connection.execute(
            text("SELECT id,marker FROM regulatory_document_intake")
        ).one() == (identifier, "legacy")
        assert "preparation_policy" not in {
            c["name"]
            for c in inspect(connection).get_columns(
                "regulatory_document_intake", schema=schema
            )
        }
    finally:
        connection.exec_driver_sql("SET LOCAL search_path TO public")
        connection.exec_driver_sql(f'DROP SCHEMA "{schema}" CASCADE')
