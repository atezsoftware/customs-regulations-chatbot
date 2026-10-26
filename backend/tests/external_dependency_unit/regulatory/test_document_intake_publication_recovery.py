"""Frozen failed intake work is replaced under closed ownership, with SQL proof."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import document_intake_publication_recovery as repository
from onyx.db import regulatory_publication, regulatory_writer_publication
from onyx.db.enums import UserFileProjectionRepairStatus, UserFileStatus
from onyx.db.models import (
    DocumentSet,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryContextSnapshot,
    RegulatoryDocumentIntakePublicationRecovery,
    RegulatoryFilePublication,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryPublicationClock,
    RegulatoryPublicationOrdinal,
    RegulatorySourceAsset,
    RegulatoryTemporalProjection,
    User,
    UserFile,
    UserFileProjectionRepair,
)
from onyx.db.regulatory_annex_changes import capture_canonical_scope
from onyx.db.regulatory_publication import PublicationStore
from onyx.db.regulatory_writer_publication import (
    canonical_scope_digest,
    finalize_writer_publication,
    writer_index_state_digest,
)
from onyx.db.user_file import (
    finish_user_file_index_request,
    start_user_file_index_request,
)
from onyx.document_index.publication_models import (
    FrozenPublicationProjection,
    PublicationIndexSnapshot,
    PublicationScope,
    PublicationVerification,
    publication_digest,
    publication_streaming_digest,
)
from onyx.file_processing.original_ingestion import OriginalIngestionReceipt
from onyx.regulatory.amendments.annexes import config
from onyx.regulatory.amendments.annexes.context_dependencies import (
    rebuild_context_aggregates,
)
from onyx.regulatory.amendments.annexes.models import AnnexTemporalProjection
from onyx.regulatory.chunk_evidence import chunk_evidence
from onyx.regulatory.document_intake import publication_recovery as worker
from onyx.regulatory.document_intake.temporal_context import (
    intake_temporal_context_rows,
)
from onyx.regulatory.writer_publication_models import WriterPublicationManifest
from tests.external_dependency_unit.regulatory.test_document_intake import (
    _ENVIRONMENT,
    _admitted,
    _completed_labels,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_owner as intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_session as intake_session,
)
from tests.external_dependency_unit.regulatory.test_document_intake_index_retry import (
    _rows,
)


def _binding(
    row: RegulatoryChunk, index: PublicationIndexSnapshot
) -> AnnexTemporalProjection:
    identifier = uuid4()
    evidence = chunk_evidence(row.chunk_metadata)
    source = {
        "document_id": str(row.user_file_id),
        "chunk_index": row.projection_ordinal,
        "regulatory_chunk_id": row.id,
        "source_type": "file",
        "public": False,
        "access_control_list": [],
        "global_boost": 1,
        "semantic_identifier": "Law",
        "blurb": row.text,
        "content": row.text,
        "doc_summary": "",
        "chunk_context": "",
        "content_vector": [0.1, 0.2, 0.3],
        "hidden": False,
        "image_file_id": evidence.image_file_id,
        "source_links": json.dumps(evidence.source_links)
        if evidence.source_links
        else None,
        "validity_start_date": int(
            datetime.combine(
                row.validity_start_date, datetime.min.time(), timezone.utc
            ).timestamp()
        )
        if row.validity_start_date
        else None,
        "validity_end_date": None,
    }
    aggregate = row.chunk_metadata.get("chunk_variant") == "hierarchical_aggregate"
    return AnnexTemporalProjection(
        id=identifier,
        index=index,
        projection=FrozenPublicationProjection(
            ordinal=row.projection_ordinal,
            context_projection_id=str(identifier),
            source_json=json.dumps(source),
            embedding_inputs=(row.text,),
            embedding_config_json='{"model":"fixture"}',
        ),
        canonical_base_sha256=publication_digest(row.text),
        derived_role="hierarchical_aggregate" if aggregate else "canonical",
        dependency_ids=row.chunk_metadata.get("source_regulatory_chunk_ids", []),
        representation_text=row.text,
        representation_metadata=row.chunk_metadata,
        reference_date=row.validity_start_date,
        effective_start=row.validity_start_date,
        effective_end=None,
        semantic_position=row.position,
    )


def _fixture(
    session: Session,
    owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    *,
    invalid_manifest_hash: bool = False,
) -> tuple[Any, Any, Any, WriterPublicationManifest]:
    user, target, model = owner
    batch, members = _admitted(session, user, target, model)
    main = next(member for member in members if member.role == "root")
    annex = next(member for member in members if member.role == "annex")
    atomic = session.scalar(
        select(RegulatoryChunk).where(RegulatoryChunk.user_file_id == main.user_file_id)
    )
    assert atomic is not None
    atomic.validity_start_date = date(2025, 5, 9)
    atomic.chunk_metadata = {**atomic.chunk_metadata, "intake_chunking_version": "1"}
    aggregate = RegulatoryChunk(
        id=f"rc_aggregate_{uuid4().hex}",
        user_file_id=main.user_file_id,
        text="pending",
        position=1,
        projection_ordinal=1,
        heading_path=["MADDE 1"],
        chunk_type="hierarchical_aggregate",
        source="indexed",
        status="active",
        chunk_metadata={
            "intake_chunking_version": "1",
            "chunk_variant": "hierarchical_aggregate",
            "hierarchy_root_path": ["MADDE 1"],
            "source_regulatory_chunk_ids": [atomic.id],
        },
    )
    aggregate.text = rebuild_context_aggregates(
        [atomic, aggregate], changed_ids=[aggregate.id]
    )[-1].text
    session.add(aggregate)
    session.flush()
    scope = PublicationScope(
        tenant_id="public",
        environment=_ENVIRONMENT,
        database_identity=config.ANNEX_DATABASE_IDENTITY,
    )
    monkeypatch.setattr(config, "REGULATORY_ANNEX_ENVIRONMENT", _ENVIRONMENT)
    for member in members:
        file = session.get_one(UserFile, member.user_file_id)
        asset = session.get_one(RegulatorySourceAsset, member.source_asset_id)
        file.regulatory_chunk_generation_hash = "a" * 64
        receipt = OriginalIngestionReceipt(
            file_id=file.file_id,
            raw_sha256=asset.sha256,
            documents_sha256="b" * 64,
            plaintext_sha256=asset.sha256,
            canonical_sha256=canonical_scope_digest(session, file.id),
            generation_hash="a" * 64,
        )
        session.add(
            RegulatoryFilePublication(
                user_file_id=file.id,
                scope_key=publication_digest(scope.model_dump(mode="json")),
                owner_id=uuid4(),
                fencing_token=1,
                lease_expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
                gate_closed=False,
                epoch=0,
                next_ordinal=2,
                original_ingestion_receipt=receipt.model_dump(mode="json"),
            )
        )
        session.flush()
        for chunk in session.scalars(
            select(RegulatoryChunk).where(RegulatoryChunk.user_file_id == file.id)
        ):
            session.add(
                RegulatoryPublicationOrdinal(
                    user_file_id=file.id,
                    allocation_key=f"canonical:{chunk.id}",
                    ordinal=chunk.projection_ordinal,
                )
            )
    _completed_labels(session, batch, members)
    intake.refresh_intake(session, batch)
    assert batch.review_sha256
    tokens = dict(intake.approve_intake(session, batch, batch.review_sha256))
    session.commit()
    assert start_user_file_index_request(
        session, main.user_file_id, tokens[main.user_file_id]
    )
    assert finish_user_file_index_request(
        session,
        main.user_file_id,
        tokens[main.user_file_id],
        failure_code="index:ValueError",
    )
    index = PublicationIndexSnapshot(
        index_name="recovery-sql-fixture",
        index_uuid=uuid4().hex,
        search_settings_id=1,
        model_provider="fixture",
        model_name="fixture",
        vector_dimension=3,
        embedding_config_sha256=publication_digest({"model": "fixture"}),
        multitenant=False,
    )
    old = WriterPublicationManifest(
        id=uuid4(),
        scope=scope,
        user_file_id=main.user_file_id,
        kind="reindex",
        index_state_sha256=writer_index_state_digest(session, [1]),
        canonical_before_sha256=canonical_scope_digest(session, main.user_file_id),
        canonical_after=capture_canonical_scope(session, main.user_file_id),
        complete_file=True,
        chunk_count_after=2,
        indexes=[index],
        previous_binding_ids=[],
        bindings=[_binding(atomic, index), _binding(aggregate, index)],
    )
    publication = session.get_one(RegulatoryFilePublication, main.user_file_id)
    publication.writer_manifest = old.model_dump(mode="json")
    publication.writer_manifest_sha256 = publication_streaming_digest(
        publication.writer_manifest
    )
    if invalid_manifest_hash:
        publication.writer_manifest_sha256 = "0" * 64
    publication.gate_closed, publication.epoch = True, 19
    session.merge(RegulatoryPublicationClock(scope_key=publication.scope_key, epoch=19))
    intake.refresh_intake(session, batch)
    assert batch.status == "failed"
    session.commit()
    connection = session.connection()

    @contextmanager
    def shared(**_kwargs: object) -> Iterator[Session]:
        with Session(
            connection, join_transaction_mode="create_savepoint", expire_on_commit=False
        ) as current:
            yield current

    for module in (repository, regulatory_publication, regulatory_writer_publication):
        monkeypatch.setattr(module, "get_session_with_tenant", shared)
    monkeypatch.setattr("onyx.db.engine.sql_engine.get_session_with_tenant", shared)
    return batch, main, annex, old


def test_normal_retry_rebuilds_closed_manifest_and_prepared_resume_uses_no_model(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = intake_session
    batch, main, annex, old = _fixture(session, intake_owner, monkeypatch)
    annex_token = annex.approval_attempt_id
    retained = _rows(
        session,
        RegulatoryChunk,
        RegulatoryLabelingItem,
        RegulatoryLabelingRun,
        RegulatorySourceAsset,
    )
    reviewed = batch.review_sha256
    original_receipt = deepcopy(
        session.get_one(
            RegulatoryFilePublication, main.user_file_id
        ).original_ingestion_receipt
    )
    deliveries = dict(intake.retry_intake(session, batch))
    assert deliveries[annex.user_file_id] == annex_token
    recovery = session.scalar(
        select(RegulatoryDocumentIntakePublicationRecovery).where(
            RegulatoryDocumentIntakePublicationRecovery.user_file_id
            == main.user_file_id
        )
    )
    assert recovery is not None and recovery.original_manifest == old.model_dump(
        mode="json"
    )
    assert recovery.previous_attempt_id != deliveries[main.user_file_id]
    assert batch.review_sha256 == reviewed
    token = deliveries[main.user_file_id]
    assert start_user_file_index_request(session, main.user_file_id, token)
    prepares = []

    def prepare(
        _owner: Any, _client: Any, inputs: Any, _after: Any, **_kwargs: Any
    ) -> WriterPublicationManifest:
        assert inputs.cached.calls == [] and inputs.bindings == []
        prepares.append(True)
        rows = list(
            session.scalars(
                select(RegulatoryChunk)
                .where(RegulatoryChunk.user_file_id == main.user_file_id)
                .order_by(RegulatoryChunk.position)
            )
        )
        qualified = intake_temporal_context_rows(rows)
        assert rows[1].validity_start_date is None and qualified[
            1
        ].validity_start_date == date(2025, 5, 9)
        return old.model_copy(
            update={
                "id": uuid4(),
                "bindings": [_binding(row, old.indexes[0]) for row in qualified],
            }
        )

    @contextmanager
    def client() -> Iterator[Any]:
        yield SimpleNamespace(publication_client=lambda: None)

    @contextmanager
    def heartbeat(_owner: Any) -> Iterator[Any]:
        yield SimpleNamespace(is_set=lambda: False)

    monkeypatch.setattr(worker, "ElasticsearchClient", client)
    monkeypatch.setattr(worker, "publication_heartbeat", heartbeat)
    monkeypatch.setattr(worker, "prepare_owned_correction", prepare)
    monkeypatch.setattr(
        worker,
        "complete_writer_index_inventory",
        lambda _owner, _client, manifest: manifest,
    )

    def interrupted(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("interrupted after replacement was frozen")

    monkeypatch.setattr(worker, "execute_writer_publication", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        worker.recover_intake_publication(
            user_file_id=main.user_file_id,
            tenant_id="public",
            attempt_id=token,
            recovery_id=recovery.id,
        )
    session.expire_all()
    publication = session.get_one(RegulatoryFilePublication, main.user_file_id)
    recovery = session.get_one(RegulatoryDocumentIntakePublicationRecovery, recovery.id)
    frozen = deepcopy(recovery.replacement_manifest)
    assert (
        frozen is not None
        and publication.writer_manifest == frozen
        and publication.gate_closed
        and publication.epoch > 19
    )
    assert recovery.original_manifest == old.model_dump(mode="json")

    def execute(owner: Any, _client: Any) -> None:
        current = session.get_one(
            RegulatoryFilePublication, main.user_file_id, populate_existing=True
        )
        assert current.gate_closed
        manifest = WriterPublicationManifest.model_validate(current.writer_manifest)
        reservations = PublicationStore(owner.scope).reservations(owner)
        assert all(binding.projection.ordinal > 1 for binding in manifest.bindings)
        proof = PublicationVerification(
            reservations=reservations,
            index=manifest.indexes[0],
            live_ordinals=tuple(
                sorted(binding.projection.ordinal for binding in manifest.bindings)
            ),
            canonical_chunk_ids=frozenset(
                binding.projection.context_projection_id
                for binding in manifest.bindings
            ),
            manifest_sha256="e" * 64,
        )
        finalize_writer_publication(owner, manifest, [proof])

    monkeypatch.setattr(worker, "execute_writer_publication", execute)

    class ProcessStopped(BaseException):
        pass

    real_finish = worker.finish_recovery

    def stopped_after_commit(*_args: Any, **_kwargs: Any) -> None:
        raise ProcessStopped()

    monkeypatch.setattr(worker, "finish_recovery", stopped_after_commit)
    with pytest.raises(ProcessStopped):
        worker.recover_intake_publication(
            user_file_id=main.user_file_id,
            tenant_id="public",
            attempt_id=token,
            recovery_id=recovery.id,
        )
    session.expire_all()
    assert (
        session.get_one(RegulatoryDocumentIntakePublicationRecovery, recovery.id).status
        != "completed"
    )
    assert repository.committed_recovery_delivery(session, main.user_file_id, token)
    monkeypatch.setattr(worker, "finish_recovery", real_finish)
    from onyx.background.celery.tasks.user_file_processing import tasks

    connection = session.connection()

    @contextmanager
    def current_session() -> Iterator[Session]:
        with Session(
            connection, join_transaction_mode="create_savepoint", expire_on_commit=False
        ) as current:
            yield current

    monkeypatch.setattr(tasks, "get_session_with_current_tenant", current_session)
    tasks._index_requested_user_file(str(main.user_file_id), "public", str(token))
    session.expire_all()
    assert len(prepares) == 1
    recovery = session.get_one(RegulatoryDocumentIntakePublicationRecovery, recovery.id)
    publication = session.get_one(RegulatoryFilePublication, main.user_file_id)
    assert recovery.status == "completed" and recovery.replacement_manifest == frozen
    assert publication.writer_manifest is None and not publication.gate_closed
    assert (
        session.get_one(UserFile, main.user_file_id).status == UserFileStatus.COMPLETED
    )
    assert (
        session.get_one(UserFileProjectionRepair, main.user_file_id).status
        == UserFileProjectionRepairStatus.SUCCEEDED
    )
    assert (
        session.get_one(UserFileProjectionRepair, annex.user_file_id).attempt_id
        == annex_token
    )
    assert publication.original_ingestion_receipt == original_receipt
    assert (
        _rows(
            session,
            RegulatoryChunk,
            RegulatoryLabelingItem,
            RegulatoryLabelingRun,
            RegulatorySourceAsset,
        )
        == retained
    )
    assert session.scalars(
        select(RegulatoryTemporalProjection).where(
            RegulatoryTemporalProjection.user_file_id == main.user_file_id
        )
    ).all()
    # The immutable proof remains retained after successful activation.
    for field, value in [
        ("original_manifest_sha256", "0" * 64),
        ("replacement_manifest_sha256", "0" * 64),
        ("approved_review_sha256", "0" * 64),
    ]:
        with pytest.raises(DBAPIError), session.begin_nested():
            setattr(recovery, field, value)
            session.flush()
        session.refresh(recovery)
    with pytest.raises(DBAPIError), session.begin_nested():
        session.delete(recovery)
        session.flush()
    session.refresh(recovery)
    recovery.status = "completed"
    recovery.error = "audit status may be updated"
    session.flush()
    session.add(
        RegulatoryDocumentIntakePublicationRecovery(
            id=uuid4(),
            intake_id=batch.id,
            document_id=main.id,
            user_file_id=main.user_file_id,
            previous_attempt_id=uuid4(),
            replacement_attempt_id=uuid4(),
            approved_review_sha256=reviewed,
            canonical_sha256=canonical_scope_digest(session, main.user_file_id),
            original_manifest=old.model_dump(mode="json"),
            original_manifest_sha256=publication_streaming_digest(
                old.model_dump(mode="json")
            ),
            original_fencing_token=1,
            status="failed",
        )
    )
    session.flush()
    assert repository.recovery_for_attempt(session, main.user_file_id, uuid4()) is None


@pytest.mark.parametrize(
    "change",
    [
        "review",
        "source",
        "intent",
        "gate",
        "manifest_hash",
        "canonical",
        "history",
        "lease",
    ],
)
def test_closed_recovery_rejects_changed_approval_source_and_history_atomically(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    session = intake_session
    batch, main, annex, _old = _fixture(
        session,
        intake_owner,
        monkeypatch,
        invalid_manifest_hash=change == "manifest_hash",
    )
    publication = session.get_one(RegulatoryFilePublication, main.user_file_id)
    if change == "review":
        batch.review_sha256 = "0" * 64
    elif change == "source":
        session.get_one(UserFile, main.user_file_id).file_id = session.get_one(
            UserFile, annex.user_file_id
        ).file_id
    elif change == "intent":
        session.get_one(
            UserFileProjectionRepair, main.user_file_id
        ).attempt_id = uuid4()
    elif change == "gate":
        publication.gate_closed = False
    elif change == "manifest_hash":
        pass
    elif change == "canonical":
        chunk = session.scalar(
            select(RegulatoryChunk).where(
                RegulatoryChunk.user_file_id == main.user_file_id
            )
        )
        assert chunk is not None
        chunk.text += "unreviewed"
    elif change == "history":
        session.add(
            RegulatoryContextSnapshot(
                id=uuid4(), user_file_id=main.user_file_id, sha256="x" * 64, payload={}
            )
        )
    else:
        publication.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=1)
    session.flush()
    before = _rows(
        session,
        RegulatoryDocumentIntakePublicationRecovery,
        UserFile,
        UserFileProjectionRepair,
        RegulatoryFilePublication,
    )
    with pytest.raises(intake.IntakeConflictError):
        intake.retry_intake(session, batch)
    assert (
        _rows(
            session,
            RegulatoryDocumentIntakePublicationRecovery,
            UserFile,
            UserFileProjectionRepair,
            RegulatoryFilePublication,
        )
        == before
    )


def test_expired_recovery_delivery_reassociates_only_through_reviewed_retry(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.db.user_file import recover_user_file_index_requests

    session = intake_session
    batch, main, annex, _old = _fixture(session, intake_owner, monkeypatch)
    original = dict(intake.retry_intake(session, batch))
    prior_id = repository.recovery_for_attempt(
        session, main.user_file_id, original[main.user_file_id]
    )
    assert prior_id is not None
    with pytest.raises(intake.IntakeConflictError):
        repository.recovery_for_attempt(session, main.user_file_id, uuid4())
    intake.mark_intake_index_dispatch_failed(session, batch, "delivery was not queued")
    repair = session.get_one(UserFileProjectionRepair, main.user_file_id)
    repair.updated_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    session.flush()
    rotated = dict(
        recover_user_file_index_requests(
            session,
            stale_before=datetime.now(timezone.utc) - timedelta(minutes=1),
            limit=1,
        )
    )
    assert set(rotated) == {main.user_file_id}
    with pytest.raises(intake.IntakeConflictError):
        repository.recovery_for_attempt(
            session, main.user_file_id, rotated[main.user_file_id]
        )
    retried = dict(intake.retry_intake(session, batch))
    assert retried[main.user_file_id] == rotated[main.user_file_id]
    assert retried[annex.user_file_id] == original[annex.user_file_id]
    current_id = repository.recovery_for_attempt(
        session, main.user_file_id, retried[main.user_file_id]
    )
    assert current_id is not None and current_id != prior_id
    previous = session.get_one(RegulatoryDocumentIntakePublicationRecovery, prior_id)
    current = session.get_one(RegulatoryDocumentIntakePublicationRecovery, current_id)
    assert previous.status == "failed" and current.status == "queued"
    assert previous.original_manifest == current.original_manifest
    assert current.previous_attempt_id == previous.replacement_attempt_id
