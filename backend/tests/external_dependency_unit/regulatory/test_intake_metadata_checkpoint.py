"""Metadata crash recovery binds immutable source and excludes stale workers."""

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import document_intake_metadata as metadata
from onyx.db.document_intake_metadata_checkpoint import (
    metadata_execution_lock,
    require_current_metadata_checkpoint,
    try_reserve_metadata_execution,
)
from onyx.db.labeling_configuration import LabelingProviderBinding
from onyx.db.models import (
    DocumentSet,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryFilePublication,
    RegulatorySourceAsset,
    User,
)
from onyx.regulatory.document_intake import metadata as metadata_contract
from onyx.regulatory.document_intake.metadata import (
    METADATA_AUDIT_KEY,
    ChunkLegalMetadata,
    LegalMetadataFact,
    LegalMetadataResult,
    MetadataChunk,
    metadata_input_hash,
)
from onyx.regulatory.document_intake.metadata_checkpoint import (
    MetadataCheckpoint,
    MetadataLeaseLost,
    metadata_digest,
)
from onyx.regulatory.document_intake.metadata_job import metadata_recipe_sha256
from onyx.regulatory.indexing_jobs.models import VertexAuthenticationMode
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from tests.external_dependency_unit.regulatory.test_document_intake_metadata import (
    _apply,
    _ready_metadata,
)
from tests.external_dependency_unit.regulatory.test_document_intake_metadata import (
    intake_owner as _intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake_metadata import (
    intake_session as _intake_session,
)

intake_owner = _intake_owner
intake_session = _intake_session


def ready_checkpoint(
    session: Session,
    owner: tuple[User, DocumentSet, ModelConfiguration],
    *,
    title: str | None = None,
) -> tuple[
    RegulatoryDocumentIntake,
    list[RegulatoryDocumentIntakeFile],
    UUID,
    LabelingProviderBinding,
]:
    batch, members, token, _article, _closing = _ready_metadata(session, owner)
    if title is not None:
        _article.text = title
    batch.stage = "metadata"
    model = owner[2]
    binding = LabelingProviderBinding(
        model_configuration_id=model.id,
        provider_id=model.llm_provider_id,
        project="metadata-test",
        location="global",
        authentication_mode=VertexAuthenticationMode.WORKLOAD_IDENTITY,
        credential_identity="workload_identity",
        model="gemini-3.5-flash",
        transport="vertex_direct_v1",
    )
    require_current_metadata_checkpoint(
        session,
        batch.id,
        token,
        provider_binding=binding,
        recipe_sha256=metadata_recipe_sha256(),
        capture=True,
    )
    session.flush()
    return batch, members, token, binding


@pytest.mark.parametrize(
    "change",
    [
        "text",
        "heading",
        "source",
        "model",
        "recipe",
        "seed",
        "checksum",
        "publication",
        "parent",
        "text_file",
        "text_hash",
        "original_url",
        "final_url",
    ],
)
def test_checkpoint_rejects_changed_source_or_request_contract(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    change: str,
) -> None:
    batch, members, token, binding = ready_checkpoint(intake_session, intake_owner)
    row = intake_session.scalar(
        select(RegulatoryChunk)
        .where(RegulatoryChunk.user_file_id == members[0].user_file_id)
        .limit(1)
    )
    assert row is not None
    recipe = metadata_recipe_sha256()
    if change == "text":
        row.text += " Changed"
    elif change == "heading":
        row.heading_path = ["Changed"]
    elif change == "source":
        row.chunk_metadata = {
            **row.chunk_metadata,
            "intake_chunking_version": "changed",
        }
    elif change == "model":
        binding = binding.model_copy(update={"project": "changed"})
    elif change == "recipe":
        recipe = "f" * 64
    elif change == "seed":
        row.chunk_metadata = {**row.chunk_metadata, "document_type": "changed"}
    elif change == "checksum":
        assert batch.metadata_checkpoint is not None
        batch.metadata_checkpoint = {
            **batch.metadata_checkpoint,
            "checkpoint_sha256": "f" * 64,
        }
    elif change == "parent":
        members[0].parent_document_id = members[1].id
    elif change in {"text_file", "text_hash", "original_url", "final_url"}:
        asset = intake_session.get_one(
            RegulatorySourceAsset, members[0].source_asset_id
        )
        setattr(
            asset,
            {"text_file": "text_file_id", "text_hash": "text_sha256"}.get(
                change, change
            ),
            "changed",
        )
    else:
        intake_session.get_one(
            RegulatoryFilePublication, members[0].user_file_id
        ).epoch = 1
    with pytest.raises(intake.IntakeConflictError):
        require_current_metadata_checkpoint(
            intake_session,
            batch.id,
            token,
            provider_binding=binding,
            recipe_sha256=recipe,
            capture=False,
        )


def test_completed_document_metadata_does_not_change_frozen_source(
    intake_session: Session, intake_owner: tuple[User, DocumentSet, ModelConfiguration]
) -> None:
    batch, members, token, binding = ready_checkpoint(intake_session, intake_owner)
    assert batch.metadata_checkpoint is not None
    saved = dict(batch.metadata_checkpoint)
    rows = list(
        intake_session.scalars(
            select(RegulatoryChunk)
            .where(RegulatoryChunk.user_file_id == members[0].user_file_id)
            .order_by(RegulatoryChunk.position)
        )
    )
    _apply(intake_session, batch.id, token, members[0].id, rows, rows[-1])
    require_current_metadata_checkpoint(
        intake_session,
        batch.id,
        token,
        provider_binding=binding,
        recipe_sha256=metadata_recipe_sha256(),
        capture=False,
    )
    _batch, pending = metadata.load_metadata_work(intake_session, batch.id, token)
    assert [identifier for identifier, _chunks in pending] == [members[1].id]
    assert batch.metadata_checkpoint == saved


@pytest.mark.parametrize("operation", ["heartbeat", "apply"])
@pytest.mark.parametrize("expired", [False, True])
def test_stale_or_expired_owner_cannot_heartbeat_or_apply(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    operation: str,
    expired: bool,
) -> None:
    batch, members, old_token, _binding = ready_checkpoint(intake_session, intake_owner)
    if expired:
        batch.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    else:
        batch.lease_token = uuid4()
    intake_session.flush()
    token, deadline = batch.lease_token, batch.lease_expires_at
    with pytest.raises(MetadataLeaseLost):
        if operation == "heartbeat":
            metadata.heartbeat_metadata(intake_session, batch.id, old_token)
        else:
            rows = list(
                intake_session.scalars(
                    select(RegulatoryChunk).where(
                        RegulatoryChunk.user_file_id == members[0].user_file_id
                    )
                )
            )
            metadata.apply_metadata_results(
                intake_session,
                intake_id=batch.id,
                lease_token=old_token,
                document_id=members[0].id,
                expected_input_sha256=metadata_input_hash(
                    metadata.metadata_chunks(rows)
                ),
                results=[],
            )
    assert (batch.lease_token, batch.lease_expires_at) == (token, deadline)
    intake.fail_intake(intake_session, batch.id, old_token, "stale error")
    if not expired:
        assert batch.status == "preparing"


def test_execution_mutex_blocks_overlapping_reservation_without_open_transaction(
    intake_session: Session,
) -> None:
    identifier = uuid4()
    with metadata_execution_lock(identifier) as acquired:
        assert acquired
        with metadata_execution_lock(identifier) as duplicate:
            assert not duplicate
        assert not try_reserve_metadata_execution(intake_session, identifier)
    assert try_reserve_metadata_execution(intake_session, identifier)


@pytest.mark.parametrize("operation", ["heartbeat", "apply", "checkpoint"])
def test_same_token_wrong_stage_cannot_mutate_metadata(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    operation: str,
) -> None:
    batch, members, token, binding = ready_checkpoint(intake_session, intake_owner)
    batch.stage = "labeling"
    intake_session.flush()
    deadline = batch.lease_expires_at
    with pytest.raises(MetadataLeaseLost):
        if operation == "heartbeat":
            metadata.heartbeat_metadata(intake_session, batch.id, token)
        elif operation == "apply":
            metadata.apply_metadata_results(
                intake_session,
                intake_id=batch.id,
                lease_token=token,
                document_id=members[0].id,
                expected_input_sha256="a" * 64,
                results=[],
            )
        else:
            require_current_metadata_checkpoint(
                intake_session,
                batch.id,
                token,
                provider_binding=binding,
                recipe_sha256=metadata_recipe_sha256(),
                capture=False,
            )
    assert batch.stage == "labeling" and batch.lease_expires_at == deadline


@pytest.mark.parametrize("stage", ["chunking", "reprepare"])
def test_initial_metadata_loader_enters_stage_without_weakening_writes(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    stage: str,
) -> None:
    batch, _members, token, _binding = ready_checkpoint(intake_session, intake_owner)
    batch.stage = stage
    intake_session.flush()
    current, work = metadata.load_metadata_work(intake_session, batch.id, token)
    assert current.stage == "metadata" and len(work) == 2


def test_checkpoint_checksums_all_frozen_fields() -> None:
    checkpoint = MetadataCheckpoint.build(
        recipe_sha256="a" * 64,
        provider_binding={"model": "test"},
        source_snapshot={"source": "original"},
        request_hashes={"document": ["b" * 64]},
    )
    checkpoint.validate_checksum()
    assert checkpoint.checkpoint_sha256 == metadata_digest(
        checkpoint.model_dump(mode="json", exclude={"checkpoint_sha256"})
    )
    with pytest.raises(ValueError):
        checkpoint.model_copy(update={"request_hashes": {}}).validate_checksum()


def test_completed_title_growth_never_rebuilds_requests_and_receipt_loss_fails_closed(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    title = ("İTHALATTA BELGELERİN İBRAZI HAKKINDA YÖNETMELİK " * 40).strip()
    batch, members, token, binding = ready_checkpoint(
        intake_session, intake_owner, title=title
    )
    rows = list(
        intake_session.scalars(
            select(RegulatoryChunk)
            .where(RegulatoryChunk.user_file_id == members[0].user_file_id)
            .order_by(RegulatoryChunk.position)
        )
    )
    article = rows[0]
    metadata.apply_metadata_results(
        intake_session,
        intake_id=batch.id,
        lease_token=token,
        document_id=members[0].id,
        expected_input_sha256=metadata_input_hash(metadata.metadata_chunks(rows)),
        results=[
            (
                "a" * 64,
                LegalMetadataResult(
                    chunks=[
                        ChunkLegalMetadata(
                            chunk_id=row.id,
                            fields=[
                                LegalMetadataFact(
                                    field="document_title",
                                    value=title,
                                    source_chunk_id=article.id,
                                    evidence_quote=title,
                                )
                            ],
                        )
                        for row in rows
                    ]
                ),
            )
        ],
    )
    assert article.chunk_metadata["intake_display_name"]["title"] == title
    original_builder = metadata_contract.build_metadata_request
    pending_calls: list[list[str]] = []

    def pending_builder(
        chunks: Sequence[MetadataChunk], target_ids: Sequence[str]
    ) -> VertexBatchRequest:
        assert article.id not in {chunk.chunk_id for chunk in chunks}, (
            "completed requests must not be rebuilt"
        )
        pending_calls.append(list(target_ids))
        return original_builder(chunks, target_ids)

    monkeypatch.setattr(metadata_contract, "build_metadata_request", pending_builder)
    require_current_metadata_checkpoint(
        intake_session,
        batch.id,
        token,
        provider_binding=binding,
        recipe_sha256=metadata_recipe_sha256(),
        capture=False,
    )
    assert len(pending_calls) == 1
    # First capture can also begin after one file already has complete receipts.
    batch.metadata_checkpoint = None
    captured = require_current_metadata_checkpoint(
        intake_session,
        batch.id,
        token,
        provider_binding=binding,
        recipe_sha256=metadata_recipe_sha256(),
        capture=True,
    )
    assert captured.request_hashes[str(members[0].id)] == []
    assert captured.request_hashes[str(members[1].id)]
    monkeypatch.setattr(metadata_contract, "build_metadata_request", original_builder)
    for row in rows:
        row.chunk_metadata = {
            key: value
            for key, value in row.chunk_metadata.items()
            if key != METADATA_AUDIT_KEY
        }
    with pytest.raises(
        intake.IntakeConflictError,
        match="request contract changed|source, recipe or provider",
    ):
        require_current_metadata_checkpoint(
            intake_session,
            batch.id,
            token,
            provider_binding=binding,
            recipe_sha256=metadata_recipe_sha256(),
            capture=False,
        )
