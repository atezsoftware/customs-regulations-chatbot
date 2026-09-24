"""Transaction-isolated PostgreSQL recovery tests; never publish source changes."""

from collections.abc import Generator
from uuid import uuid4

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from onyx.db.amendment_match_checkpoints import (
    capture_match_evidence,
    load_match_checkpoint,
    persist_match_checkpoint,
)
from onyx.db.engine.sql_engine import get_sqlalchemy_engine
from onyx.db.models import AmendmentBatch, DocumentSet, RegulatoryChunk
from onyx.regulatory.amendments.match_checkpoint import MatchedInstruction
from onyx.regulatory.amendments.models import AmendmentInstruction, MatchResult
from onyx.regulatory.amendments.ranker import CandidateChunk


@pytest.fixture
def checkpoint_session(
    db_session: Session, tenant_context: None
) -> Generator[Session, None, None]:
    del db_session, tenant_context
    from onyx.db.models import AmendmentMatchCheckpoint

    with get_sqlalchemy_engine().connect() as connection:
        transaction = connection.begin()
        try:
            AmendmentMatchCheckpoint.metadata.tables[
                "amendment_match_checkpoint"
            ].create(connection, checkfirst=True)
            with Session(
                connection, join_transaction_mode="create_savepoint"
            ) as session:
                yield session
        finally:
            transaction.rollback()


def test_match_survives_session_restart_and_rejects_changed_source_or_inputs(
    checkpoint_session: Session,
) -> None:
    session = checkpoint_session
    source = session.scalar(
        select(RegulatoryChunk).where(RegulatoryChunk.status == "active").limit(1)
    )
    assert source is not None
    source = RegulatoryChunk(
        id=f"checkpoint-{uuid4()}",
        user_file_id=source.user_file_id,
        text="checkpoint fixture",
        position=0,
        projection_ordinal=uuid4().int % 700000000 + 100000000,
    )
    session.add(source)
    session.flush()
    docset = DocumentSet(
        name=f"checkpoint-{uuid4()}",
        description="checkpoint recovery",
        is_up_to_date=True,
    )
    session.add(docset)
    session.flush()
    batch = AmendmentBatch(
        document_set_id=docset.id,
        raw_text="test",
        status="analyzing",
        stage="processing",
        lease_generation=2,
        instruction_count=2,
        user_file_ids=[str(source.user_file_id)],
    )
    session.add(batch)
    session.flush()
    batch_id = batch.id
    checkpoint = MatchedInstruction(
        instruction_index=0,
        instruction=AmendmentInstruction(instruction_text="First instruction"),
        candidates=[
            CandidateChunk(
                chunk_id=source.id,
                user_file_id=str(source.user_file_id),
                text=source.text,
            )
        ],
        match=MatchResult(old_chunk_id=source.id, confidence=1, rationale="verified"),
    )
    checkpoint.evidence = capture_match_evidence(
        session, batch.user_file_ids, checkpoint.candidates
    )
    assert persist_match_checkpoint(
        session,
        batch_id=batch_id,
        lease_generation=2,
        input_sha256="a" * 64,
        checkpoint=checkpoint,
    )
    session.expunge_all()
    restored = load_match_checkpoint(
        session, batch_id=batch_id, instruction_index=0, input_sha256="a" * 64
    )
    assert restored == checkpoint
    assert (
        load_match_checkpoint(
            session, batch_id=batch_id, instruction_index=0, input_sha256="b" * 64
        )
        is None
    )
    assert not persist_match_checkpoint(
        session,
        batch_id=batch_id,
        lease_generation=1,
        input_sha256="a" * 64,
        checkpoint=checkpoint,
    )
    source = session.get(RegulatoryChunk, checkpoint.candidates[0].chunk_id)
    assert source is not None
    # Only an outer-transaction test write: no committed corpus mutation.
    source.text += " changed in test transaction"
    session.flush()
    assert (
        load_match_checkpoint(
            session, batch_id=batch_id, instruction_index=0, input_sha256="a" * 64
        )
        is None
    )


def test_duplicate_match_write_does_not_increment_finalized_progress(
    checkpoint_session: Session,
) -> None:
    from onyx.db.models import AmendmentMatchCheckpoint

    session = checkpoint_session
    source = session.scalar(
        select(RegulatoryChunk).where(RegulatoryChunk.status == "active").limit(1)
    )
    assert source is not None
    source = RegulatoryChunk(
        id=f"checkpoint-{uuid4()}",
        user_file_id=source.user_file_id,
        text="checkpoint fixture",
        position=0,
        projection_ordinal=uuid4().int % 700000000 + 100000000,
    )
    session.add(source)
    session.flush()
    docset = DocumentSet(
        name=f"checkpoint-{uuid4()}",
        description="checkpoint idempotency",
        is_up_to_date=True,
    )
    session.add(docset)
    session.flush()
    batch = AmendmentBatch(
        document_set_id=docset.id,
        raw_text="test",
        status="analyzing",
        stage="processing",
        lease_generation=3,
        instruction_count=1,
        user_file_ids=[str(source.user_file_id)],
    )
    session.add(batch)
    session.flush()
    checkpoint = MatchedInstruction(
        instruction_index=0,
        instruction=AmendmentInstruction(instruction_text="First"),
        candidates=[
            CandidateChunk(
                chunk_id=source.id,
                user_file_id=str(source.user_file_id),
                text=source.text,
            )
        ],
        match=MatchResult(old_chunk_id=source.id, confidence=1, rationale="verified"),
    )
    checkpoint.evidence = capture_match_evidence(
        session, batch.user_file_ids, checkpoint.candidates
    )
    for _ in range(2):
        assert persist_match_checkpoint(
            session,
            batch_id=batch.id,
            lease_generation=3,
            input_sha256="c" * 64,
            checkpoint=checkpoint,
        )
    assert (
        len(
            session.scalars(
                select(AmendmentMatchCheckpoint).where(
                    AmendmentMatchCheckpoint.batch_id == batch.id
                )
            ).all()
        )
        == 1
    )
    session.refresh(batch)
    assert batch.processed_instruction_count == 0
    assert batch.processed_instruction_indices == []


def test_atomic_heading_and_addition_approval_is_resumable(
    checkpoint_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import datetime, timezone
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from onyx.db import regulatory_chunks, regulatory_publication
    from onyx.db.models import AmendmentProposal
    from onyx.db.regulatory_amendments import approve_amendment_proposal
    from onyx.document_index.publication_models import FileOwnership, PublicationScope

    session = checkpoint_session
    existing_file = session.scalar(select(RegulatoryChunk.user_file_id).limit(1))
    assert existing_file is not None
    old = RegulatoryChunk(
        id=f"checkpoint-{uuid4()}",
        user_file_id=existing_file,
        text="(1) Existing.",
        position=0,
        projection_ordinal=uuid4().int % 700000000 + 100000000,
        chunk_type="paragraph",
        chunk_metadata={"article_no": "10", "paragraph_no": "1"},
        heading_path=["Source", "MADDE 10 - Yetki", "(1) Existing."],
    )
    docset = DocumentSet(
        name=f"checkpoint-{uuid4()}", description="atomic approval", is_up_to_date=True
    )
    session.add_all([old, docset])
    session.flush()
    batch = AmendmentBatch(
        document_set_id=docset.id,
        raw_text="test",
        status="analyzed",
        instruction_count=1,
    )
    session.add(batch)
    session.flush()
    monkeypatch.setattr(
        regulatory_chunks,
        "get_active_chunks_by_structural_reference",
        lambda *_args, **_kwargs: [SimpleNamespace(chunk=old)],
    )
    base = {
        "user_file_id": str(existing_file),
        "chunk_type": "paragraph",
        "effective_start_date": None,
        "effective_end_date": None,
    }
    addition = {
        **base,
        "position": 1,
        "text": "(4) New paragraph.",
        "heading_path": [
            "Source",
            "MADDE 10 - Yetki, denetim ve izleme",
            "(4) New paragraph.",
        ],
        "metadata": {"article_no": "10", "paragraph_no": "4"},
    }
    heading = {
        **base,
        "position": 0,
        "text": old.text,
        "heading_path": [
            "Source",
            "MADDE 10 - Yetki, denetim ve izleme",
            "(1) Existing.",
        ],
        "metadata": old.chunk_metadata,
    }
    proposal = AmendmentProposal(
        batch_id=batch.id,
        instruction_index=0,
        instruction_text="Add paragraph and change heading",
        instruction_indices=[0],
        instruction_texts=["Add paragraph and change heading"],
        status="approving",
        old_chunk_snapshot={
            "heading_change_scope": {"article_no": "10", "chunk_ids": [old.id]},
        },
        new_chunk_draft=addition,
        chunk_changes=[
            {
                "old_chunk_id": None,
                "old_chunk_snapshot": {},
                "new_chunk_draft": addition,
            },
            {
                "old_chunk_id": old.id,
                "old_chunk_snapshot": {
                    "id": old.id,
                    "text": old.text,
                    "chunk_type": old.chunk_type,
                    "metadata": old.chunk_metadata,
                    "heading_path": old.heading_path,
                    "heading_change": {
                        "article_no": "10",
                        "title": "Yetki, denetim ve izleme",
                    },
                },
                "new_chunk_draft": heading,
            },
        ],
    )
    session.add(proposal)
    session.flush()
    store = MagicMock()
    store.allocate_in_session.side_effect = [
        uuid4().int % 700000000 + 100000000,
        uuid4().int % 700000000 + 100000000,
    ]
    monkeypatch.setattr(
        regulatory_publication, "PublicationStore", lambda _scope: store
    )
    owner = FileOwnership(
        scope=PublicationScope(
            tenant_id="public",
            environment="dev",
            database_identity="customs-regulations-dev",
        ),
        user_file_id=existing_file,
        owner_id=uuid4(),
        fencing_token=1,
        expires_at=datetime.now(timezone.utc),
    )
    result = approve_amendment_proposal(session, proposal, publication_owner=owner)
    assert len(result.new_chunks) == 2
    assert result.old_chunks[0] is None
    assert result.new_chunks[0].supersedes_chunk_id is None
    assert result.new_chunks[1].supersedes_chunk_id == old.id
    assert result.new_chunks[1].heading_path[1] == "MADDE 10 - Yetki, denetim ve izleme"
    assert result.new_chunks[1].chunk_metadata["paragraph_no"] == "1"
    assert old.status == "superseded"
    resumed = approve_amendment_proposal(session, proposal, publication_owner=owner)
    assert [row.id for row in resumed.new_chunks] == [
        row.id for row in result.new_chunks
    ]
    assert store.allocate_in_session.call_count == 2


@pytest.mark.parametrize("change_before_save", [False, True])
def test_matching_evidence_rejects_allowed_source_identity_changes(
    checkpoint_session: Session, change_before_save: bool
) -> None:
    from onyx.db import amendment_match_checkpoints as checkpoints
    from onyx.db.models import UserFile

    session = checkpoint_session
    files = session.scalars(select(UserFile).limit(2)).all()
    assert len(files) == 2
    source = RegulatoryChunk(
        id=f"checkpoint-{uuid4()}",
        user_file_id=files[0].id,
        text="source fixture",
        position=0,
        projection_ordinal=uuid4().int % 700000000 + 100000000,
    )
    docset = DocumentSet(
        name=f"checkpoint-{uuid4()}", description="scope", is_up_to_date=True
    )
    session.add_all([source, docset])
    session.flush()
    batch = AmendmentBatch(
        document_set_id=docset.id,
        raw_text="test",
        status="analyzing",
        stage="processing",
        lease_generation=1,
        instruction_count=1,
        user_file_ids=[str(f.id) for f in files],
    )
    session.add(batch)
    session.flush()
    candidate = CandidateChunk(
        chunk_id=source.id, user_file_id=str(source.user_file_id), text=source.text
    )
    evidence = checkpoints.capture_match_evidence(
        session, batch.user_file_ids, [candidate]
    )
    checkpoint = MatchedInstruction(
        instruction_index=0,
        instruction=AmendmentInstruction(instruction_text="Add paragraph"),
        candidates=[candidate],
        match=MatchResult(old_chunk_id=None, confidence=1, rationale="source verified"),
        evidence=evidence,
    )
    if not change_before_save:
        assert persist_match_checkpoint(
            session,
            batch_id=batch.id,
            lease_generation=1,
            input_sha256="e" * 64,
            checkpoint=checkpoint,
        )
    # A different allowed source can make the formerly unique source ambiguous.
    # Change only the fixture transaction; the outer fixture rolls it back.
    files[1 if not change_before_save else 0].name += " renamed law 4458"
    session.flush()
    if change_before_save:
        with pytest.raises(ValueError, match="evidence changed"):
            persist_match_checkpoint(
                session,
                batch_id=batch.id,
                lease_generation=1,
                input_sha256="e" * 64,
                checkpoint=checkpoint,
            )
    else:
        assert (
            load_match_checkpoint(
                session, batch_id=batch.id, instruction_index=0, input_sha256="e" * 64
            )
            is None
        )


def test_resource_deferral_preserves_progress_fences_writer_and_bounds_retries(
    checkpoint_session: Session,
) -> None:
    import importlib.util
    from datetime import timedelta
    from pathlib import Path

    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    from onyx.db.amendment_resources import (
        defer_analysis,
        owns_analysis,
        parallel_analysis_allowed,
    )
    from onyx.db.regulatory_amendments import claim_stale_batches_for_recovery

    session = checkpoint_session
    # Exercise the actual migration against a session-local clone only. Never
    # take ALTER TABLE locks or change constraints on the shared DEV table.
    session.execute(
        text(
            "CREATE TEMP TABLE amendment_batch (LIKE public.amendment_batch INCLUDING ALL) ON COMMIT DROP"
        )
    )
    assert session.scalar(
        text("SELECT pg_table_is_visible('pg_temp.amendment_batch'::regclass)")
    )
    spec = importlib.util.spec_from_file_location(
        "resource_migration",
        Path(__file__).parents[3]
        / "alembic/versions/d8b7a14f902e_amendment_resource_wait_states.py",
    )
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    migration.op = Operations(MigrationContext.configure(session.connection()))
    migration.upgrade()
    docset = DocumentSet(
        name=f"resources-{uuid4()}", description="resource test", is_up_to_date=True
    )
    session.add(docset)
    session.flush()
    batch = AmendmentBatch(
        document_set_id=docset.id,
        raw_text="test",
        status="analyzing",
        stage="processing",
        lease_generation=1,
        instruction_count=3,
        processed_instruction_count=1,
        processed_instruction_indices=[2],
        unmatched_instructions=[],
        user_file_ids=[str(uuid4())],
    )
    session.add(batch)
    session.flush()
    assert parallel_analysis_allowed(session, batch.id)
    assert defer_analysis(
        session,
        batch_id=batch.id,
        lease_generation=1,
        reason="memory_pressure",
        started=True,
    )
    session.refresh(batch)
    assert (batch.status, batch.stage) == ("queued", "waiting_resources")
    assert batch.processed_instruction_indices == [2]
    assert batch.processed_instruction_count == 1
    assert batch.unmatched_instructions == []
    assert not owns_analysis(session, batch_id=batch.id, lease_generation=1)
    assert not parallel_analysis_allowed(session, batch.id)
    waiting_since = batch.heartbeat_at
    assert (
        claim_stale_batches_for_recovery(
            session,
            stale_before=waiting_since - timedelta(minutes=10),
            claimed_at=waiting_since + timedelta(seconds=59),
        )
        == []
    )
    assert claim_stale_batches_for_recovery(
        session,
        stale_before=waiting_since - timedelta(minutes=10),
        claimed_at=waiting_since + timedelta(seconds=61),
    ) == [batch.id]

    assert not defer_analysis(
        session, batch_id=batch.id, lease_generation=1, reason="late", started=True
    )
    batch.status = "analyzing"
    session.flush()
    generation = batch.lease_generation
    assert defer_analysis(
        session,
        batch_id=batch.id,
        lease_generation=generation,
        reason="memory_pressure",
        started=True,
    )
    session.refresh(batch)
    assert batch.status == "paused"
    assert batch.unmatched_instructions == []

    migration.downgrade()
    session.expire_all()
    session.refresh(batch)
    assert batch.status == "failed"
    assert batch.stage == "queued"
    assert batch.processed_instruction_indices == [2]


def test_stale_confirmation_cannot_finalize_an_attention_item(
    checkpoint_session: Session,
) -> None:
    from onyx.db.regulatory_amendments import persist_unmatched_checkpoint

    session = checkpoint_session
    file_id = session.scalar(select(RegulatoryChunk.user_file_id).limit(1))
    assert file_id is not None
    source = RegulatoryChunk(
        id=f"unmatched-{uuid4()}",
        user_file_id=file_id,
        text="Original evidence",
        position=0,
        projection_ordinal=uuid4().int % 700000000 + 100000000,
    )
    docset = DocumentSet(
        name=f"unmatched-{uuid4()}", description="evidence race", is_up_to_date=True
    )
    session.add_all([source, docset])
    session.flush()
    batch = AmendmentBatch(
        document_set_id=docset.id,
        raw_text="test",
        status="analyzing",
        stage="processing",
        lease_generation=1,
        instruction_count=1,
        user_file_ids=[str(file_id)],
    )
    session.add(batch)
    session.flush()
    evidence = capture_match_evidence(
        session,
        batch.user_file_ids,
        [
            CandidateChunk(
                chunk_id=source.id, user_file_id=str(file_id), text=source.text
            )
        ],
    )
    source.text = "Different evidence while model was running"
    session.flush()
    with pytest.raises(ValueError, match="evidence changed"):
        persist_unmatched_checkpoint(
            session,
            batch_id=batch.id,
            lease_generation=1,
            instruction_index=0,
            instruction_text="Attention: declined",
            expected_evidence=evidence,
        )
    assert batch.processed_instruction_count == 0
    assert batch.unmatched_instructions == []
