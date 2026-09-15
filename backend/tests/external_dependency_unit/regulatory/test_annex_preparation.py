from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db.models import (
    AmendmentBatch,
    AmendmentProposal,
    AnnexChangeSet,
    AnnexReviewPreparation,
    DocumentSet,
    RegulatoryChunk,
    UserFile,
)
from onyx.db.regulatory_annex_changes import (
    persist_annex_checkpoint,
    require_current_annex_review,
    revise_annex_review,
)
from onyx.db.regulatory_annex_preparation import (
    claim_review_preparation,
    queue_review_preparation,
)
from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft
from tests.external_dependency_unit.conftest import create_test_user
from tests.external_dependency_unit.regulatory.test_amendment_sources import (
    source_session as source_session,
)


def test_review_preparation_is_idempotent_fenced_and_preserves_previous_revision(
    source_session: Session,
) -> None:
    user = create_test_user(source_session, "review_preparation")
    docset = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    source_session.add(docset)
    source_session.flush()
    batch = AmendmentBatch(
        document_set_id=docset.id,
        raw_text="EK-1",
        status="analyzing",
        lease_generation=1,
        instruction_count=1,
    )
    source_session.add(batch)
    source_session.flush()
    draft = AnnexChangeDraft(
        instruction_indices=[0],
        instruction_texts=["EK-1"],
        annex_label="EK-1",
        issues=["missing_original"],
    )
    review = persist_annex_checkpoint(
        source_session,
        batch_id=batch.id,
        lease_generation=1,
        draft=draft,
        environment="test",
    )
    assert review is not None
    frozen = dict(review.review_payload)

    def queue() -> AnnexChangeSet:
        return queue_review_preparation(
            source_session,
            review_id=review.id,
            expected_review_sha256=review.review_sha256,
            corrections=None,
            corrected_by=user.id,
            tenant_id="public",
            environment="test",
            database_identity="test",
        )

    queued = queue()
    again = queue()
    assert queued.id == again.id and queued.preparation is not None
    assert queued.preparation.status == "queued"
    from onyx.server.features.regulatory.models import AnnexReviewSnapshot

    response = AnnexReviewSnapshot.model_validate(queued).model_dump(mode="json")
    assert response["preparation"]["status"] == "queued"
    assert "baseline_context" not in response["review_payload"]
    assert "baseline_scope" not in response["review_payload"]
    assert response["review_sha256"] == queued.review_sha256
    with pytest.raises(ValueError, match="still running"):
        require_current_annex_review(
            source_session,
            change_set_id=review.id,
            expected_review_sha256=review.review_sha256,
            environment="test",
        )
    source_session.rollback()

    def claim() -> AnnexReviewPreparation | None:
        return claim_review_preparation(
            source_session,
            review_id=review.id,
            tenant_id="public",
            environment="test",
            database_identity="test",
        )

    claimed = claim()
    assert claimed is not None
    assert claim() is None
    persisted = source_session.get(AnnexReviewPreparation, review.id)
    assert persisted is not None
    persisted.checkpoint = draft.model_dump(mode="json")
    persisted.heartbeat_at = datetime.now(timezone.utc) - timedelta(minutes=11)
    source_session.commit()
    reclaimed = claim()
    assert reclaimed is not None and reclaimed.generation > claimed.generation
    assert reclaimed.checkpoint == draft.model_dump(mode="json")
    with pytest.raises(ValueError, match="lease lost"):
        revise_annex_review(
            source_session,
            change_set_id=review.id,
            expected_review_sha256=review.review_sha256,
            environment="test",
            draft=draft,
            preparation_generation=claimed.generation,
        )
    source_session.rollback()
    revised = revise_annex_review(
        source_session,
        change_set_id=review.id,
        expected_review_sha256=review.review_sha256,
        environment="test",
        draft=draft.model_copy(update={"issues": ["different_check"]}),
        preparation_generation=reclaimed.generation,
    )
    source_session.refresh(review)
    assert review.review_payload == frozen
    assert review.preparation is not None and review.preparation.status == "completed"
    assert review.preparation.result_review_id == revised.id
    assert revised.review_revision == 2 and revised.publication_generation == 0


@pytest.mark.parametrize("change_child", [False, True])
def test_full_provision_replacement_includes_all_descendants_atomically(
    source_session: Session, change_child: bool
) -> None:
    from onyx.db.enums import UserFileStatus
    from onyx.db.regulatory_amendments import approve_amendment_proposal
    from onyx.db.regulatory_chunks import load_active_structural_descendants
    from onyx.regulatory.amendments.pipeline import _chunk_to_review_dict

    user = create_test_user(source_session, "scope_replacement")
    docset = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    file = UserFile(
        id=uuid4(),
        user_id=user.id,
        file_id=str(uuid4()),
        name="TIR.md",
        file_type="text/markdown",
        status=UserFileStatus.COMPLETED,
    )
    source_session.add_all([docset, file])
    source_session.flush()
    rows = [
        RegulatoryChunk(
            id=f"rc_{uuid4().hex}",
            user_file_id=file.id,
            position=i,
            projection_ordinal=i,
            text=text,
            chunk_type=kind,
            heading_path=path,
            chunk_metadata={"article_no": "20", "paragraph_no": paragraph},
            status="active",
            source="indexed",
        )
        for i, (text, kind, paragraph, path) in enumerate(
            [
                ("(3) Eski yöntemler:", "paragraph", "3", ["MADDE 20", "(3)"]),
                ("a) Eski ilk yöntem", "clause", "3", ["MADDE 20", "(3)", "a)"]),
                ("b) Eski ikinci yöntem", "clause", "3", ["MADDE 20", "(3)", "b)"]),
                ("(4) İlgisiz hüküm", "paragraph", "4", ["MADDE 20", "(4)"]),
            ]
        )
    ]
    source_session.add_all(rows)
    source_session.flush()
    assert load_active_structural_descendants(source_session, rows[0]) == rows[1:3]
    body = "(3) Yeni yöntemler: a) Yeni ilk yöntem. b) Yeni ikinci yöntem."
    instruction = (
        f"20 nci maddenin üçüncü fıkrası aşağıdaki şekilde değiştirilmiştir: “{body}”"
    )
    batch = AmendmentBatch(
        document_set_id=docset.id,
        raw_text=instruction,
        status="analyzed",
        user_file_ids=[str(file.id)],
    )
    source_session.add(batch)
    source_session.flush()
    snapshot = _chunk_to_review_dict(rows[0])
    snapshot["descendant_snapshots"] = [_chunk_to_review_dict(row) for row in rows[1:3]]
    proposal = AmendmentProposal(
        batch_id=batch.id,
        instruction_index=0,
        instruction_indices=[0],
        instruction_text=instruction,
        instruction_texts=[instruction],
        old_chunk_id=rows[0].id,
        old_chunk_snapshot=snapshot,
        status="approving",
        new_chunk_draft={
            "user_file_id": str(file.id),
            "position": 0,
            "text": body,
            "chunk_type": "paragraph",
            "heading_path": ["MADDE 20", "(3)"],
            "metadata": {"article_no": "20", "paragraph_no": "3"},
            "effective_start_date": "2026-07-04",
        },
    )
    source_session.add(proposal)
    source_session.commit()
    if change_child:
        rows[2].text = "Someone edited this child after review"
        source_session.commit()
        with pytest.raises(ValueError, match="changed after analysis"):
            approve_amendment_proposal(source_session, proposal)
        source_session.rollback()
        assert all(row.status == "active" for row in rows)
        assert proposal.applied_new_chunk_id is None
    else:
        result = approve_amendment_proposal(source_session, proposal)
        source_session.commit()
        assert result.new_chunk.text == body
        assert all(
            row.status == "superseded"
            and row.superseded_by_chunk_id == result.new_chunk.id
            for row in rows[:3]
        )
        assert rows[3].status == "active"
        assert (
            approve_amendment_proposal(source_session, proposal).new_chunk.id
            == result.new_chunk.id
        )


def test_retry_attention_preserves_existing_proposals(source_session: Session) -> None:
    from onyx.db.regulatory_amendments import reset_batch_attention_for_retry

    docset = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    source_session.add(docset)
    source_session.flush()
    batch = AmendmentBatch(
        document_set_id=docset.id,
        raw_text="done; unresolved",
        status="analyzed",
        instruction_count=2,
        processed_instruction_count=2,
        processed_instruction_indices=[0, 1],
        segmented_instructions=[
            {"instruction_text": "done"},
            {"instruction_text": "unresolved"},
        ],
        unmatched_instructions=["unresolved\n\nAttention: previous guard"],
    )
    source_session.add(batch)
    source_session.flush()
    proposal = AmendmentProposal(
        batch_id=batch.id,
        instruction_index=0,
        instruction_indices=[0],
        instruction_text="done",
        instruction_texts=["done"],
        old_chunk_snapshot={},
        new_chunk_draft={},
        status="pending",
    )
    source_session.add(proposal)
    source_session.commit()
    retried = reset_batch_attention_for_retry(source_session, batch_id=batch.id)
    assert retried is not None and retried.status == "queued"
    assert retried.processed_instruction_indices == [0]
    assert retried.processed_instruction_count == 1
    persisted_proposal = source_session.get(AmendmentProposal, proposal.id)
    assert persisted_proposal is not None and persisted_proposal.status == "pending"
    assert retried.unmatched_instructions == []


def test_context_checkpoint_reuses_exact_inputs_without_index_writes(
    source_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from collections.abc import Iterator
    from contextlib import contextmanager
    from unittest.mock import MagicMock

    from onyx.db.enums import UserFileStatus
    from onyx.db.regulatory_context_projections import resolve_preparation_context_call
    from onyx.regulatory.amendments.annexes.models import ContextGenerationCall

    user = create_test_user(source_session, "context_checkpoint")
    file = UserFile(
        id=uuid4(),
        user_id=user.id,
        file_id=str(uuid4()),
        name="context.md",
        file_type="text/markdown",
        status=UserFileStatus.COMPLETED,
    )
    source_session.add(file)
    source_session.commit()

    @contextmanager
    def session_scope() -> Iterator[Session]:
        yield source_session

    monkeypatch.setattr(
        "onyx.db.engine.sql_engine.get_session_with_current_tenant", session_scope
    )
    call = ContextGenerationCall(
        request_sha256="a" * 64,
        stage="durable_chunk",
        prompt_json='{"prompt":"scope"}',
        config_sha256="b" * 64,
        output="",
        source_text="scope",
        token_budget=1000,
        tokenizer="test",
        generation_path="durable",
    )
    generate = MagicMock(return_value="Generated context")
    first = resolve_preparation_context_call(
        user_file_id=file.id, call=call, generate=generate
    )
    same = resolve_preparation_context_call(
        user_file_id=file.id, call=call, generate=generate
    )
    assert first == same and first.output == "Generated context"
    generate.assert_called_once()
    with pytest.raises(ValueError, match="input proof changed"):
        resolve_preparation_context_call(
            user_file_id=file.id,
            call=call.model_copy(update={"source_text": "different"}),
            generate=generate,
        )
