"""Real separate-session review transitions with observed PostgreSQL lock waits."""

import hashlib
import json
import time
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier, Event

import pytest
from sqlalchemy import event, select, text
from sqlalchemy.orm import Session

from onyx.db.engine.sql_engine import get_sqlalchemy_engine
from onyx.db.models import AmendmentBatch, AnnexChangeSet, AnnexPublicationIntent
from onyx.db.regulatory_annex_changes import (
    capture_canonical_scope,
    create_source_text_revision,
    list_annex_review_revisions,
    queue_annex_publication,
    revise_annex_review,
)
from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft
from tests.external_dependency_unit.regulatory.test_annex_review_revisions import (
    LiveReview,
)
from tests.external_dependency_unit.regulatory.test_annex_review_revisions import (
    es as es,
)
from tests.external_dependency_unit.regulatory.test_annex_review_revisions import (
    live_review as live_review,
)
from tests.external_dependency_unit.regulatory.test_annex_review_revisions import (
    source_session as source_session,
)


@pytest.mark.parametrize("other", ["edit", "source"])
@pytest.mark.parametrize("winner", ["approve", "other"])
def test_review_transition_waits_and_rechecks_stale_state(
    other: str,
    winner: str,
    live_review: LiveReview,
    source_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.file_store.file_store import get_default_file_store

    @contextmanager
    def independent_read_session() -> Generator[Session, None, None]:
        with Session(get_sqlalchemy_engine()) as session:
            yield session

    # The reusable preparation fixture uses one session; evidence readers in the
    # concurrent phase must have their own connections as in the application.
    monkeypatch.setattr(
        "onyx.db.engine.sql_engine.get_session_with_current_tenant",
        independent_read_session,
    )
    batch_id, review_id = live_review.batch.id, live_review.review.id
    review_hash = live_review.review.review_sha256
    old_payload = json.dumps(live_review.review.review_payload, sort_keys=True)
    old_text = live_review.batch.raw_text
    source_hash = hashlib.sha256(old_text.encode()).hexdigest()
    owner_id = live_review.file.user_id
    draft = AnnexChangeDraft.model_validate(live_review.review.review_payload)
    store = get_default_file_store()
    original_id = live_review.file.file_id
    original_bytes = store.read_file(original_id, mode="b").read()
    source_session.commit()
    preloaded = Barrier(3)
    winner_locked, allow_commit = Event(), Event()
    pids: dict[str, int] = {}
    winning_operation = "approve" if winner == "approve" else other
    losing_operation = other if winner == "approve" else "approve"

    def transition(operation: str) -> str:
        with Session(get_sqlalchemy_engine(), expire_on_commit=False) as session:
            old_batch = session.get(AmendmentBatch, batch_id)
            old_review = session.get(AnnexChangeSet, review_id)
            assert old_batch is not None and old_review is not None
            assert old_batch.superseded_by_batch_id is None
            assert old_review.status == "pending"
            pids[operation] = session.scalar(text("SELECT pg_backend_pid()"))
            preloaded.wait(timeout=15)

            def pause_commit(_session: Session) -> None:
                winner_locked.set()
                assert allow_commit.wait(timeout=15), "winner commit not released"

            if operation == winning_operation:
                event.listen(session, "before_commit", pause_commit, once=True)
            else:
                assert winner_locked.wait(timeout=15), "winner did not hold batch lock"
            try:
                if operation == "approve":
                    queue_annex_publication(
                        session,
                        change_set_id=review_id,
                        expected_review_sha256=review_hash,
                        environment="local-test",
                        tenant_id="public",
                        database_identity="local-db",
                        decided_by=owner_id,
                    )
                elif operation == "edit":
                    revise_annex_review(
                        session,
                        change_set_id=review_id,
                        expected_review_sha256=review_hash,
                        draft=draft.model_copy(
                            update={"issues": ["human edit requires revalidation"]}
                        ),
                        environment="local-test",
                    )
                else:
                    create_source_text_revision(
                        session,
                        batch_id=batch_id,
                        raw_text=old_text + "\nRevised fictional source instruction.",
                        expected_source_text_sha256=source_hash,
                        environment="local-test",
                        created_by=owner_id,
                    )
                return "committed"
            except ValueError as exc:
                session.rollback()
                return str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(transition, winning_operation)
        second = pool.submit(transition, losing_operation)
        try:
            preloaded.wait(timeout=15)
            assert winner_locked.wait(timeout=15), "winner did not reach commit barrier"
            deadline = time.monotonic() + 10
            blockers: list[int] = []
            while time.monotonic() < deadline:
                blockers = source_session.scalar(
                    text("SELECT pg_blocking_pids(:pid)"),
                    {"pid": pids[losing_operation]},
                )
                if pids[winning_operation] in blockers:
                    break
                time.sleep(0.01)
            assert pids[winning_operation] in blockers, (
                "loser never waited on winner's PostgreSQL lock"
            )
        finally:
            allow_commit.set()
        first_result, second_result = (
            first.result(timeout=15),
            second.result(timeout=15),
        )
    print(
        json.dumps(
            {
                "winner": winning_operation,
                "loser": losing_operation,
                "winner_pid": pids[winning_operation],
                "loser_pid": pids[losing_operation],
                "observed_blockers": blockers,
                "winner_result": first_result,
                "loser_result": second_result,
            }
        )
    )
    assert first_result == "committed"
    expected_error = {
        ("edit", "other"): "stale review revision or hash",
        ("edit", "approve"): "review state does not allow edits",
        ("source", "other"): "stale source text revision",
        ("source", "approve"): "publication state prevents source edits",
    }[(other, winner)]
    assert second_result == expected_error
    source_session.expire_all()
    original_review = source_session.get(AnnexChangeSet, review_id)
    original_batch = source_session.get(AmendmentBatch, batch_id)
    assert original_review is not None and original_batch is not None
    assert original_review.review_sha256 == review_hash
    assert json.dumps(original_review.review_payload, sort_keys=True) == old_payload
    assert original_batch.raw_text == old_text
    assert store.read_file(original_id, mode="b").read() == original_bytes
    revisions = list_annex_review_revisions(
        source_session, logical_group_id=original_review.logical_group_id
    )
    assert len(revisions) == (2 if winning_operation == "edit" else 1)
    intents = list(
        source_session.scalars(
            select(AnnexPublicationIntent).where(
                AnnexPublicationIntent.change_set_id == review_id
            )
        )
    )
    assert len(intents) == (1 if winning_operation == "approve" else 0)
    assert (original_batch.superseded_by_batch_id is not None) == (
        winning_operation == "source"
    )
    assert (
        capture_canonical_scope(source_session, live_review.file.id)
        == live_review.before
    )
    print(
        json.dumps(
            {
                "winner": winning_operation,
                "loser": losing_operation,
                "winner_pid": pids[winning_operation],
                "loser_pid": pids[losing_operation],
                "observed_blockers": blockers,
                "loser_result": second_result,
                "review_revision_count": len(revisions),
                "publication_intent_count": len(intents),
                "immutable_old_bytes": True,
            }
        )
    )
