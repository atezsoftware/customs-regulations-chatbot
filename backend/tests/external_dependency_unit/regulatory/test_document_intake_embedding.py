"""Only a current approved initial intake writer may batch frozen embeddings."""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import document_intake_embedding as policy
from onyx.db.enums import UserFileProjectionRepairStatus
from onyx.db.models import (
    DocumentSet,
    DocumentSet__UserFile,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryFilePublication,
    RegulatoryTemporalProjection,
    User,
    UserFile,
    UserFileProjectionRepair,
)
from onyx.db.regulatory_writer_publication import canonical_scope_digest
from onyx.db.user_file import start_user_file_index_request
from onyx.document_index.publication_models import (
    FileOwnership,
    PublicationScope,
    publication_digest,
)
from onyx.regulatory.amendments.annexes import config
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_owner as intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_session as intake_session,
)
from tests.external_dependency_unit.regulatory.test_document_intake_index_retry import (
    _failed_initial_publication,
)


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "ordinary",
        "pending",
        "foreign-token",
        "owner",
        "membership",
        "published",
        "history",
        "canonical",
        "review",
        "lease",
        "gate",
    ],
)
def test_initial_intake_embedding_policy_is_fresh_scoped_and_read_only(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    session = intake_session
    batch, members, _ = _failed_initial_publication(session, intake_owner)
    tokens = dict(intake.retry_intake(session, batch))
    member = members[0]
    token = tokens[member.user_file_id]
    assert start_user_file_index_request(session, member.user_file_id, token)
    scope = PublicationScope(
        tenant_id="public",
        environment=config.REGULATORY_ANNEX_ENVIRONMENT,
        database_identity=config.ANNEX_DATABASE_IDENTITY,
    )
    publication = session.get_one(RegulatoryFilePublication, member.user_file_id)
    publication.scope_key = publication_digest(scope.model_dump(mode="json"))
    publication.owner_id = uuid4()
    publication.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    owner = FileOwnership(
        scope=scope,
        user_file_id=member.user_file_id,
        owner_id=publication.owner_id,
        fencing_token=publication.fencing_token,
        expires_at=publication.lease_expires_at,
    )
    original = canonical_scope_digest(session, member.user_file_id)
    file = session.get_one(UserFile, member.user_file_id)
    request = session.get_one(UserFileProjectionRepair, member.user_file_id)
    if change == "ordinary":
        batch.status = "completed"
        batch.stage = "completed"
    elif change == "pending":
        request.status = UserFileProjectionRepairStatus.PENDING
    elif change == "foreign-token":
        request.attempt_id = uuid4()
    elif change == "owner":
        file.user_id = uuid4()
    elif change == "membership":
        session.execute(
            delete(DocumentSet__UserFile).where(
                DocumentSet__UserFile.user_file_id == file.id
            )
        )
    elif change == "published":
        publication.epoch = 1
    elif change == "history":
        chunk = session.scalar(
            select(RegulatoryChunk).where(RegulatoryChunk.user_file_id == file.id)
        )
        assert chunk is not None
        session.add(
            RegulatoryTemporalProjection(
                id=uuid4(),
                user_file_id=file.id,
                canonical_chunk_id=chunk.id,
                index_uuid="prior-index",
                index_identity_sha256="a" * 64,
                projection_ordinal=0,
                payload={},
                payload_sha256="b" * 64,
            )
        )
    elif change == "canonical":
        chunk = session.scalar(
            select(RegulatoryChunk).where(RegulatoryChunk.user_file_id == file.id)
        )
        assert chunk is not None
        chunk.text += " changed"
    elif change == "review":
        batch.review_sha256 = "f" * 64
    elif change == "lease":
        publication.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    elif change == "gate":
        publication.gate_closed = True
    if change == "owner":
        # Keep the foreign identity relationally valid without assigning a nonexistent user.
        file.user_id = None
    session.flush()
    before = (
        publication.epoch,
        publication.gate_closed,
        publication.fencing_token,
        publication.lease_expires_at,
        request.attempt_id,
        request.status,
        canonical_scope_digest(session, file.id),
    )

    @contextmanager
    def scoped_session(*, tenant_id: str) -> Iterator[Session]:
        assert tenant_id == "public"
        yield session

    monkeypatch.setattr(policy, "get_session_with_tenant", scoped_session)
    if change == "lease":
        with pytest.raises(ValueError):
            policy.current_intake_embedding_scope(
                owner, token, canonical_sha256=original
            )
    else:
        assert policy.current_intake_embedding_scope(
            owner, token, canonical_sha256=original
        ) == (change == "none")
    assert before == (
        publication.epoch,
        publication.gate_closed,
        publication.fencing_token,
        publication.lease_expires_at,
        request.attempt_id,
        request.status,
        canonical_scope_digest(session, file.id),
    )
    assert not session.dirty and not session.new
