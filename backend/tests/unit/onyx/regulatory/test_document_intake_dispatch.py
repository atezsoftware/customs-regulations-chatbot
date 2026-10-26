from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db import document_intake as repository
from onyx.db.enums import UserFileProjectionRepairStatus
from onyx.db.models import RegulatoryDocumentIntake


@pytest.mark.parametrize("claimed", [False, True])
def test_preparation_dispatch_failure_preserves_a_live_worker_claim(
    claimed: bool,
) -> None:
    batch = cast(
        RegulatoryDocumentIntake,
        SimpleNamespace(
            id=uuid4(),
            status="preparing",
            stage="chunking",
            lease_token=uuid4() if claimed else None,
            error=None,
        ),
    )
    session = MagicMock(spec=Session)
    session.get.return_value = batch
    repository.mark_intake_dispatch_failed(session, batch, "Queue unavailable")
    assert batch.status == ("preparing" if claimed else "failed")
    assert batch.error == (None if claimed else "Queue unavailable")
    if claimed:
        assert batch.lease_token is not None


@pytest.mark.parametrize(
    "status",
    [UserFileProjectionRepairStatus.PENDING, UserFileProjectionRepairStatus.RUNNING],
)
def test_index_dispatch_failure_only_flags_undelivered_attempts(
    monkeypatch: pytest.MonkeyPatch, status: UserFileProjectionRepairStatus
) -> None:
    attempt_id = uuid4()
    batch = cast(
        RegulatoryDocumentIntake,
        SimpleNamespace(id=uuid4(), status="approving", stage="indexing", error=None),
    )
    request = SimpleNamespace(attempt_id=attempt_id, status=status, initial_index=True)
    member = SimpleNamespace(user_file_id=uuid4(), approval_attempt_id=attempt_id)
    session = MagicMock(spec=Session)
    session.get.side_effect = [batch, request]
    monkeypatch.setattr(repository, "intake_files", lambda *_args: [member])
    repository.mark_intake_index_dispatch_failed(session, batch, "Queue unavailable")
    assert batch.status == (
        "failed" if status is UserFileProjectionRepairStatus.PENDING else "approving"
    )
    assert request.status == status
