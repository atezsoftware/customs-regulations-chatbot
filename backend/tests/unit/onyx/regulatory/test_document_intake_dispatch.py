from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, call
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.background.celery.tasks.regulatory_amendments.document_intake import (
    enqueue_intake_indexes,
)
from onyx.background.celery.versioned_apps import client
from onyx.configs.constants import (
    CELERY_REGULATORY_INDEXING_TASK_EXPIRES,
    CELERY_USER_FILE_PROCESSING_TASK_EXPIRES,
    OnyxCeleryPriority,
    OnyxCeleryQueues,
    OnyxCeleryTask,
)
from onyx.db import document_intake as repository
from onyx.db.enums import UserFileProjectionRepairStatus
from onyx.db.models import RegulatoryDocumentIntake


def test_main_and_annex_index_hints_survive_a_long_sibling_in_the_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deliveries = [(uuid4(), uuid4()), (uuid4(), uuid4())]
    publish = MagicMock()
    monkeypatch.setattr(client.app, "send_task", publish)

    enqueue_intake_indexes(deliveries, tenant_id="intake-tenant")

    assert publish.call_args_list == [
        call(
            OnyxCeleryTask.INDEX_SINGLE_USER_FILE,
            kwargs={
                "user_file_id": str(file_id),
                "tenant_id": "intake-tenant",
                "index_request_attempt_id": str(token),
            },
            queue=OnyxCeleryQueues.USER_FILE_PROCESSING,
            priority=OnyxCeleryPriority.HIGH,
            expires=1800,
        )
        for file_id, token in deliveries
    ]
    sibling_processing_seconds = 140
    assert all(
        invocation.kwargs["expires"] > sibling_processing_seconds
        for invocation in publish.call_args_list
    )
    assert CELERY_REGULATORY_INDEXING_TASK_EXPIRES == 1800
    assert CELERY_USER_FILE_PROCESSING_TASK_EXPIRES == 60


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
    assert batch.status == "preparing"
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
