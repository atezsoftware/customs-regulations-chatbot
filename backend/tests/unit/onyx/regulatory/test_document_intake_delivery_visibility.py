import datetime
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db.document_intake import refresh_intake
from onyx.db.models import RegulatoryDocumentIntake
from onyx.regulatory.document_intake.models import INTAKE_DELIVERY_SECONDS


@pytest.mark.parametrize(
    ("delivery_overdue", "lease_seconds", "expected_status"),
    [
        (True, None, "failed"),
        (False, -60, "failed"),
        (True, 300, "preparing"),
        (False, None, "preparing"),
    ],
    ids=["expired-delivery", "dead-worker", "live-worker", "queued-delivery"],
)
def test_preparation_timeout_is_visible_without_stomping_live_work(
    delivery_overdue: bool,
    lease_seconds: int | None,
    expected_status: str,
) -> None:
    now = datetime.datetime.now(datetime.timezone.utc)
    lease_token = uuid4() if lease_seconds is not None else None
    lease_expires_at = (
        now + datetime.timedelta(seconds=lease_seconds)
        if lease_seconds is not None
        else None
    )
    batch = RegulatoryDocumentIntake(
        status="preparing",
        stage="chunking",
        updated_at=now
        - datetime.timedelta(
            seconds=INTAKE_DELIVERY_SECONDS + 60 if delivery_overdue else 60
        ),
        lease_token=lease_token,
        lease_expires_at=lease_expires_at,
        error=None,
    )
    session = MagicMock(spec=Session)

    refresh_intake(cast(Session, session), batch)

    assert batch.status == expected_status
    if expected_status == "failed":
        assert batch.error is not None
        assert "expired; retry" in batch.error
        assert batch.lease_token is None
        assert batch.lease_expires_at is None
    else:
        assert batch.error is None
        assert batch.lease_token == lease_token
        assert batch.lease_expires_at == lease_expires_at
    session.flush.assert_called_once_with()
    session.refresh.assert_called_once_with(batch, with_for_update=True)
    session.get.assert_not_called()
