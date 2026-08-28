from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from onyx.db.enums import AmendmentBatchStatus
from onyx.db.models import User
from onyx.error_handling.exceptions import OnyxError
from onyx.server.features.regulatory import api
from onyx.server.features.regulatory.models import (
    AnalyzeAmendmentRequest,
    ApproveAmendmentProposalRequest,
)


def test_analyze_queues_batch_without_invoking_llm() -> None:
    user_file_id = uuid4()
    document_set = SimpleNamespace(
        id=7,
        user_files=[SimpleNamespace(id=user_file_id)],
    )
    user = cast(User, SimpleNamespace(id=uuid4()))
    batch = SimpleNamespace(
        id=42,
        document_set_id=7,
        raw_text="MADDE 1 değiştirilmiştir.",
        reference_date=None,
        status=AmendmentBatchStatus.QUEUED.value,
        stage="queued",
        instruction_count=0,
        processed_instruction_count=0,
        error_message=None,
        created_by=user.id,
        created_at=MagicMock(),
        updated_at=MagicMock(),
        started_at=None,
        heartbeat_at=None,
        completed_at=None,
    )
    db_session = MagicMock()

    with (
        patch.object(api, "_get_editable_document_set", return_value=document_set),
        patch.object(api, "create_batch", return_value=batch),
        patch.object(api, "enqueue_amendment_batch") as enqueue,
    ):
        snapshot = api.analyze_amendment_text(
            AnalyzeAmendmentRequest(
                document_set_id=7,
                raw_text="MADDE 1 değiştirilmiştir.",
            ),
            user=user,
            db_session=db_session,
            tenant_id="public",
        )

    assert snapshot.id == 42
    assert snapshot.status == "queued"
    enqueue.assert_called_once_with(batch_id=42, tenant_id="public")
    db_session.commit.assert_called_once_with()


def test_pdf_extracted_and_pasted_text_share_analysis_request_contract() -> None:
    pdf_text = AnalyzeAmendmentRequest(document_set_id=7, raw_text="PDF MADDE 1")
    pasted_text = AnalyzeAmendmentRequest(document_set_id=7, raw_text="TEXT MADDE 1")

    assert pdf_text.raw_text == "PDF MADDE 1"
    assert pasted_text.raw_text == "TEXT MADDE 1"


def test_proposal_cannot_be_approved_before_batch_finishes() -> None:
    proposal = SimpleNamespace(batch_id=42)
    batch = SimpleNamespace(
        id=42,
        document_set_id=7,
        status=AmendmentBatchStatus.ANALYZING.value,
    )
    user = cast(User, SimpleNamespace(id=uuid4()))

    with (
        patch.object(api, "get_proposal", return_value=proposal),
        patch.object(api, "get_batch", return_value=batch),
        patch.object(api, "_get_editable_document_set"),
        patch.object(api, "approve_amendment_proposal") as approve,
        pytest.raises(OnyxError, match="after analysis is complete"),
    ):
        api.approve_proposal(9, user=user, db_session=MagicMock())

    approve.assert_not_called()


def test_approval_projects_only_the_affected_user_file_before_commit() -> None:
    user_file_id = uuid4()
    proposal = SimpleNamespace(batch_id=42)
    batch = SimpleNamespace(
        id=42,
        document_set_id=7,
        status=AmendmentBatchStatus.ANALYZED.value,
    )
    user = cast(User, SimpleNamespace(id=uuid4()))
    affected_user_file = SimpleNamespace(id=user_file_id)
    approval_result = SimpleNamespace(
        new_chunk=SimpleNamespace(user_file_id=user_file_id)
    )
    snapshot = MagicMock()
    db_session = MagicMock()
    db_session.get.return_value = affected_user_file
    events: list[str] = []
    db_session.commit.side_effect = lambda: events.append("commit")
    reviewed_draft = {
        "user_file_id": str(user_file_id),
        "position": 15,
        "text": "MADDE 15 - (2) a) (Mülga)",
    }
    approval_request = ApproveAmendmentProposalRequest(new_chunk_draft=reviewed_draft)

    with (
        patch.object(api, "get_proposal", return_value=proposal),
        patch.object(api, "get_batch", return_value=batch),
        patch.object(api, "_get_editable_document_set"),
        patch.object(
            api, "approve_amendment_proposal", return_value=approval_result
        ) as approve_amendment,
        patch.object(api, "project_user_file_to_index") as project_file,
        patch.object(api, "get_current_tenant_id", return_value="tenant-a"),
        patch.object(
            api.AmendmentProposalSnapshot, "from_model", return_value=snapshot
        ),
    ):
        project_file.side_effect = lambda *_args: events.append("project")
        result = api.approve_proposal(
            9,
            approval_request=approval_request,
            user=user,
            db_session=db_session,
        )

    assert result is snapshot
    approve_amendment.assert_called_once_with(
        db_session,
        proposal,
        decided_by=user.id,
        reviewed_new_chunk_draft=reviewed_draft,
    )
    db_session.get.assert_called_once_with(api.UserFile, user_file_id)
    project_file.assert_called_once_with(db_session, affected_user_file, "tenant-a")
    db_session.commit.assert_called_once_with()
    assert events == ["project", "commit"]
