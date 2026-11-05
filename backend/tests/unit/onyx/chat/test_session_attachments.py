from uuid import uuid4

import pytest

from onyx.chat.models import ChatLoadedFile
from onyx.chat.session_attachments import (
    build_asv3_attachment_context,
    supports_asv3_session_attachments,
)
from onyx.error_handling.exceptions import OnyxError
from onyx.file_store.models import ChatFileType
from onyx.server.query_and_chat.models import SendMessageRequest


def test_attachments_only_enable_for_tek_and_exp_cropped() -> None:
    request = SendMessageRequest(
        message="Soruyu incele",
        chat_session_id=uuid4(),
        atez_search_v3=True,
        asv3_research_profile="normal",
    )
    assert supports_asv3_session_attachments(request)
    assert supports_asv3_session_attachments(
        request.model_copy(update={"asv3_exp_cropped": True})
    )
    for update in (
        {"atez_search_v3": False},
        {"asv3_research_profile": "deep"},
        {"asv3_fast": True},
        {"asv3_kompozit_cropped": True},
        {"asv3_guardrails_v3": True},
    ):
        assert not supports_asv3_session_attachments(request.model_copy(update=update))


def test_attachment_context_rejects_missing_text_and_over_budget_without_truncation() -> (
    None
):
    file = ChatLoadedFile(
        file_id="private-id",
        file_type=ChatFileType.DOC,
        filename="dilekce.pdf",
        content=b"",
        content_text=None,
        token_count=0,
    )
    with pytest.raises(OnyxError, match="unavailable"):
        build_asv3_attachment_context([file], len, 100_000)
    file.content_text = "full content" * 100
    with pytest.raises(OnyxError, match="no attachment text has been truncated"):
        build_asv3_attachment_context([file], len, 100)
    assert file.content_text == "full content" * 100
