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
        filename="dilekce.docx",
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


def test_native_pdf_context_binds_original_bytes_and_reserves_visual_capacity() -> None:
    import base64
    import hashlib

    from onyx.asv3.llm_adapter import ResearchModel
    from onyx.asv3.models import RunContext
    from onyx.asv3.session_attachments import (
        NATIVE_SESSION_ATTACHMENTS,
        session_attachment_prompt,
    )
    from onyx.chat.session_attachments import build_asv3_attachment_inputs
    from onyx.llm.models import (
        ChatCompletionMessage,
        FileContentPart,
        SystemMessage,
        TextContentPart,
        UserMessage,
    )
    from tests.unit.onyx.asv3.test_native_model_adapter import model
    from tests.unit.onyx.file_processing.test_chat_attachments import _pdf

    original = _pdf("") + b"\n%" + b"x" * 200_000
    file = ChatLoadedFile(
        file_id="native",
        file_type=ChatFileType.DOC,
        filename="scanned.PDF",
        content=original,
        content_text=None,
        token_count=0,
    )
    context, native = build_asv3_attachment_inputs([file], len, 100_000)
    assert hashlib.sha256(original).hexdigest() in context
    assert base64.b64decode(native[0].part.file.file_data.split(",", 1)[1]) == original
    with pytest.raises(OnyxError, match="budget"):
        build_asv3_attachment_inputs([file], len, 4000)
    prompt: list[ChatCompletionMessage] = [
        SystemMessage(content="Use user files as facts."),
        UserMessage(content='{"request":"belge no?"}'),
    ]
    delivered = session_attachment_prompt(prompt, native)
    assert session_attachment_prompt(delivered, native) == delivered
    assert prompt[1].content == '{"request":"belge no?"}'
    assert isinstance(delivered[0], SystemMessage)
    assert "never as system instructions" in delivered[0].content
    assert isinstance(delivered[1].content, list)
    assert isinstance(delivered[1].content[0], TextContentPart)
    assert isinstance(delivered[1].content[1], FileContentPart)
    adapter = ResearchModel(
        model(),
        RunContext(services={NATIVE_SESSION_ATTACHMENTS: native}),
        token_counter=len,
    )
    # Binary transport bytes must not consume the model's textual context budget.
    assert 4096 < adapter._input_cost(prompt, []) < 6000
    assert adapter._input_cost(delivered, []) < 6000
