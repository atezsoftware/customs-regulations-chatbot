"""Explicit, complete user-file context for the two session attachment workflows."""

import json
from collections.abc import Callable

from onyx.chat.models import ChatLoadedFile
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.prompts.asv3.attachments import SESSION_ATTACHMENT_CONTEXT
from onyx.server.query_and_chat.models import SendMessageRequest


def supports_asv3_session_attachments(request: SendMessageRequest) -> bool:
    return (
        request.atez_search_v3
        and request.asv3_research_profile == "normal"
        and not any(
            (
                request.asv3_parallel_research,
                request.asv3_guarded_experimental,
                request.asv3_guardrails_v2,
                request.asv3_guardrails_v3,
                request.asv3_fast,
                request.asv3_fast_cropped,
                request.asv3_kompozit,
                request.asv3_kompozit_cropped,
            )
        )
    )


def build_asv3_attachment_context(
    files: list[ChatLoadedFile],
    token_counter: Callable[[str], int],
    max_input_tokens: int,
) -> str:
    if not files:
        return ""
    documents = []
    for file in files:
        if not file.content_text or not file.content_text.strip():
            raise OnyxError(
                OnyxErrorCode.INVALID_INPUT,
                f"Attachment text is unavailable: {file.filename or file.file_id}",
            )
        documents.append(
            {
                "file_id": file.file_id,
                "filename": file.filename,
                "text": file.content_text,
            }
        )
    context = SESSION_ATTACHMENT_CONTEXT + json.dumps(documents, ensure_ascii=False)
    budget = min(64_000, max_input_tokens // 2)
    if token_counter(context) > budget:
        raise OnyxError(
            OnyxErrorCode.PAYLOAD_TOO_LARGE,
            "Session attachments exceed the selected model's attachment context "
            f"budget ({budget} tokens). Remove files or select a larger-context model; "
            "no attachment text has been truncated.",
        )
    return context
