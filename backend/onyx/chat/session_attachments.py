"""Explicit, complete user-file context for the two session attachment workflows."""

import base64
import hashlib
import json
from collections.abc import Callable
from pathlib import PurePosixPath

from onyx.asv3.session_attachments import NativeSessionAttachment
from onyx.chat.models import ChatLoadedFile
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.file_processing.chat_attachments import validate_chat_attachment_pdf
from onyx.llm.models import FileContentPart, FileDetail
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
    return build_asv3_attachment_inputs(files, token_counter, max_input_tokens)[0]


def build_asv3_attachment_inputs(
    files: list[ChatLoadedFile],
    token_counter: Callable[[str], int],
    max_input_tokens: int,
) -> tuple[str, list[NativeSessionAttachment]]:
    if not files:
        return "", []
    documents = []
    native: list[NativeSessionAttachment] = []
    for file in files:
        if PurePosixPath(file.filename or "").suffix.lower() == ".pdf":
            content = file.content
            try:
                page_count = validate_chat_attachment_pdf(content)
            except ValueError as error:
                raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(error)) from error
            digest = hashlib.sha256(content).hexdigest()
            native.append(
                NativeSessionAttachment(
                    file_id=file.file_id,
                    filename=file.filename or "attachment.pdf",
                    sha256=digest,
                    page_count=page_count,
                    part=FileContentPart(
                        file=FileDetail(
                            filename=file.filename or "attachment.pdf",
                            file_data="data:application/pdf;base64,"
                            + base64.b64encode(content).decode("ascii"),
                        )
                    ),
                )
            )
            documents.append(
                {
                    "file_id": file.file_id,
                    "filename": file.filename,
                    "delivery": "native_pdf",
                    "page_count": page_count,
                    "sha256": digest,
                }
            )
            continue
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
    if (
        token_counter(context) + sum(item.input_token_reserve for item in native)
        > budget
    ):
        raise OnyxError(
            OnyxErrorCode.PAYLOAD_TOO_LARGE,
            "Session attachments exceed the selected model's attachment context "
            f"budget ({budget} tokens). Remove files or select a larger-context model; "
            "no attachment text has been truncated.",
        )
    return context, native
